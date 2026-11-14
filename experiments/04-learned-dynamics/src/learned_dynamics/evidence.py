"""Immutable Stage 4 records and an inference-free independent semantic audit."""

from collections import defaultdict
from dataclasses import asdict
import hashlib
import os
from pathlib import Path
import tomllib

from forward_dynamics.dataset import PredictionCase, load_prediction_cases
from forward_dynamics.metrics import residual
from forward_dynamics.serialization import (
    decode_object, json_bytes, require_equal, require_fields, sha256_file,
)
from forward_dynamics.types import DynamicsState
from learned_dynamics.config import RunSpec
from learned_dynamics.evaluation import BASELINE_IDS, EvaluationCandidate, summarize_records
from learned_dynamics.evaluation_config import (
    ARTIFACT_BUDGET_BYTES, FIELDS as CONFIG_FIELDS, EvaluationConfig, HORIZONS,
)
from learned_dynamics.training_evidence import load_run_manifest
from learned_dynamics.types import ModelSpec, validate_hash


FILES = ("evaluation-config.toml", "training-run-manifest.json", "evaluation.json", "records.jsonl")
DATA_FILES = ("manifest.json", "train.jsonl", "validation.jsonl", "test.jsonl")
RECORD_FIELDS = {
    "predictor_id", "candidate", "membership", "split", "episode_id",
    "scenario_id", "controller_id", "data_seed", "start_step_index",
    "horizon_steps", "horizon_s", "status", "unavailable_reason",
    "applied_actions_nm", "expected_endpoint_state", "predicted_endpoint_state",
    "residual", "failure_step", "failure_reason",
}
EVALUATION_FIELDS = {
    "schema_version", "config_sha256", "dataset_version", "dataset_dir",
    "dataset_files", "training_roster_dir", "training_roster_sha256", "case_ids",
    "record_count", "summaries",
}


def _case_id(case: PredictionCase) -> dict:
    return {"split": case.split, "episode_id": case.episode_id, "step_index": case.step_index}


def write_evidence(output_dir: Path, config: EvaluationConfig, roster: dict,
                   records: tuple[dict, ...], cases: tuple[PredictionCase, ...]) -> Path:
    """Publish precomputed records once; semantic validation belongs to the loader."""
    if not isinstance(config, EvaluationConfig) or not isinstance(roster, dict):
        raise ValueError("evidence requires checked config and roster")
    output = Path(output_dir)
    if output.exists() or output.is_symlink() or output.resolve() != config.output_dir:
        raise ValueError("evidence output must be the new configured directory")
    if not isinstance(records, tuple) or not records or not isinstance(cases, tuple) or not cases:
        raise ValueError("evidence requires immutable nonempty records and cases")
    config_bytes = config.config_path.read_bytes()
    require_equal(hashlib.sha256(config_bytes).hexdigest(), config.config_sha256,
                  "evaluation config hash")
    require_fields(roster, {"schema_version", "training_config_sha256", "planned_runs", "results"},
                   "evaluation roster snapshot")
    roster_bytes = json_bytes(roster, indent=2)
    dataset_files = {name: sha256_file(config.dataset_dir / name) for name in DATA_FILES}
    payload = {
        "schema_version": 1, "config_sha256": config.config_sha256,
        "dataset_version": config.dataset_version,
        "dataset_dir": os.path.relpath(config.dataset_dir, output),
        "dataset_files": dataset_files,
        "training_roster_dir": os.path.relpath(config.training_roster_dir, output),
        "training_roster_sha256": hashlib.sha256(roster_bytes).hexdigest(),
        "case_ids": [_case_id(case) for case in cases],
        "record_count": len(records), "summaries": list(summarize_records(records)),
    }
    encoded_records = b"".join(json_bytes(row) for row in records)
    total = len(config_bytes) + len(roster_bytes) + len(encoded_records) + len(json_bytes(payload, indent=2))
    if total > config.artifact_budget_bytes:
        raise ValueError("evaluation artifacts exceed configured budget")
    output.mkdir(parents=True)
    (output / "evaluation-config.toml").write_bytes(config_bytes)
    (output / "training-run-manifest.json").write_bytes(roster_bytes)
    (output / "evaluation.json").write_bytes(json_bytes(payload, indent=2))
    (output / "records.jsonl").write_bytes(encoded_records)
    files = {name: sha256_file(output / name) for name in FILES}
    (output / "manifest.json").write_bytes(json_bytes({"schema_version": 1, "files": files}, indent=2))
    return output


def _load_records(path: Path) -> tuple[dict, ...]:
    try:
        rows = tuple(decode_object(line) for line in path.read_text().splitlines())
    except (OSError, UnicodeError) as error:
        raise ValueError("cannot read evaluation records") from error
    if not rows:
        raise ValueError("evaluation records are empty")
    for row in rows:
        require_fields(row, RECORD_FIELDS, "evaluation record")
    return rows


def _audit_config_snapshot(path: Path, payload: dict) -> None:
    try:
        raw = tomllib.loads(path.read_text())
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as error:
        raise ValueError("cannot read evaluation config snapshot") from error
    require_fields(raw, CONFIG_FIELDS, "evaluation config snapshot")
    require_equal({
        "schema_version": raw["schema_version"],
        "dataset_version": raw["dataset_version"],
        "artifact_budget_bytes": raw["artifact_budget_bytes"],
        "horizons": raw["horizons"], "memberships": raw["memberships"],
        "baseline_ids": raw["baseline_ids"],
        "prediction_tolerance": raw["prediction_tolerance"],
    }, {
        "schema_version": 1, "dataset_version": payload["dataset_version"],
        "artifact_budget_bytes": ARTIFACT_BUDGET_BYTES,
        "horizons": list(HORIZONS),
        "memberships": ["all", "history_common", "loss_common"],
        "baseline_ids": list(BASELINE_IDS), "prediction_tolerance": 1e-12,
    }, "evaluation config protocol")
    for name in ("dataset_dir", "training_roster_dir", "output_dir"):
        value = raw[name]
        if (not isinstance(value, str) or not value or value.startswith("~") or
                "$" in value or Path(value).is_absolute()):
            raise ValueError("evaluation config paths must remain literal and relative")


def _roster_identities(roster: dict) -> tuple[dict, dict]:
    require_fields(roster, {"schema_version", "training_config_sha256", "planned_runs", "results"},
                   "evaluation roster snapshot")
    if type(roster["schema_version"]) is not int or roster["schema_version"] != 1:
        raise ValueError("unsupported roster snapshot")
    validate_hash(roster["training_config_sha256"])
    planned, results = {}, {}
    if not isinstance(roster["planned_runs"], list) or not isinstance(roster["results"], list):
        raise ValueError("roster entries must be lists")
    for item in roster["planned_runs"]:
        require_fields(item, {"run_id", "run_spec"}, "planned evaluation run")
        spec = item["run_spec"]
        model = spec.get("model") if isinstance(spec, dict) else None
        run = RunSpec(spec.get("variant_id"), ModelSpec(**model), spec.get("loss"),
                      spec.get("membership"), spec.get("seed"), spec.get("device"),
                      spec.get("repeat"))
        require_equal(item, {"run_id": run.run_id, "run_spec": run.mapping()}, "planned run")
        if run.run_id in planned:
            raise ValueError("duplicate planned run")
        planned[run.run_id] = run
    for item in roster["results"]:
        require_fields(item, {"run_id", "status", "checkpoint_dir", "checkpoint_sha256",
                              "selected_epoch", "selected_validation_mse", "failure_reason"},
                       "evaluation run result")
        if item["run_id"] not in planned or item["run_id"] in results:
            raise ValueError("result roster does not match planned runs")
        results[item["run_id"]] = item
    if set(results) != set(planned):
        raise ValueError("result roster is incomplete")
    return planned, results


def _candidate_identity(predictor_id: str, membership: str, planned: dict,
                        results: dict, training_hash: str) -> dict:
    if predictor_id in BASELINE_IDS:
        return EvaluationCandidate.baseline(predictor_id, object()).identity()
    if predictor_id not in planned or planned[predictor_id].membership != membership:
        raise ValueError("learned candidate is absent from declared membership roster")
    run, result = planned[predictor_id], results[predictor_id]
    if result["status"] == "failed":
        return EvaluationCandidate.failed(
            predictor_id=predictor_id, predictor=object(), run_spec=run.mapping(),
            training_config_sha256=training_hash,
            failure_reason=result["failure_reason"]).identity()
    if result["status"] != "success":
        raise ValueError("training result status is unsupported")
    return EvaluationCandidate.learned(
        predictor_id=predictor_id, predictor=object(), run_spec=run.mapping(),
        checkpoint_dir=result["checkpoint_dir"], checkpoint_sha256=result["checkpoint_sha256"],
        selected_epoch=result["selected_epoch"],
        selected_validation_mse=result["selected_validation_mse"],
        training_config_sha256=training_hash).identity()


def _episodes(cases: tuple[PredictionCase, ...]) -> dict[tuple[str, str], tuple[PredictionCase, ...]]:
    grouped = defaultdict(list)
    for case in cases:
        grouped[(case.split, case.episode_id)].append(case)
    return {key: tuple(sorted(value, key=lambda case: case.step_index)) for key, value in grouped.items()}


def _audit_record(row: dict, case: PredictionCase, episode: tuple[PredictionCase, ...],
                  start: int, identity: dict) -> None:
    horizon, membership = row["horizon_steps"], row["membership"]
    if type(horizon) is not int or horizon not in HORIZONS:
        raise ValueError("record horizon is outside protocol")
    expected_base = {
        "predictor_id": row["predictor_id"], "candidate": identity,
        "membership": membership, "split": case.split, "episode_id": case.episode_id,
        "scenario_id": case.scenario_id, "controller_id": case.controller_id,
        "data_seed": case.seed, "start_step_index": case.step_index,
        "horizon_steps": horizon, "horizon_s": horizon * float(case.context.control_interval_s),
    }
    require_equal({key: row[key] for key in expected_base}, expected_base, "record identity")
    unavailable = membership == "history_common" and start == 0
    required = max(horizon, 5 if membership == "loss_common" else horizon)
    unavailable = unavailable or len(episode) - start < required
    if unavailable:
        expected = dict(expected_base, status="unavailable",
            unavailable_reason=("history_common requires a previous row" if membership == "history_common" and start == 0 else "insufficient episode-local future rows"),
            applied_actions_nm=[], expected_endpoint_state=None, predicted_endpoint_state=None,
            residual=None, failure_step=None, failure_reason=None)
        require_equal(row, expected, "unavailable record")
        return
    window = episode[start:start + horizon]
    actions = [float(item.applied_torque_nm) for item in window]
    expected_state = asdict(window[-1].expected_next_state)
    require_equal(row["applied_actions_nm"], actions, "recorded actions")
    require_equal(row["expected_endpoint_state"], expected_state, "expected endpoint")
    if row["status"] == "success":
        if any(row[name] is not None for name in ("unavailable_reason", "failure_step", "failure_reason")):
            raise ValueError("successful record contains failure fields")
        predicted = row["predicted_endpoint_state"]
        if not isinstance(predicted, dict) or set(predicted) != {"position_rad", "velocity_rad_s"}:
            raise ValueError("predicted endpoint fields do not match schema")
        state = DynamicsState(predicted["position_rad"], predicted["velocity_rad_s"])
        require_equal(row["residual"], asdict(residual(state, window[-1].expected_next_state)),
                      "saved residual")
    elif row["status"] == "failed":
        if (row["predicted_endpoint_state"] is not None or row["residual"] is not None or
                row["unavailable_reason"] is not None or type(row["failure_step"]) is not int or
                not 1 <= row["failure_step"] <= horizon or
                not isinstance(row["failure_reason"], str) or not row["failure_reason"]):
            raise ValueError("failed record fields do not match schema")
    else:
        raise ValueError("eligible record status must be success or failed")


def load_evidence(input_dir: Path) -> dict:
    """Audit saved bytes and semantic derivations without fitting or prediction."""
    directory = Path(input_dir)
    try:
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError("evaluation evidence must be a real directory")
        if any(path.is_symlink() for path in directory.iterdir()):
            raise ValueError("evaluation evidence cannot contain symlinks")
        if {path.name for path in directory.iterdir()} != set(FILES) | {"manifest.json"}:
            raise ValueError("evaluation evidence files do not match schema")
        manifest = decode_object((directory / "manifest.json").read_text())
        require_equal(manifest, {"schema_version": 1,
            "files": {name: sha256_file(directory / name) for name in FILES}},
            "evaluation evidence checksums")
        payload = decode_object((directory / "evaluation.json").read_text())
        require_fields(payload, EVALUATION_FIELDS, "evaluation evidence")
        if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
            raise ValueError("unsupported evaluation evidence schema")
        require_equal(payload["config_sha256"], sha256_file(directory / "evaluation-config.toml"),
                      "evaluation config identity")
        _audit_config_snapshot(directory / "evaluation-config.toml", payload)
        roster_bytes = (directory / "training-run-manifest.json").read_bytes()
        require_equal(payload["training_roster_sha256"], hashlib.sha256(roster_bytes).hexdigest(),
                      "training roster identity")
        roster = decode_object(roster_bytes.decode())
        planned, results = _roster_identities(roster)
        roster_dir = (directory / payload["training_roster_dir"]).resolve()
        live_roster = load_run_manifest(roster_dir, validate_checkpoints=False)
        require_equal({key: live_roster[key] for key in roster}, roster,
                      "saved training roster snapshot")
        dataset = (directory / payload["dataset_dir"]).resolve()
        require_equal(payload["dataset_files"], {name: sha256_file(dataset / name) for name in DATA_FILES},
                      "immutable dataset files")
        all_cases = load_prediction_cases(dataset, payload["dataset_version"])
        lookup = {(case.split, case.episode_id, case.step_index): case for case in all_cases}
        selected = []
        for value in payload["case_ids"]:
            require_fields(value, {"split", "episode_id", "step_index"}, "case identity")
            key = (value["split"], value["episode_id"], value["step_index"])
            if key not in lookup:
                raise ValueError("evaluation case is absent from immutable dataset")
            selected.append(lookup[key])
        if len(selected) != len(set((c.split, c.episode_id, c.step_index) for c in selected)):
            raise ValueError("duplicate evaluation case")
        require_equal(
            sorted((c.split, c.episode_id, c.step_index) for c in selected),
            sorted((c.split, c.episode_id, c.step_index) for c in all_cases if c.split == "test"),
            "complete final-test membership")
        cases, episodes = tuple(selected), _episodes(tuple(selected))
        positions = {(case.split, case.episode_id, case.step_index): index
                     for episode in episodes.values() for index, case in enumerate(episode)}
        rows = _load_records(directory / "records.jsonl")
        expected_keys, actual_keys = set(), set()
        for membership in ("all", "history_common", "loss_common"):
            ids = (*BASELINE_IDS, *(run_id for run_id, run in planned.items()
                                    if run.membership == membership))
            for predictor_id in ids:
                for case in cases:
                    for horizon in HORIZONS:
                        expected_keys.add((predictor_id, membership, case.split, case.episode_id,
                                           case.step_index, horizon))
        for row in rows:
            key = (row["predictor_id"], row["membership"], row["split"], row["episode_id"],
                   row["start_step_index"], row["horizon_steps"])
            if key in actual_keys or key not in expected_keys:
                raise ValueError("record membership is duplicate or undeclared")
            actual_keys.add(key)
            case = lookup[(row["split"], row["episode_id"], row["start_step_index"])]
            episode = episodes[(case.split, case.episode_id)]
            identity = _candidate_identity(row["predictor_id"], row["membership"], planned,
                                           results, roster["training_config_sha256"])
            _audit_record(row, case, episode, positions[(case.split, case.episode_id, case.step_index)], identity)
        require_equal(sorted(actual_keys), sorted(expected_keys), "complete evaluation membership")
        require_equal(payload["record_count"], len(rows), "evaluation record count")
        require_equal(payload["summaries"], list(summarize_records(rows)), "evaluation summaries")
        return payload | {"records": rows, "roster": roster}
    except (OSError, UnicodeError, KeyError, TypeError, ValueError) as error:
        if isinstance(error, ValueError):
            raise
        raise ValueError("cannot audit evaluation evidence") from error
