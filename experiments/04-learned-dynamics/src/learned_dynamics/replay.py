"""Explicit checkpoint replay, separate from the saved-record semantic audit."""

from pathlib import Path
import math

from forward_dynamics.dataset import load_prediction_cases
from forward_dynamics.serialization import require_equal
from learned_dynamics.checkpoint import load_checkpoint
from learned_dynamics.evaluation import EvaluationCandidate, evaluate, final_test_runs
from learned_dynamics.evidence import load_evidence
from learned_dynamics.predictor import LearnedPredictor, SUPPORTED_CONTEXT
from learned_dynamics.training_evidence import load_run_manifest


ABSOLUTE_TOLERANCES = {"position_rad": 1e-6, "velocity_rad_s": 1e-5}


def _compare_predictions(actual: list[dict], stored: tuple[dict, ...]) -> None:
    require_equal(len(actual), len(stored), "checkpoint replay count")
    for replayed, recorded in zip(actual, stored, strict=True):
        if replayed["status"] != "success" or recorded["status"] != "success":
            require_equal(replayed, recorded, "checkpoint replay predictions")
            continue
        numeric = {"predicted_endpoint_state", "residual"}
        require_equal({key: value for key, value in replayed.items() if key not in numeric},
                      {key: value for key, value in recorded.items() if key not in numeric},
                      "checkpoint replay identity")
        for field, tolerance in ABSOLUTE_TOLERANCES.items():
            first, second = replayed["predicted_endpoint_state"][field], recorded["predicted_endpoint_state"][field]
            if (type(first) not in (int, float) or type(second) not in (int, float) or
                    not math.isfinite(first) or not math.isfinite(second) or
                    abs(first - second) > tolerance):
                raise ValueError(f"checkpoint replay {field} exceeds declared tolerance")
            residual_field = "position_error_rad" if field == "position_rad" else "velocity_error_rad_s"
            a, b = replayed["residual"][residual_field], recorded["residual"][residual_field]
            if (type(a) not in (int, float) or type(b) not in (int, float) or
                    not math.isfinite(a) or not math.isfinite(b) or abs(a - b) > tolerance):
                raise ValueError("checkpoint replay residual exceeds declared tolerance")


def replay_checkpoints(input_dir: Path, device: str = "recorded") -> dict:
    if device not in ("recorded", "cpu", "mps"):
        raise ValueError("replay device must be recorded, cpu or mps")
    evidence = load_evidence(input_dir)
    directory = Path(input_dir)
    roster_dir = (directory / evidence["training_roster_dir"]).resolve()
    live = load_run_manifest(roster_dir)
    snapshot = evidence["roster"]
    require_equal({key: live[key] for key in snapshot}, snapshot, "replay training roster")
    dataset = (directory / evidence["dataset_dir"]).resolve()
    all_cases = load_prediction_cases(dataset, evidence["dataset_version"])
    wanted = {(row["split"], row["episode_id"], row["step_index"])
              for row in evidence["case_ids"]}
    cases = tuple(case for case in all_cases
                  if (case.split, case.episode_id, case.step_index) in wanted)
    planned = {row["run_id"]: row["run_spec"] for row in snapshot["planned_runs"]}
    publication = final_test_runs(snapshot)
    stored = tuple(row for row in evidence["records"]
                   if row["candidate"]["kind"] == "learned")
    replayed = []
    for result in snapshot["results"]:
        if result["run_id"] not in publication or result["status"] != "success":
            continue
        run_spec = planned[result["run_id"]]
        target = run_spec["device"] if device == "recorded" else device
        bundle = load_checkpoint(roster_dir / result["checkpoint_dir"], device=target)
        require_equal(bundle.bundle_sha256, result["checkpoint_sha256"], "checkpoint replay hash")
        predictor = LearnedPredictor(bundle.model, bundle.normalizer, SUPPORTED_CONTEXT,
                                     result["run_id"], target)
        candidate = EvaluationCandidate.learned(
            predictor_id=result["run_id"], predictor=predictor, run_spec=run_spec,
            checkpoint_dir=result["checkpoint_dir"], checkpoint_sha256=result["checkpoint_sha256"],
            selected_epoch=result["selected_epoch"],
            selected_validation_mse=result["selected_validation_mse"],
            training_config_sha256=snapshot["training_config_sha256"])
        replayed.extend(evaluate(cases, (candidate,), (1, 5, 10, 25, 50, 100),
                                 run_spec["membership"]))
    _compare_predictions(replayed, stored)
    return {"schema_version": 1, "replayed_candidates": len({r["predictor_id"] for r in replayed}),
            "replayed_records": len(replayed), "status": "verified",
            "comparison_scope": "cross_device_diagnostic" if any(
                device not in ("recorded", planned[r["predictor_id"]]["device"])
                for r in replayed) else "same_device",
            "absolute_tolerances": ABSOLUTE_TOLERANCES.copy()}
