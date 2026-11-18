"""Frozen study matrix, validation selection freeze, and phase orchestration."""

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import tomllib
from typing import Literal

from forward_dynamics.serialization import decode_object, json_bytes, require_equal, require_fields, sha256_file
from learned_dynamics.benchmarking import benchmark_inference, pin_cpu_threads
from learned_dynamics.config import PROTOCOL_VARIANTS, TrainingConfig, load_training_config
from learned_dynamics.data import load_views
from learned_dynamics.evaluation import run_evaluation
from learned_dynamics.evaluation_config import load_evaluation_config
from learned_dynamics.training import train
from learned_dynamics.training_evidence import load_run_manifest


FIELDS = {
    "schema_version", "training_config", "evaluation_config", "output_dir",
    "benchmark_output_dir", "artifact_budget_bytes",
}


def _relative(path: Path, value: object, label: str) -> Path:
    if (not isinstance(value, str) or not value or value.startswith("~") or "$" in value or
            Path(value).is_absolute()):
        raise ValueError(f"{label} must be a literal relative path")
    return (path.parent / value).resolve()


@dataclass(frozen=True)
class StudyConfig:
    schema_version: int
    config_path: Path
    config_sha256: str
    training_config: Path
    evaluation_config: Path
    output_dir: Path
    benchmark_output_dir: Path
    artifact_budget_bytes: int

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported study config schema")
        if type(self.artifact_budget_bytes) is not int or self.artifact_budget_bytes != 500 * 1024 * 1024:
            raise ValueError("study artifact budget does not match protocol")
        if len({self.training_config, self.evaluation_config,
                self.output_dir, self.benchmark_output_dir}) != 4:
            raise ValueError("study inputs and outputs must be distinct")


def load_study_config(path: Path) -> StudyConfig:
    config_path = Path(path).resolve()
    try:
        with config_path.open("rb") as stream:
            raw = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ValueError("cannot read study config") from error
    require_fields(raw, FIELDS, "study config")
    config = StudyConfig(
        raw["schema_version"], config_path, sha256_file(config_path),
        _relative(config_path, raw["training_config"], "training_config"),
        _relative(config_path, raw["evaluation_config"], "evaluation_config"),
        _relative(config_path, raw["output_dir"], "output_dir"),
        _relative(config_path, raw["benchmark_output_dir"], "benchmark_output_dir"),
        raw["artifact_budget_bytes"],
    )
    if not config.training_config.is_file() or not config.evaluation_config.is_file():
        raise ValueError("study config inputs must exist")
    study_matrix(load_training_config(config.training_config))
    return config


def study_matrix(config: TrainingConfig) -> dict:
    if not isinstance(config, TrainingConfig):
        raise ValueError("study matrix requires a checked training config")
    expected_membership = {name: values[3] for name, values in PROTOCOL_VARIANTS.items()}
    actual_membership = {}
    final, diagnostic = [], []
    for run in config.runs:
        actual_membership.setdefault(run.variant_id, run.membership)
        if actual_membership[run.variant_id] != run.membership:
            raise ValueError("variant membership changes across runs")
        (final if run.device == "mps" and run.repeat == 0 else diagnostic).append(run)
    if actual_membership != expected_membership:
        raise ValueError("study variants do not match the six frozen arms")
    expected_final = {(variant, seed) for variant in PROTOCOL_VARIANTS for seed in range(5)}
    if {(run.variant_id, run.seed) for run in final} != expected_final or len(final) != 30:
        raise ValueError("final-test matrix must contain five MPS repeat-0 seeds per variant")
    expected_diagnostic = {
        ("canonical32", device, seed, repeat)
        for device, repeat in (("mps", 1), ("cpu", 0), ("cpu", 1))
        for seed in range(5)
    }
    if ({(run.variant_id, run.device, run.seed, run.repeat) for run in diagnostic}
            != expected_diagnostic or len(config.runs) != 45):
        raise ValueError("diagnostic matrix must contain the 15 canonical device/repeat runs")
    return {
        "variant_membership": actual_membership,
        "planned_run_count": len(config.runs), "final_test_run_count": len(final),
        "diagnostic_run_count": len(diagnostic),
        "final_test_seeds": sorted({run.seed for run in final}),
        "matched_pairs": [
            {"variants": ["history0", "history1"], "membership": "history_common"},
            {"variants": ["loss1", "loss5"], "membership": "loss_common"},
        ],
    }


def _selection_snapshot(roster: dict) -> list[dict]:
    planned = {row["run_id"]: row["run_spec"] for row in roster["planned_runs"]}
    return [{
        "run_id": row["run_id"], "run_spec": planned[row["run_id"]],
        "status": row["status"], "checkpoint_dir": row["checkpoint_dir"],
        "checkpoint_sha256": row["checkpoint_sha256"],
        "selected_epoch": row["selected_epoch"],
        "selected_validation_mse": row["selected_validation_mse"],
        "failure_reason": row["failure_reason"],
    } for row in roster["results"]]


def freeze_validation_roster(study_dir: Path, roster_dir: Path, roster: dict,
                             study_config_sha256: str) -> Path:
    output = Path(study_dir)
    path = output / "validation-freeze.json"
    if path.exists() or path.is_symlink():
        raise ValueError("validation selection is already frozen")
    required = {"schema_version", "training_config_sha256", "provenance", "planned_runs", "results"}
    if not isinstance(roster, dict) or not required.issubset(roster):
        raise ValueError("validation roster fields do not match schema")
    payload = {
        "schema_version": 1, "study_config_sha256": study_config_sha256,
        "training_config_sha256": roster["training_config_sha256"],
        "training_source_sha": roster["provenance"]["source_sha"],
        "training_working_tree_dirty": roster["provenance"]["working_tree_dirty"],
        "roster_dir": os.path.relpath(Path(roster_dir).resolve(), output.resolve()),
        "run_manifest_sha256": sha256_file(Path(roster_dir) / "run-manifest.json"),
        "selections": _selection_snapshot(roster),
    }
    output.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json_bytes(payload, indent=2))
    return path


def require_frozen_validation(study_dir: Path, *, roster: dict | None = None) -> dict:
    path = Path(study_dir) / "validation-freeze.json"
    try:
        if path.is_symlink() or not path.is_file():
            raise ValueError("frozen validation selection is missing")
        payload = decode_object(path.read_text())
        require_fields(payload, {
            "schema_version", "study_config_sha256", "training_config_sha256",
            "training_source_sha", "training_working_tree_dirty", "roster_dir",
            "run_manifest_sha256", "selections",
        }, "frozen validation selection")
        if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
            raise ValueError("unsupported frozen validation schema")
        roster_dir = (Path(study_dir) / payload["roster_dir"]).resolve()
        require_equal(payload["run_manifest_sha256"],
                      sha256_file(roster_dir / "run-manifest.json"),
                      "frozen validation run manifest")
        if roster is not None:
            require_equal(payload["training_config_sha256"], roster["training_config_sha256"],
                          "frozen validation training config")
            require_equal(payload["selections"], _selection_snapshot(roster),
                          "frozen validation selections")
        return payload
    except (OSError, UnicodeError, KeyError, TypeError) as error:
        raise ValueError("cannot validate frozen validation selection") from error


def run_studies(config_path: Path, output_dir: Path,
                phase: Literal["validation", "final_test"]) -> Path:
    """Run validation/freeze first; permit final test only against that exact roster."""
    config = load_study_config(config_path)
    study_output = Path(output_dir).resolve()
    if study_output != config.output_dir:
        raise ValueError("study output must equal the configured output")
    training = load_training_config(config.training_config)
    if phase == "validation":
        if study_output.exists() or config.benchmark_output_dir.exists():
            raise ValueError("validation study outputs must be fresh")
        pin_cpu_threads()
        roster_dir = train(config.training_config, training.output_dir, allow_full_study=True)
        roster = load_run_manifest(roster_dir)
        frozen = freeze_validation_roster(study_output, roster_dir, roster, config.config_sha256)
        canonical = next((row for row in roster["results"]
                          if row["run_id"] == "canonical32-mps-seed0-repeat0"), None)
        if canonical is None:
            raise ValueError("canonical benchmark run is absent from the frozen roster")
        if canonical["status"] == "success":
            validation = load_views(training.dataset_dir, training.dataset_version).validation
            benchmark_inference(roster_dir / canonical["checkpoint_dir"], validation,
                                config.benchmark_output_dir)
        else:
            (study_output / "benchmark-unavailable.json").write_bytes(json_bytes({
                "schema_version": 1, "run_id": canonical["run_id"],
                "status": "unavailable", "reason": canonical["failure_reason"],
            }, indent=2))
        return frozen
    if phase == "final_test":
        roster = load_run_manifest(training.output_dir)
        frozen = require_frozen_validation(study_output, roster=roster)
        require_equal(frozen["study_config_sha256"], config.config_sha256,
                      "frozen validation study config")
        evaluation = load_evaluation_config(config.evaluation_config)
        evidence_dir = run_evaluation(config.evaluation_config, training.output_dir,
                                      evaluation.output_dir)
        path = study_output / "final-test.json"
        if path.exists() or path.is_symlink():
            raise ValueError("final test has already been executed")
        manifest = Path(evidence_dir) / "manifest.json"
        if not manifest.is_file() or manifest.is_symlink():
            raise ValueError("final-test evidence manifest is missing or unsafe")
        benchmark_manifest = config.benchmark_output_dir / "manifest.json"
        benchmark_unavailable = study_output / "benchmark-unavailable.json"
        if benchmark_manifest.is_file() and not benchmark_manifest.is_symlink():
            benchmark_status = "available"
            benchmark_dir = os.path.relpath(config.benchmark_output_dir, study_output)
            benchmark_manifest_sha256 = sha256_file(benchmark_manifest)
            benchmark_unavailable_sha256 = None
        elif benchmark_unavailable.is_file() and not benchmark_unavailable.is_symlink():
            unavailable = decode_object(benchmark_unavailable.read_text())
            require_fields(unavailable, {
                "schema_version", "run_id", "status", "reason",
            }, "benchmark-unavailable record")
            if (unavailable["schema_version"] != 1 or
                    unavailable["status"] != "unavailable" or
                    not isinstance(unavailable["reason"], str) or
                    not unavailable["reason"]):
                raise ValueError("benchmark-unavailable record is invalid")
            benchmark_status = "unavailable"
            benchmark_dir = None
            benchmark_manifest_sha256 = None
            benchmark_unavailable_sha256 = sha256_file(benchmark_unavailable)
        else:
            raise ValueError("benchmark outcome evidence is missing or unsafe")
        path.write_bytes(json_bytes({
            "schema_version": 1, "study_config_sha256": config.config_sha256,
            "validation_freeze_sha256": sha256_file(study_output / "validation-freeze.json"),
            "evaluation_dir": os.path.relpath(Path(evidence_dir).resolve(), study_output),
            "evaluation_manifest_sha256": sha256_file(manifest),
            "benchmark_status": benchmark_status, "benchmark_dir": benchmark_dir,
            "benchmark_manifest_sha256": benchmark_manifest_sha256,
            "benchmark_unavailable_sha256": benchmark_unavailable_sha256,
        }, indent=2))
        return path
    raise ValueError("study phase must be validation or final_test")
