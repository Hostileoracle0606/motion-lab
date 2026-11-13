"""Complete, non-overwriting run rosters for training and selection."""

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
import math
import re
import subprocess
import tempfile

from forward_dynamics.serialization import (
    decode_object, json_bytes, require_equal, require_fields, sha256_file,
)
from learned_dynamics.checkpoint import load_checkpoint
from learned_dynamics.checkpoint_metadata import REPOSITORY_ROOT
from learned_dynamics.config import RunSpec, TrainingConfig
from learned_dynamics.types import ModelSpec, validate_hash


MAX_MANIFEST_BYTES = 2 * 1024 * 1024
PROVENANCE_FIELDS = {
    "source_sha", "working_tree_dirty", "dataset_manifest_sha256", "train_sha256",
    "normalizer_sha256", "environment_sha256", "actual_command",
}
PAYLOAD_FIELDS = {
    "schema_version", "publication_status", "training_config_sha256", "config",
    "planned_runs", "provenance", "results",
}
RESULT_FIELDS = {
    "run_id", "status", "checkpoint_dir", "checkpoint_sha256", "selected_epoch",
    "selected_validation_mse", "failure_reason",
}


@dataclass(frozen=True)
class RunResult:
    run_id: str
    status: str
    checkpoint_dir: Path | None
    checkpoint_sha256: str | None
    selected_epoch: int | None
    selected_validation_mse: float | None
    failure_reason: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id:
            raise ValueError("run_id must be nonempty")
        if self.status == "failed":
            if any(value is not None for value in (
                self.checkpoint_dir, self.checkpoint_sha256, self.selected_epoch,
                self.selected_validation_mse,
            )) or not isinstance(self.failure_reason, str) or not self.failure_reason.strip():
                raise ValueError("failed run must retain only a structured failure reason")
        elif self.status == "success":
            if (not isinstance(self.checkpoint_dir, Path) or
                    type(self.selected_epoch) is not int or self.selected_epoch <= 0 or
                    type(self.selected_validation_mse) not in (int, float) or
                    not math.isfinite(self.selected_validation_mse) or
                    self.selected_validation_mse < 0 or self.failure_reason is not None):
                raise ValueError("successful run requires finite selection and checkpoint")
            validate_hash(self.checkpoint_sha256)
        else:
            raise ValueError("run status must be success or failed")


def _config_mapping(config: TrainingConfig) -> dict:
    return {
        "schema_version": config.schema_version, "epochs": config.epochs,
        "batch_size": config.batch_size, "sampler_seed_offset": config.sampler_seed_offset,
        "validation_interval_epochs": config.validation_interval_epochs,
        "selection_metric": config.selection_metric, "tie_break": config.tie_break,
        "artifact_budget_bytes": config.artifact_budget_bytes, "resume": config.resume,
        "optimizer": asdict(config.optimizer),
        "normalization": {
            "statistics": config.normalization_statistics,
            "constant_threshold": config.constant_threshold,
            "target_transform": config.target_transform,
        },
    }


def _validate_provenance(value: object) -> None:
    require_fields(value, PROVENANCE_FIELDS, "training provenance")
    if not isinstance(value["source_sha"], str) or re.fullmatch(r"[0-9a-f]{40}", value["source_sha"]) is None:
        raise ValueError("source_sha must be a full lowercase commit SHA")
    if type(value["working_tree_dirty"]) is not bool:
        raise ValueError("working_tree_dirty must be boolean")
    if (not value["working_tree_dirty"] and
            subprocess.run(["git", "cat-file", "-e", value["source_sha"] + "^{commit}"],
                           cwd=REPOSITORY_ROOT, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL).returncode):
        raise ValueError("clean source SHA must resolve in this repository")
    for field in ("dataset_manifest_sha256", "train_sha256", "normalizer_sha256", "environment_sha256"):
        validate_hash(value[field])
    command = value["actual_command"]
    if (not isinstance(command, list) or not command or
            any(not isinstance(part, str) or not part for part in command)):
        raise ValueError("actual_command must be a nonempty string array")


def _validate_result(value: object, run: dict, epochs: int, root: Path | None,
                     config_sha256: str, provenance: dict) -> None:
    require_fields(value, RESULT_FIELDS, "run result")
    if value["run_id"] != run["run_id"]:
        raise ValueError("results do not match the complete planned roster")
    if value["status"] == "failed":
        if (any(value[field] is not None for field in (
                "checkpoint_dir", "checkpoint_sha256", "selected_epoch", "selected_validation_mse")) or
                not isinstance(value["failure_reason"], str) or not value["failure_reason"].strip()):
            raise ValueError("failed run evidence is inconsistent")
        return
    if value["status"] != "success":
        raise ValueError("unsupported run status")
    if type(value["selected_epoch"]) is not int or not 1 <= value["selected_epoch"] <= epochs:
        raise ValueError("selected epoch is outside the configured budget")
    metric = value["selected_validation_mse"]
    if type(metric) not in (int, float) or not math.isfinite(metric) or metric < 0:
        raise ValueError("selected validation MSE must be finite and nonnegative")
    if value["failure_reason"] is not None:
        raise ValueError("successful run cannot contain a failure reason")
    validate_hash(value["checkpoint_sha256"])
    if (not isinstance(value["checkpoint_dir"], str) or not value["checkpoint_dir"] or
            Path(value["checkpoint_dir"]).is_absolute() or ".." in Path(value["checkpoint_dir"]).parts):
        raise ValueError("checkpoint_dir must be a local relative path")
    if root is not None:
        checkpoint = root / value["checkpoint_dir"]
        bundle = load_checkpoint(checkpoint, device="cpu")
        require_equal(bundle.bundle_sha256, value["checkpoint_sha256"], "checkpoint bundle hash")
        require_equal(bundle.metadata["run_spec"], run["run_spec"], "checkpoint run specification")
        require_equal(bundle.metadata["selected_epoch"], value["selected_epoch"], "checkpoint selected epoch")
        require_equal(bundle.metadata["selected_validation_mse"], metric, "checkpoint selected metric")
        require_equal(bundle.metadata["training_config_sha256"], config_sha256, "checkpoint config hash")
        for field in ("source_sha", "working_tree_dirty", "dataset_manifest_sha256", "train_sha256",
                      "normalizer_sha256"):
            require_equal(bundle.metadata[field], provenance[field], f"checkpoint {field}")
        require_equal(bundle.metadata["context"]["environment_version"],
                      provenance["environment_sha256"], "checkpoint environment hash")


def _validate_payload(payload: dict, root: Path | None) -> None:
    require_fields(payload, PAYLOAD_FIELDS, "training run manifest")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise ValueError("unsupported run manifest schema")
    validate_hash(payload["training_config_sha256"])
    config = payload["config"]
    require_fields(config, {
        "schema_version", "epochs", "batch_size", "sampler_seed_offset",
        "validation_interval_epochs", "selection_metric", "tie_break",
        "artifact_budget_bytes", "resume", "optimizer", "normalization",
    }, "embedded training settings")
    for field in ("epochs", "batch_size", "sampler_seed_offset", "validation_interval_epochs", "artifact_budget_bytes"):
        if type(config[field]) is not int or config[field] <= 0:
            raise ValueError(f"embedded {field} must be a positive integer")
    if (not 1 <= config["epochs"] <= 300 or config["batch_size"] != 64 or
            config["sampler_seed_offset"] != 100000 or
            config["validation_interval_epochs"] != 1 or
            config["selection_metric"] != "validation_normalized_one_step_mse" or
            config["tie_break"] != "earliest_epoch" or config["resume"] != "reject"):
        raise ValueError("embedded training settings do not match protocol v1")
    optimizer = config["optimizer"]
    require_fields(optimizer, {"name", "learning_rate", "betas", "epsilon", "weight_decay"},
                   "embedded optimizer")
    if (optimizer["name"] != "adam" or optimizer["learning_rate"] != 0.001 or
            optimizer["betas"] not in ((0.9, 0.999), [0.9, 0.999]) or
            optimizer["epsilon"] != 1e-8 or optimizer["weight_decay"] != 0.0):
        raise ValueError("embedded optimizer does not match protocol v1")
    normalization = config["normalization"]
    require_fields(normalization, {"statistics", "constant_threshold", "target_transform"},
                   "embedded normalization")
    if normalization != {"statistics": "float64_population_std",
                          "constant_threshold": 1e-12,
                          "target_transform": "next_minus_current"}:
        raise ValueError("embedded normalization does not match protocol v1")
    _validate_provenance(payload["provenance"])
    planned = payload["planned_runs"]
    results = payload["results"]
    if not isinstance(planned, list) or not planned or not isinstance(results, list):
        raise ValueError("planned runs/results must be arrays")
    ids = []
    for row in planned:
        require_fields(row, {"run_id", "run_spec"}, "planned run")
        require_fields(row["run_spec"], {"variant_id", "model", "loss", "membership", "seed", "device", "repeat"}, "planned run spec")
        spec = row["run_spec"]
        require_fields(spec["model"], {"hidden_width", "history_steps"}, "planned model")
        reconstructed = RunSpec(spec["variant_id"], ModelSpec(**spec["model"]),
                                spec["loss"], spec["membership"], spec["seed"], spec["device"], spec["repeat"])
        if row["run_id"] != reconstructed.run_id:
            raise ValueError("planned run id does not match run specification")
        ids.append(row["run_id"])
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate planned run id")
    if len(results) != len(planned):
        raise ValueError("results do not match the complete planned roster")
    seen = [row.get("run_id") if isinstance(row, dict) else None for row in results]
    if len(set(seen)) != len(seen):
        raise ValueError("duplicate run result")
    for value, run in zip(results, planned, strict=True):
        _validate_result(value, run, config["epochs"], root,
                         payload["training_config_sha256"], payload["provenance"])
    expected_status = "exploratory_dirty_source" if payload["provenance"]["working_tree_dirty"] else "clean_source"
    if payload["publication_status"] != expected_status:
        raise ValueError("publication status disagrees with source state")


def _result_mapping(result: RunResult, output: Path) -> dict:
    checkpoint = None
    if result.checkpoint_dir is not None:
        resolved = result.checkpoint_dir.resolve()
        if not resolved.is_relative_to(output):
            raise ValueError("checkpoint must be inside the training output")
        checkpoint = str(resolved.relative_to(output))
    return {
        "run_id": result.run_id, "status": result.status, "checkpoint_dir": checkpoint,
        "checkpoint_sha256": result.checkpoint_sha256,
        "selected_epoch": result.selected_epoch,
        "selected_validation_mse": result.selected_validation_mse,
        "failure_reason": result.failure_reason,
    }


def write_run_manifest(output_dir: Path, config: TrainingConfig,
                       results: tuple[RunResult, ...], provenance: dict) -> Path:
    """Publish an exact roster after validating every checkpoint and failure row."""
    if not isinstance(config, TrainingConfig) or not isinstance(results, tuple):
        raise ValueError("manifest requires TrainingConfig and immutable results")
    output = Path(output_dir).resolve()
    if output != config.output_dir or output.is_symlink():
        raise ValueError("manifest output must equal the configured real output")
    ids = [result.run_id for result in results]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate run result")
    if ids != [run.run_id for run in config.runs]:
        raise ValueError("results do not match the complete planned roster")
    if output.exists() and ((output / "run-manifest.json").exists() or (output / "manifest.json").exists()):
        raise ValueError("run manifest output already exists")
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "publication_status": "exploratory_dirty_source" if provenance.get("working_tree_dirty") else "clean_source",
        "training_config_sha256": config.config_sha256,
        "config": _config_mapping(config),
        "planned_runs": [{"run_id": run.run_id, "run_spec": run.mapping()} for run in config.runs],
        "provenance": provenance,
        "results": [_result_mapping(result, output) for result in results],
    }
    _validate_payload(payload, output)
    content = json_bytes(payload, indent=2)
    outer = json_bytes({"schema_version": 1, "files": {
        "run-manifest.json": hashlib.sha256(content).hexdigest()}}, indent=2)
    existing_paths = tuple(output.rglob("*"))
    if any(path.is_symlink() for path in existing_paths):
        raise ValueError("training output cannot contain symlinked artifacts")
    existing = sum(path.stat().st_size for path in existing_paths if path.is_file())
    if existing + len(content) + len(outer) > config.artifact_budget_bytes:
        raise ValueError("training artifacts exceed configured budget")
    with tempfile.TemporaryDirectory(prefix=".run-manifest-stage-", dir=output.parent) as temp:
        stage = Path(temp)
        (stage / "run-manifest.json").write_bytes(content)
        (stage / "manifest.json").write_bytes(outer)
        load_run_manifest(stage, validate_checkpoints=False)
        for name in ("run-manifest.json", "manifest.json"):
            try:
                with (output / name).open("xb") as stream:
                    stream.write((stage / name).read_bytes())
            except FileExistsError as error:
                raise ValueError("run manifest output already exists") from error
    return output


def load_run_manifest(input_dir: Path, *, validate_checkpoints: bool = True) -> dict:
    directory = Path(input_dir)
    try:
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError("run manifest must be a real directory")
        for name in ("run-manifest.json", "manifest.json"):
            path = directory / name
            if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_MANIFEST_BYTES:
                raise ValueError("run manifest file is missing, unsafe or oversized")
        outer = decode_object((directory / "manifest.json").read_text())
        require_equal(outer, {"schema_version": 1, "files": {
            "run-manifest.json": sha256_file(directory / "run-manifest.json")}}, "run manifest checksums")
        payload = decode_object((directory / "run-manifest.json").read_text())
        _validate_payload(payload, directory if validate_checkpoints else None)
        return payload
    except (OSError, UnicodeError) as error:
        raise ValueError("cannot load run manifest") from error
