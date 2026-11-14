"""Complete, non-overwriting run rosters for training and selection."""

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
import math
import re
import subprocess
import tempfile
import tomllib

from forward_dynamics.serialization import (
    decode_object, json_bytes, require_equal, require_fields, sha256_file,
)
from learned_dynamics.checkpoint import load_checkpoint
from learned_dynamics.checkpoint_metadata import REPOSITORY_ROOT
from learned_dynamics.config import (
    PROTOCOL_ARTIFACT_BUDGET_BYTES, PROTOCOL_BATCH_SIZE,
    PROTOCOL_SAMPLER_SEED_OFFSET, PROTOCOL_VALIDATION_INTERVAL_EPOCHS,
    RunSpec, TrainingConfig,
)
from learned_dynamics.types import DATASET_VERSION, ModelSpec, validate_hash


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
CURVE_FIELDS = {
    "schema_version", "run_id", "run_spec", "training_config_sha256",
    "selected_epoch", "selected_validation_mse",
    "deterministic_algorithms_requested", "epochs",
}
EPOCH_FIELDS = {
    "epoch", "training_loss", "validation_mse", "batch_sizes",
    "updates_this_epoch", "optimizer_updates", "training_seconds",
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
        "schema_version": config.schema_version, "dataset_version": config.dataset_version,
        "epochs": config.epochs,
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
        if root is not None:
            curve_path = root / value["run_id"] / "learning-curve.json"
            if curve_path.is_file():
                _validate_curve(curve_path, value, run, epochs, config_sha256)
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
        _validate_curve(root / value["run_id"] / "learning-curve.json", value, run,
                        epochs, config_sha256)


def _finite_nonnegative(value: object, description: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{description} must be finite and nonnegative")
    return float(value)


def _validate_curve(path: Path, result: dict, run: dict, epochs: int,
                    config_sha256: str) -> None:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_MANIFEST_BYTES:
        raise ValueError("learning curve is missing, unsafe or oversized")
    curve = decode_object(path.read_text())
    require_fields(curve, CURVE_FIELDS, "learning curve")
    if type(curve["schema_version"]) is not int or curve["schema_version"] != 1:
        raise ValueError("unsupported learning curve schema")
    require_equal(curve["run_id"], result["run_id"], "learning curve run id")
    require_equal(curve["run_spec"], run["run_spec"], "learning curve run specification")
    require_equal(curve["training_config_sha256"], config_sha256,
                  "learning curve config hash")
    if curve["deterministic_algorithms_requested"] is not True:
        raise ValueError("learning curve must record the deterministic algorithm request")
    rows = curve["epochs"]
    if not isinstance(rows, list) or len(rows) > epochs:
        raise ValueError("learning curve epoch count exceeds configured budget")
    expected_updates = 0
    validation_values = []
    for expected_epoch, row in enumerate(rows, start=1):
        require_fields(row, EPOCH_FIELDS, "learning curve epoch")
        if type(row["epoch"]) is not int or row["epoch"] != expected_epoch:
            raise ValueError("learning curve epochs must be consecutive from one")
        _finite_nonnegative(row["training_loss"], "learning curve training loss")
        validation_values.append(_finite_nonnegative(
            row["validation_mse"], "learning curve validation MSE"))
        sizes = row["batch_sizes"]
        if (not isinstance(sizes, list) or not sizes or
                any(type(size) is not int or not 1 <= size <= PROTOCOL_BATCH_SIZE
                    for size in sizes)):
            raise ValueError("learning curve batch sizes are invalid")
        if type(row["updates_this_epoch"]) is not int or row["updates_this_epoch"] != len(sizes):
            raise ValueError("learning curve update count disagrees with batch sizes")
        expected_updates += len(sizes)
        if type(row["optimizer_updates"]) is not int or row["optimizer_updates"] != expected_updates:
            raise ValueError("learning curve optimizer updates are not cumulative")
        _finite_nonnegative(row["training_seconds"], "learning curve timing")
    if result["status"] == "success":
        if len(rows) != epochs:
            raise ValueError("successful learning curve is incomplete")
        if (type(curve["selected_epoch"]) is not int or
                type(curve["selected_validation_mse"]) not in (int, float)):
            raise ValueError("successful learning curve selection has invalid types")
        selected_epoch = validation_values.index(min(validation_values)) + 1
        if curve["selected_epoch"] != selected_epoch:
            raise ValueError("learning curve selected epoch is not the earliest minimum")
        if curve["selected_validation_mse"] != validation_values[selected_epoch - 1]:
            raise ValueError("learning curve selected metric disagrees with its epochs")
        require_equal(curve["selected_epoch"], result["selected_epoch"],
                      "learning curve selected epoch")
        require_equal(curve["selected_validation_mse"], result["selected_validation_mse"],
                      "learning curve selected metric")
    elif curve["selected_epoch"] is not None or curve["selected_validation_mse"] is not None:
        raise ValueError("failed learning curve cannot declare a selection")


def _validate_config_snapshot(root: Path, payload: dict) -> None:
    path = root / "training-config.toml"
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_MANIFEST_BYTES:
        raise ValueError("original training config snapshot is missing, unsafe or oversized")
    require_equal(sha256_file(path), payload["training_config_sha256"],
                  "original training config hash")
    try:
        raw = tomllib.loads(path.read_text())
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as error:
        raise ValueError("cannot read original training config snapshot") from error
    try:
        training = raw["training"]
        optimizer = raw["optimizer"]
        normalization = raw["normalization"]
        snapshot_config = {
            "schema_version": raw["schema_version"],
            "dataset_version": raw["dataset_version"],
            "epochs": training["epochs"], "batch_size": training["batch_size"],
            "sampler_seed_offset": training["sampler_seed_offset"],
            "validation_interval_epochs": training["validation_interval_epochs"],
            "selection_metric": training["selection_metric"],
            "tie_break": training["tie_break"],
            "artifact_budget_bytes": raw["artifact_budget_bytes"],
            "resume": raw["resume"],
            "optimizer": {
                "name": optimizer["name"], "learning_rate": optimizer["learning_rate"],
                "betas": [optimizer["beta1"], optimizer["beta2"]],
                "epsilon": optimizer["epsilon"], "weight_decay": optimizer["weight_decay"],
            },
            "normalization": {
                "statistics": normalization["statistics"],
                "constant_threshold": normalization["constant_threshold"],
                "target_transform": normalization["target_transform"],
            },
        }
    except (KeyError, TypeError) as error:
        raise ValueError("original training config settings are incomplete") from error
    require_equal(payload["config"], snapshot_config, "original training config settings")
    if not isinstance(raw.get("runs"), list):
        raise ValueError("original training config roster is missing")
    planned = []
    for value in raw["runs"]:
        require_fields(value, {
            "variant_id", "hidden_width", "history_steps", "loss", "membership",
            "seed", "device", "repeat",
        }, "original training config run")
        spec = RunSpec(value["variant_id"], ModelSpec(value["hidden_width"], value["history_steps"]),
                       value["loss"], value["membership"], value["seed"], value["device"],
                       value["repeat"])
        planned.append({"run_id": spec.run_id, "run_spec": spec.mapping()})
    require_equal(payload["planned_runs"], planned, "original training config roster")


def _validate_payload(payload: dict, root: Path | None, *, validate_snapshot: bool = True) -> None:
    require_fields(payload, PAYLOAD_FIELDS, "training run manifest")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise ValueError("unsupported run manifest schema")
    validate_hash(payload["training_config_sha256"])
    config = payload["config"]
    require_fields(config, {
        "schema_version", "dataset_version", "epochs", "batch_size", "sampler_seed_offset",
        "validation_interval_epochs", "selection_metric", "tie_break",
        "artifact_budget_bytes", "resume", "optimizer", "normalization",
    }, "embedded training settings")
    for field in ("epochs", "batch_size", "sampler_seed_offset", "validation_interval_epochs", "artifact_budget_bytes"):
        if type(config[field]) is not int or config[field] <= 0:
            raise ValueError(f"embedded {field} must be a positive integer")
    if (not 1 <= config["epochs"] <= 300 or config["batch_size"] != PROTOCOL_BATCH_SIZE or
            config["sampler_seed_offset"] != PROTOCOL_SAMPLER_SEED_OFFSET or
            config["validation_interval_epochs"] != PROTOCOL_VALIDATION_INTERVAL_EPOCHS or
            config["artifact_budget_bytes"] != PROTOCOL_ARTIFACT_BUDGET_BYTES or
            config["selection_metric"] != "validation_normalized_one_step_mse" or
            config["tie_break"] != "earliest_epoch" or config["resume"] != "reject"):
        raise ValueError("embedded training settings do not match protocol v1")
    if config["dataset_version"] != DATASET_VERSION:
        raise ValueError("embedded dataset version does not match protocol v1")
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
    if root is not None and validate_snapshot:
        _validate_config_snapshot(root, payload)


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
    if output.exists() and any((output / name).exists() for name in (
            "run-manifest.json", "manifest.json", "training-config.toml")):
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
    _validate_payload(payload, output, validate_snapshot=False)
    content = json_bytes(payload, indent=2)
    try:
        config_content = config.config_path.read_bytes()
    except OSError as error:
        raise ValueError("cannot snapshot original training config") from error
    require_equal(hashlib.sha256(config_content).hexdigest(), config.config_sha256,
                  "original training config hash")
    existing_paths = tuple(output.rglob("*"))
    if any(path.is_symlink() for path in existing_paths):
        raise ValueError("training output cannot contain symlinked artifacts")
    existing = sum(path.stat().st_size for path in existing_paths if path.is_file())
    if existing >= config.artifact_budget_bytes:
        raise ValueError("training artifacts exceed configured budget")
    files = {
        str(path.relative_to(output)): sha256_file(path)
        for path in sorted(existing_paths)
        if path.is_file()
    }
    files["run-manifest.json"] = hashlib.sha256(content).hexdigest()
    files["training-config.toml"] = hashlib.sha256(config_content).hexdigest()
    outer = json_bytes({"schema_version": 1, "files": files}, indent=2)
    if existing + len(config_content) + len(content) + len(outer) > config.artifact_budget_bytes:
        raise ValueError("training artifacts exceed configured budget")
    with tempfile.TemporaryDirectory(prefix=".run-manifest-stage-", dir=output.parent) as temp:
        stage = Path(temp)
        (stage / "run-manifest.json").write_bytes(content)
        (stage / "manifest.json").write_bytes(outer)
        (stage / "training-config.toml").write_bytes(config_content)
        for name in ("training-config.toml", "run-manifest.json", "manifest.json"):
            try:
                with (output / name).open("xb") as stream:
                    stream.write((stage / name).read_bytes())
            except FileExistsError as error:
                raise ValueError("run manifest output already exists") from error
    load_run_manifest(output)
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
        paths = tuple(directory.rglob("*"))
        if any(path.is_symlink() for path in paths):
            raise ValueError("run manifest cannot contain symlinked artifacts")
        actual_files = {
            str(path.relative_to(directory)): sha256_file(path)
            for path in sorted(paths)
            if path.is_file() and path.relative_to(directory) != Path("manifest.json")
        }
        outer = decode_object((directory / "manifest.json").read_text())
        require_equal(outer, {"schema_version": 1, "files": actual_files},
                      "run manifest checksums")
        payload = decode_object((directory / "run-manifest.json").read_text())
        _validate_payload(payload, directory if validate_checkpoints else None)
        return payload
    except (OSError, UnicodeError) as error:
        raise ValueError("cannot load run manifest") from error
