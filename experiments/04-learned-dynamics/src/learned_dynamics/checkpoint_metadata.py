"""Strict checkpoint provenance relationships, independent of tensor loading."""

from dataclasses import asdict
from pathlib import Path
import re
import subprocess

from forward_dynamics.serialization import require_equal, require_fields
from forward_dynamics.types import DynamicsContext, validate_finite_scalar
from learned_dynamics.predictor import validate_context
from learned_dynamics.types import ModelSpec, Normalizer, validate_hash


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
DATASET_DIR = REPOSITORY_ROOT / "experiments/02-state-transition-dataset/data/dataset-v1"
METADATA_FIELDS = {
    "schema_version", "model_spec", "run_spec", "context", "selected_epoch",
    "selection_metric", "selected_validation_mse", "dataset_version",
    "dataset_manifest_sha256", "train_sha256", "training_config_sha256",
    "source_sha", "working_tree_dirty", "runtime", "normalizer_sha256",
    "tensor_file", "tensor_sha256",
}
DERIVED_FIELDS = {"normalizer_sha256", "tensor_file", "tensor_sha256"}


def validate_metadata(metadata: dict, normalizer: Normalizer) -> ModelSpec:
    require_fields(metadata, METADATA_FIELDS, "checkpoint metadata")
    if type(metadata["schema_version"]) is not int or metadata["schema_version"] != 1:
        raise ValueError("unsupported checkpoint schema version")
    require_fields(metadata["model_spec"], {"hidden_width", "history_steps"}, "model spec")
    spec = ModelSpec(**metadata["model_spec"])
    require_fields(metadata["context"], set(DynamicsContext.__dataclass_fields__), "context")
    validate_context(DynamicsContext(**metadata["context"]))
    run = metadata["run_spec"]
    require_fields(run, {"variant_id", "model", "loss", "membership", "seed", "device", "repeat"}, "run spec")
    require_equal(run["model"], asdict(spec), "run model specification")
    variants = {
        "canonical32": (32, 0, "one_step", "all"),
        "capacity16": (16, 0, "one_step", "all"),
        "history0": (32, 0, "one_step", "history_common"),
        "history1": (32, 1, "one_step", "history_common"),
        "loss1": (32, 0, "one_step", "loss_common"),
        "loss5": (32, 0, "rollout_5", "loss_common"),
    }
    if (not isinstance(run["variant_id"], str) or run["variant_id"] not in variants or
            (spec.hidden_width, spec.history_steps, run["loss"], run["membership"]) != variants[run["variant_id"]]):
        raise ValueError("variant/model/loss/membership do not match the protocol")
    for name, allowed in (("seed", range(5)), ("repeat", (0, 1))):
        if type(run[name]) is not int or run[name] not in allowed:
            raise ValueError(f"{name} must be a declared integer")
    if run["device"] not in ("cpu", "mps"):
        raise ValueError("run device must be cpu or mps")
    if type(metadata["selected_epoch"]) is not int or not 1 <= metadata["selected_epoch"] <= 300:
        raise ValueError("selected_epoch must be an integer in 1..300")
    if metadata["selection_metric"] != "validation_normalized_one_step_mse":
        raise ValueError("unsupported selection metric")
    validate_finite_scalar("selected_validation_mse", metadata["selected_validation_mse"])
    if metadata["selected_validation_mse"] < 0:
        raise ValueError("validation MSE cannot be negative")
    for field in ("normalizer_sha256", "dataset_manifest_sha256", "train_sha256",
                  "training_config_sha256", "tensor_sha256"):
        validate_hash(metadata[field])
    if (metadata["dataset_version"] != normalizer.dataset_version or
            metadata["dataset_manifest_sha256"] != normalizer.manifest_sha256 or
            metadata["train_sha256"] != normalizer.train_sha256):
        raise ValueError("normalizer and checkpoint dataset provenance disagree")
    if metadata["tensor_file"] != "weights.npz":
        raise ValueError("tensor_file must be the fixed local weights.npz")
    source = metadata["source_sha"]
    if not isinstance(source, str) or re.fullmatch(r"[0-9a-f]{40}", source) is None:
        raise ValueError("source_sha must be a full commit SHA")
    if subprocess.run(["git", "cat-file", "-e", source + "^{commit}"],
                      cwd=REPOSITORY_ROOT, stdout=subprocess.DEVNULL,
                      stderr=subprocess.DEVNULL).returncode:
        raise ValueError("checkpoint source SHA does not resolve in this repository")
    if type(metadata["working_tree_dirty"]) is not bool:
        raise ValueError("working_tree_dirty must be boolean")
    runtime = metadata["runtime"]
    require_fields(runtime, {"python", "torch", "numpy", "os", "architecture", "device"}, "runtime")
    if any(not isinstance(value, str) or not value.strip() for value in runtime.values()):
        raise ValueError("runtime values must be nonempty strings")
    if (re.fullmatch(r"3\.12\.\d+", runtime["python"]) is None or
            runtime["torch"] != "2.15.0" or runtime["device"] != run["device"]):
        raise ValueError("runtime Python/PyTorch/device is inconsistent with the protocol")
    return spec
