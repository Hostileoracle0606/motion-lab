"""Strict versioned one-step evaluation protocol with file-relative paths."""

from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
import tomllib

from forward_dynamics.types import validate_finite_scalar


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG = REPOSITORY_ROOT / "experiments/03-forward-dynamics/configs/evaluation-v1.toml"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "experiments/03-forward-dynamics/results/evaluation-v1"
BASE_PREDICTORS = ("mujoco", "no_change", "constant_velocity")


@dataclass(frozen=True)
class EvaluationConfig:
    schema_version: int
    scope: str
    dataset_version: str
    dataset_dir: Path
    model_path: Path
    environment_id: str
    environment_version: str
    predictor_ids: tuple[str, ...]
    simulator_absolute_tolerance: float
    horizons: tuple[int, ...]
    analytical_timesteps_s: tuple[float, ...]


def load_evaluation_config(path: Path) -> EvaluationConfig:
    """Reject unsupported/ambiguous protocols before any output is created."""
    path = path.resolve()
    with path.open("rb") as source:
        raw = tomllib.load(source)
    if set(raw) != set(EvaluationConfig.__dataclass_fields__):
        raise ValueError("evaluation config fields do not match schema")
    if type(raw["schema_version"]) is not int or raw["schema_version"] != 1:
        raise ValueError("unsupported evaluation config schema_version")
    if raw["scope"] != "one_step":
        raise ValueError("scope must be one_step")
    if raw["dataset_version"] != "experiment-02-transition-v1":
        raise ValueError("unsupported dataset_version")
    for name in ("dataset_dir", "model_path", "environment_id", "environment_version"):
        if not isinstance(raw[name], str) or not raw[name].strip():
            raise ValueError(f"{name} must be a nonempty string")
    dataset_dir = (path.parent / raw["dataset_dir"]).resolve()
    model_path = (path.parent / raw["model_path"]).resolve()
    if not dataset_dir.is_dir() or not model_path.is_file():
        raise ValueError("dataset_dir and model_path must exist")
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != raw["environment_version"]:
        raise ValueError("environment_version does not match model checksum")
    ids = raw["predictor_ids"]
    if (not isinstance(ids, list) or not ids or
            any(not isinstance(item, str) or item not in BASE_PREDICTORS for item in ids) or
            len(set(ids)) != len(ids)):
        raise ValueError("predictor_ids must be unique supported adapter IDs")
    horizons = raw["horizons"]
    if (not isinstance(horizons, list) or not horizons or
            any(type(item) is not int or item <= 0 for item in horizons) or
            horizons != sorted(set(horizons))):
        raise ValueError("horizons must be increasing unique positive integers")
    tolerance = _positive("simulator_absolute_tolerance", raw["simulator_absolute_tolerance"])
    steps = raw["analytical_timesteps_s"]
    if not isinstance(steps, list) or not steps:
        raise ValueError("analytical_timesteps_s must be a nonempty list")
    steps = tuple(_positive("analytical_timesteps_s", step) for step in steps)
    if len(set(steps)) != len(steps):
        raise ValueError("analytical_timesteps_s must be unique")
    for step in steps:
        ratio = 0.010 / step
        if not math.isfinite(ratio) or not math.isclose(ratio, round(ratio), rel_tol=0, abs_tol=1e-12):
            raise ValueError("analytical timesteps must divide the control interval")
    return EvaluationConfig(
        1, "one_step", raw["dataset_version"], dataset_dir, model_path,
        raw["environment_id"], raw["environment_version"], tuple(ids), tolerance,
        tuple(horizons), steps,
    )


def protocol_mapping(config: EvaluationConfig, root: Path = REPOSITORY_ROOT) -> dict:
    """Canonical recorded protocol, independent of absolute checkout location."""
    payload = asdict(config)
    for name in ("dataset_dir", "model_path"):
        try:
            payload[name] = getattr(config, name).relative_to(root).as_posix()
        except ValueError as error:
            raise ValueError(f"{name} must be inside the evaluator checkout") from error
    return payload


def _positive(name: str, value: float) -> float:
    validate_finite_scalar(name, value)
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return float(value)
