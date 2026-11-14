"""Strict, file-relative configuration for Stage 4 evaluation evidence."""

from dataclasses import dataclass
import math
from pathlib import Path
import tomllib

from forward_dynamics.serialization import require_fields, sha256_file
from learned_dynamics.evaluation import BASELINE_IDS, MEMBERSHIPS
from learned_dynamics.types import DATASET_VERSION


HORIZONS = (1, 5, 10, 25, 50, 100)
ARTIFACT_BUDGET_BYTES = 500 * 1024 * 1024
FIELDS = {
    "schema_version", "dataset_dir", "dataset_version", "training_roster_dir",
    "output_dir", "artifact_budget_bytes", "horizons", "memberships",
    "baseline_ids", "prediction_tolerance",
}


def _relative(config_path: Path, value: object, label: str) -> Path:
    if (not isinstance(value, str) or not value or value.startswith("~") or "$" in value or
            Path(value).is_absolute()):
        raise ValueError(f"{label} must be a safe literal relative path")
    return (config_path.parent / str(value)).resolve()


@dataclass(frozen=True)
class EvaluationConfig:
    schema_version: int
    config_path: Path
    config_sha256: str
    dataset_dir: Path
    dataset_version: str
    training_roster_dir: Path
    output_dir: Path
    artifact_budget_bytes: int
    horizons: tuple[int, ...]
    memberships: tuple[str, ...]
    baseline_ids: tuple[str, ...]
    prediction_tolerance: float

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported evaluation config schema")
        if self.dataset_version != DATASET_VERSION:
            raise ValueError("unsupported dataset version")
        if type(self.artifact_budget_bytes) is not int or self.artifact_budget_bytes != ARTIFACT_BUDGET_BYTES:
            raise ValueError("artifact budget does not match protocol")
        if self.horizons != HORIZONS or self.memberships != MEMBERSHIPS:
            raise ValueError("evaluation horizons/memberships do not match protocol")
        if self.baseline_ids != BASELINE_IDS:
            raise ValueError("evaluation must include exactly the six frozen baselines")
        if (type(self.prediction_tolerance) is not float or
                not math.isfinite(self.prediction_tolerance) or
                self.prediction_tolerance != 1e-12):
            raise ValueError("prediction tolerance does not match protocol")
        if len({self.dataset_dir, self.training_roster_dir, self.output_dir}) != 3:
            raise ValueError("evaluation inputs and output must be distinct")


def load_evaluation_config(path: Path) -> EvaluationConfig:
    config_path = Path(path).resolve()
    try:
        with config_path.open("rb") as stream:
            raw = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ValueError("cannot read evaluation config") from error
    require_fields(raw, FIELDS, "evaluation config")
    return EvaluationConfig(
        schema_version=raw["schema_version"], config_path=config_path,
        config_sha256=sha256_file(config_path),
        dataset_dir=_relative(config_path, raw["dataset_dir"], "dataset_dir"),
        dataset_version=raw["dataset_version"],
        training_roster_dir=_relative(config_path, raw["training_roster_dir"], "training_roster_dir"),
        output_dir=_relative(config_path, raw["output_dir"], "output_dir"),
        artifact_budget_bytes=raw["artifact_budget_bytes"],
        horizons=tuple(raw["horizons"]) if isinstance(raw["horizons"], list) else (),
        memberships=tuple(raw["memberships"]) if isinstance(raw["memberships"], list) else (),
        baseline_ids=tuple(raw["baseline_ids"]) if isinstance(raw["baseline_ids"], list) else (),
        prediction_tolerance=raw["prediction_tolerance"],
    )
