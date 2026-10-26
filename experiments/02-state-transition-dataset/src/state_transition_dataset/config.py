"""Versioned configuration for the state-transition dataset."""

from dataclasses import dataclass
import math
from pathlib import Path
import tomllib
from typing import Literal, cast

from feedback_control.config import load_evaluation_config


SplitName = Literal["train", "validation", "test"]
SPLIT_NAMES: tuple[SplitName, ...] = ("train", "validation", "test")


@dataclass(frozen=True)
class DatasetConfig:
    """Validated inputs and split policy for one dataset version."""

    schema_version: int
    dataset_version: str
    source_evaluation_dir: Path
    source_evaluation_config: Path
    replay_absolute_tolerance: float
    scenario_splits: tuple[tuple[str, SplitName], ...]

    @property
    def scenario_to_split(self) -> dict[str, SplitName]:
        """Return the scenario-level partition assignment."""
        return dict(self.scenario_splits)


def load_dataset_config(path: Path) -> DatasetConfig:
    """Load and validate a dataset config, resolving its source paths."""
    config_path = path.resolve()
    with config_path.open("rb") as config_file:
        raw = tomllib.load(config_file)

    schema_version = _require_integer("schema_version", raw["schema_version"])
    if schema_version != 1:
        raise ValueError(
            f"unsupported dataset config schema version: {schema_version}"
        )

    dataset_version = str(raw["dataset_version"])
    if not dataset_version:
        raise ValueError("dataset_version must not be empty")

    tolerance = float(raw["replay_absolute_tolerance"])
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError(
            "replay_absolute_tolerance must be finite and greater than zero"
        )

    source = raw["source"]
    evaluation_dir = (
        config_path.parent / str(source["evaluation_dir"])
    ).resolve()
    evaluation_config_path = (
        config_path.parent / str(source["evaluation_config"])
    ).resolve()
    if not evaluation_dir.is_dir():
        raise ValueError(f"source evaluation directory does not exist: {evaluation_dir}")
    if not evaluation_config_path.is_file():
        raise ValueError(
            f"source evaluation config does not exist: {evaluation_config_path}"
        )

    evaluation_config = load_evaluation_config(evaluation_config_path)
    expected_scenarios = {scenario.id for scenario in evaluation_config.scenarios}

    split_values = raw["splits"]
    scenario_splits: list[tuple[str, SplitName]] = []
    assigned_scenarios: list[str] = []
    for split in SPLIT_NAMES:
        scenarios = split_values[split]
        for scenario in scenarios:
            scenario_id = str(scenario)
            if not scenario_id:
                raise ValueError("scenario split assignments must not be empty")
            assigned_scenarios.append(scenario_id)
            scenario_splits.append((scenario_id, cast(SplitName, split)))

    if len(set(assigned_scenarios)) != len(assigned_scenarios):
        raise ValueError("scenario split assignments must be unique")
    if set(assigned_scenarios) != expected_scenarios:
        raise ValueError(
            "scenario split assignments must match the source evaluation"
        )

    return DatasetConfig(
        schema_version=schema_version,
        dataset_version=dataset_version,
        source_evaluation_dir=evaluation_dir,
        source_evaluation_config=evaluation_config_path,
        replay_absolute_tolerance=tolerance,
        scenario_splits=tuple(scenario_splits),
    )


def _require_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value
