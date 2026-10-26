from pathlib import Path

import pytest

from state_transition_dataset.config import load_dataset_config


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = (
    ROOT
    / "experiments/02-state-transition-dataset/configs/dataset-v1.toml"
)
SOURCE_EVALUATION = (
    ROOT
    / "experiments/01-feedback-control/results/evaluation-v1"
).resolve()
SOURCE_CONFIG = (
    ROOT
    / "experiments/01-feedback-control/configs/evaluation-v1.toml"
).resolve()


def test_loads_versioned_dataset_config_and_scenario_splits() -> None:
    config = load_dataset_config(DEFAULT_CONFIG)

    assert config.schema_version == 1
    assert config.dataset_version == "experiment-02-transition-v1"
    assert config.source_evaluation_dir == SOURCE_EVALUATION
    assert config.source_evaluation_config == SOURCE_CONFIG
    assert config.replay_absolute_tolerance == 1e-12
    assert config.scenario_to_split == {
        "negative-to-positive-wide": "train",
        "positive-to-negative-wide": "train",
        "negative-to-positive-narrow": "validation",
        "positive-to-negative-narrow": "test",
    }


@pytest.mark.parametrize(
    ("split_block", "message"),
    [
        (
            """
train = ["negative-to-positive-wide", "positive-to-negative-wide"]
validation = ["negative-to-positive-narrow"]
test = ["negative-to-positive-narrow", "positive-to-negative-narrow"]
""",
            "scenario split assignments must be unique",
        ),
        (
            """
train = ["negative-to-positive-wide", "positive-to-negative-wide"]
validation = ["negative-to-positive-narrow"]
test = []
""",
            "scenario split assignments must match the source evaluation",
        ),
    ],
)
def test_rejects_duplicate_or_missing_scenario_assignments(
    tmp_path: Path,
    split_block: str,
    message: str,
) -> None:
    path = _write_config(tmp_path, split_block=split_block)

    with pytest.raises(ValueError, match=message):
        load_dataset_config(path)


@pytest.mark.parametrize("tolerance", [0.0, -1.0, float("inf")])
def test_rejects_invalid_replay_tolerance(
    tmp_path: Path,
    tolerance: float,
) -> None:
    path = _write_config(tmp_path, tolerance=tolerance)

    with pytest.raises(
        ValueError,
        match="replay_absolute_tolerance must be finite and greater than zero",
    ):
        load_dataset_config(path)


def test_rejects_unsupported_schema_version(tmp_path: Path) -> None:
    path = _write_config(tmp_path, schema_version=2)

    with pytest.raises(ValueError, match="unsupported dataset config schema version"):
        load_dataset_config(path)


def _write_config(
    tmp_path: Path,
    *,
    schema_version: int = 1,
    tolerance: float = 1e-12,
    split_block: str = """
train = ["negative-to-positive-wide", "positive-to-negative-wide"]
validation = ["negative-to-positive-narrow"]
test = ["positive-to-negative-narrow"]
""",
) -> Path:
    path = tmp_path / "dataset.toml"
    path.write_text(
        f"""schema_version = {schema_version}
dataset_version = "experiment-02-transition-v1"
replay_absolute_tolerance = {tolerance}

[source]
evaluation_dir = "{SOURCE_EVALUATION}"
evaluation_config = "{SOURCE_CONFIG}"

[splits]
{split_block.strip()}
""",
        encoding="utf-8",
    )
    return path
