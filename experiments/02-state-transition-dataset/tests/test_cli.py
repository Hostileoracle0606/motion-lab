from pathlib import Path

import pytest

import state_transition_dataset.generation as generation
from state_transition_dataset.generation import (
    DEFAULT_CONFIG,
    DEFAULT_OUTPUT,
    build_dataset,
)
from state_transition_dataset.validation import load_and_validate_dataset


ROOT = Path(__file__).resolve().parents[3]
DATASET_VERSION = "experiment-02-transition-v1"


def test_default_paths_are_repository_relative_contracts() -> None:
    assert DEFAULT_CONFIG == (
        ROOT / "experiments/02-state-transition-dataset/configs/dataset-v1.toml"
    )
    assert DEFAULT_OUTPUT == (
        ROOT / "experiments/02-state-transition-dataset/data/dataset-v1"
    )


def test_refuses_dirty_tree_before_writing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        generation,
        "_git_output",
        lambda *args: "dirty" if args[0] == "status" else "f" * 40,
    )
    output = tmp_path / "dataset"

    with pytest.raises(ValueError, match="working-tree changes are uncommitted"):
        build_dataset(DEFAULT_CONFIG, output)

    assert not output.exists()


def test_clean_build_records_provenance_and_validates_independently(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        generation,
        "_git_output",
        lambda *args: "f" * 40 if args[0] == "rev-parse" else "",
    )
    output = tmp_path / "dataset"

    built = build_dataset(DEFAULT_CONFIG, output)
    loaded = load_and_validate_dataset(output, DATASET_VERSION)

    assert built == loaded
    assert loaded.manifest.metadata.builder_code_version == "f" * 40
    assert loaded.manifest.metadata.builder_working_tree_dirty is False
    assert loaded.manifest.metadata.build_command == (
        "uv run python "
        "experiments/02-state-transition-dataset/scripts/build_dataset.py"
    )
    assert len(loaded.manifest.metadata.episodes) == 8
    assert {partition.split for partition in loaded.manifest.partitions} == {
        "train",
        "validation",
        "test",
    }


def test_repeated_builds_are_byte_identical(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        generation,
        "_git_output",
        lambda *args: "f" * 40 if args[0] == "rev-parse" else "",
    )
    first = tmp_path / "first"
    second = tmp_path / "second"

    build_dataset(DEFAULT_CONFIG, first)
    build_dataset(DEFAULT_CONFIG, second)

    for name in ("manifest.json", "train.jsonl", "validation.jsonl", "test.jsonl"):
        assert (first / name).read_bytes() == (second / name).read_bytes()
