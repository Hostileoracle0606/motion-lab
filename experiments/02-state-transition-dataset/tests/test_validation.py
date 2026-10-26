from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from state_transition_dataset.schema import (
    DatasetMetadata,
    EpisodeManifest,
    TransitionRow,
)
from state_transition_dataset.validation import (
    load_and_validate_dataset,
    write_dataset,
)


DATASET_VERSION = "experiment-02-transition-v1"


def test_writes_and_loads_deterministic_exact_schema(tmp_path: Path) -> None:
    metadata, rows = _valid_dataset()
    first = tmp_path / "first"
    second = tmp_path / "second"

    manifest = write_dataset(first, metadata, rows)
    write_dataset(second, metadata, rows)
    loaded = load_and_validate_dataset(first, DATASET_VERSION)

    for name in ("manifest.json", "train.jsonl", "validation.jsonl", "test.jsonl"):
        assert (first / name).read_bytes() == (second / name).read_bytes()
    assert loaded.manifest == manifest
    assert loaded.rows_by_split == rows
    row_payload = json.loads((first / "train.jsonl").read_text().splitlines()[0])
    assert set(row_payload) == {
        "schema_version",
        "dataset_version",
        "split",
        "episode_id",
        "scenario_id",
        "controller_id",
        "seed",
        "step_index",
        "time_s",
        "control_interval_s",
        "state_position_rad",
        "state_velocity_rad_s",
        "action_applied_torque_nm",
        "next_state_position_rad",
        "next_state_velocity_rad_s",
        "target_position_rad",
        "target_velocity_rad_s",
        "torque_limit_nm",
        "terminated",
        "truncated",
        "termination_reason",
        "environment_id",
        "environment_version",
        "source_code_version",
        "config_digest",
        "source_trajectory_sha256",
    }


def test_rejects_unknown_row_fields_even_with_updated_checksum(
    tmp_path: Path,
) -> None:
    metadata, rows = _valid_dataset()
    output = tmp_path / "dataset"
    write_dataset(output, metadata, rows)
    path = output / "train.jsonl"
    payloads = [json.loads(line) for line in path.read_text().splitlines()]
    payloads[0]["unexpected"] = True
    _rewrite_partition_and_checksum(output, "train", payloads)

    with pytest.raises(ValueError, match="transition row fields do not match schema"):
        load_and_validate_dataset(output, DATASET_VERSION)


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        ({"state_position_rad": float("nan")}, "must be finite"),
        ({"step_index": 2}, "step indices must be contiguous from zero"),
        ({"time_s": 0.5}, "times must follow the control schedule"),
        ({"state_position_rad": 99.0}, "state continuity"),
        ({"action_applied_torque_nm": 4.0}, "action exceeds torque limit"),
        ({"terminated": True, "termination_reason": "early"}, "final boundary"),
    ],
)
def test_rejects_invalid_transition_sequences_before_writing(
    tmp_path: Path,
    replacement: dict[str, object],
    message: str,
) -> None:
    metadata, rows = _valid_dataset()
    train_rows = list(rows["train"])
    train_rows[1] = replace(train_rows[1], **replacement)
    invalid_rows = {**rows, "train": tuple(train_rows)}
    output = tmp_path / "dataset"

    with pytest.raises(ValueError, match=message):
        write_dataset(output, metadata, invalid_rows)

    assert not output.exists()


def test_accepts_an_early_terminated_episode_with_one_final_boundary(
    tmp_path: Path,
) -> None:
    metadata, rows = _valid_dataset()
    first = replace(
        rows["train"][0],
        terminated=True,
        truncated=False,
        termination_reason="joint_limit_violation",
    )
    metadata = replace(
        metadata,
        episodes=(replace(metadata.episodes[0], transition_count=1),)
        + metadata.episodes[1:],
    )
    shortened = {**rows, "train": (first,) + rows["train"][2:]}

    write_dataset(tmp_path / "dataset", metadata, shortened)
    loaded = load_and_validate_dataset(
        tmp_path / "dataset",
        DATASET_VERSION,
    )

    assert loaded.rows_by_split["train"][0].termination_reason == (
        "joint_limit_violation"
    )


def test_rejects_partition_checksum_tampering(tmp_path: Path) -> None:
    metadata, rows = _valid_dataset()
    output = tmp_path / "dataset"
    write_dataset(output, metadata, rows)
    with (output / "train.jsonl").open("ab") as partition:
        partition.write(b"\n")

    with pytest.raises(ValueError, match="partition checksum mismatch"):
        load_and_validate_dataset(output, DATASET_VERSION)


def test_rejects_rows_without_declared_episode_membership(tmp_path: Path) -> None:
    metadata, rows = _valid_dataset()
    metadata = replace(metadata, episodes=metadata.episodes[1:])

    with pytest.raises(ValueError, match="undeclared episode"):
        write_dataset(tmp_path / "dataset", metadata, rows)


def test_rejects_duplicate_transition_keys(tmp_path: Path) -> None:
    metadata, rows = _valid_dataset()
    duplicated = {**rows, "train": rows["train"] + (rows["train"][0],)}

    with pytest.raises(ValueError, match="transition keys must be unique"):
        write_dataset(tmp_path / "dataset", metadata, duplicated)


@pytest.mark.parametrize("leakage", ["episode", "scenario"])
def test_rejects_episode_or_scenario_leakage(
    tmp_path: Path,
    leakage: str,
) -> None:
    metadata, rows = _valid_dataset()
    validation_row = rows["validation"][0]
    if leakage == "episode":
        leaked = replace(
            validation_row,
            episode_id=rows["train"][0].episode_id,
        )
    else:
        leaked = replace(
            validation_row,
            scenario_id=rows["train"][0].scenario_id,
        )
    invalid = {**rows, "validation": (leaked,)}

    with pytest.raises(ValueError, match=f"{leakage} leakage across splits"):
        write_dataset(tmp_path / "dataset", metadata, invalid)


def _valid_dataset() -> tuple[
    DatasetMetadata,
    dict[str, tuple[TransitionRow, ...]],
]:
    train_episode = _episode_rows(
        split="train",
        episode_id="p-wide-seed-0",
        scenario_id="wide",
    )
    validation_episode = _episode_rows(
        split="validation",
        episode_id="p-narrow-up-seed-0",
        scenario_id="narrow-up",
    )
    test_episode = _episode_rows(
        split="test",
        episode_id="p-narrow-down-seed-0",
        scenario_id="narrow-down",
    )
    rows = {
        "train": train_episode,
        "validation": validation_episode,
        "test": test_episode,
    }
    episodes = tuple(
        _episode_manifest(episode_rows)
        for episode_rows in rows.values()
    )
    metadata = DatasetMetadata(
        schema_version=1,
        dataset_version=DATASET_VERSION,
        dataset_config_file="dataset-v1.toml",
        dataset_config_sha256="9" * 64,
        builder_code_version="a" * 40,
        builder_working_tree_dirty=False,
        build_command="uv run python build_dataset.py",
        source_evaluation_sha256="b" * 64,
        source_code_version="c" * 40,
        replay_absolute_tolerance=1e-12,
        split_policy="complete scenarios",
        episodes=episodes,
    )
    return metadata, rows


def _episode_rows(
    *,
    split: str,
    episode_id: str,
    scenario_id: str,
) -> tuple[TransitionRow, ...]:
    common = {
        "schema_version": 1,
        "dataset_version": DATASET_VERSION,
        "split": split,
        "episode_id": episode_id,
        "scenario_id": scenario_id,
        "controller_id": "p",
        "seed": 0,
        "control_interval_s": 0.01,
        "target_position_rad": 1.0,
        "target_velocity_rad_s": 0.0,
        "torque_limit_nm": 3.0,
        "environment_id": "motion-lab/experiment-01/one-link",
        "environment_version": "d" * 64,
        "source_code_version": "c" * 40,
        "config_digest": "e" * 64,
        "source_trajectory_sha256": hashlib.sha256(
            episode_id.encode()
        ).hexdigest(),
    }
    return (
        TransitionRow(
            **common,
            step_index=0,
            time_s=0.0,
            state_position_rad=0.0,
            state_velocity_rad_s=0.0,
            action_applied_torque_nm=1.0,
            next_state_position_rad=0.01,
            next_state_velocity_rad_s=0.1,
            terminated=False,
            truncated=False,
            termination_reason=None,
        ),
        TransitionRow(
            **common,
            step_index=1,
            time_s=0.01,
            state_position_rad=0.01,
            state_velocity_rad_s=0.1,
            action_applied_torque_nm=0.5,
            next_state_position_rad=0.03,
            next_state_velocity_rad_s=0.2,
            terminated=False,
            truncated=True,
            termination_reason="time_limit",
        ),
    )


def _episode_manifest(rows: tuple[TransitionRow, ...]) -> EpisodeManifest:
    first = rows[0]
    return EpisodeManifest(
        split=first.split,
        episode_id=first.episode_id,
        scenario_id=first.scenario_id,
        controller_id=first.controller_id,
        seed=first.seed,
        transition_count=len(rows),
        source_trajectory_sha256=first.source_trajectory_sha256,
        replay_verified=True,
        max_position_error_rad=0.0,
        max_velocity_error_rad_s=0.0,
    )


def _rewrite_partition_and_checksum(
    output: Path,
    split: str,
    payloads: list[dict[str, object]],
) -> None:
    path = output / f"{split}.jsonl"
    content = "".join(
        json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True)
        + "\n"
        for payload in payloads
    ).encode()
    path.write_bytes(content)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for partition in manifest["partitions"]:
        if partition["split"] == split:
            partition["sha256"] = hashlib.sha256(content).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n"
    )
