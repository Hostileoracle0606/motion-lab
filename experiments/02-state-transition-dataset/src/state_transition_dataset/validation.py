"""Deterministic serialization and independent validation for schema v1."""

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, cast

from state_transition_dataset.config import SPLIT_NAMES, SplitName
from state_transition_dataset.schema import (
    DATASET_SCHEMA_VERSION,
    DatasetManifest,
    DatasetMetadata,
    EpisodeManifest,
    PartitionManifest,
    TransitionRow,
    ValidatedDataset,
    dataset_manifest_from_mapping,
    dataset_manifest_to_mapping,
    transition_row_from_mapping,
    transition_row_to_mapping,
)


_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_GIT_SHA_PATTERN = re.compile(r"[0-9a-f]{40}")


def write_dataset(
    output_dir: Path,
    metadata: DatasetMetadata,
    rows_by_split: Mapping[SplitName, tuple[TransitionRow, ...]],
) -> DatasetManifest:
    """Validate then write deterministic JSONL partitions and their manifest."""
    normalized = _normalize_partitions(rows_by_split)
    _validate_dataset_content(metadata, normalized)

    partition_bytes = {
        split: _serialize_rows(normalized[split]) for split in SPLIT_NAMES
    }
    partitions = tuple(
        _partition_manifest(split, normalized[split], partition_bytes[split])
        for split in SPLIT_NAMES
    )
    manifest = DatasetManifest(metadata=metadata, partitions=partitions)
    manifest_bytes = _json_bytes(dataset_manifest_to_mapping(manifest), indent=2)

    resolved_output = output_dir.resolve()
    if resolved_output.exists() and not resolved_output.is_dir():
        raise ValueError("output_dir must be a directory")
    resolved_output.mkdir(parents=True, exist_ok=True)
    for split in SPLIT_NAMES:
        (resolved_output / f"{split}.jsonl").write_bytes(partition_bytes[split])
    (resolved_output / "manifest.json").write_bytes(manifest_bytes)
    return manifest


def load_and_validate_dataset(
    output_dir: Path,
    expected_dataset_version: str,
) -> ValidatedDataset:
    """Load dataset bytes as an independent consumer and validate every contract."""
    resolved_output = output_dir.resolve()
    manifest_path = resolved_output / "manifest.json"
    payload = _load_json_object(manifest_path)
    manifest = dataset_manifest_from_mapping(payload)
    if manifest.metadata.dataset_version != expected_dataset_version:
        raise ValueError("dataset version does not match expected version")
    _validate_metadata(manifest.metadata)

    partition_by_split: dict[SplitName, PartitionManifest] = {}
    for partition in manifest.partitions:
        _validate_partition_manifest(partition)
        if partition.split in partition_by_split:
            raise ValueError("partition splits must be unique")
        partition_by_split[partition.split] = partition
    if set(partition_by_split) != set(SPLIT_NAMES):
        raise ValueError("manifest must declare every dataset split")

    rows_by_split: dict[SplitName, tuple[TransitionRow, ...]] = {}
    for split in SPLIT_NAMES:
        partition = partition_by_split[split]
        expected_file = f"{split}.jsonl"
        if partition.file != expected_file:
            raise ValueError("partition file does not match split")
        partition_path = resolved_output / expected_file
        content = partition_path.read_bytes()
        if hashlib.sha256(content).hexdigest() != partition.sha256:
            raise ValueError(f"partition checksum mismatch: {split}")
        rows = _load_rows(content)
        if any(row.split != split for row in rows):
            raise ValueError("row split does not match partition")
        if len(rows) != partition.row_count:
            raise ValueError("partition row count mismatch")
        if len({row.episode_id for row in rows}) != partition.episode_count:
            raise ValueError("partition episode count mismatch")
        if len({row.scenario_id for row in rows}) != partition.scenario_count:
            raise ValueError("partition scenario count mismatch")
        rows_by_split[split] = rows

    _validate_dataset_content(manifest.metadata, rows_by_split)
    return ValidatedDataset(manifest=manifest, rows_by_split=rows_by_split)


def _normalize_partitions(
    rows_by_split: Mapping[SplitName, tuple[TransitionRow, ...]],
) -> dict[SplitName, tuple[TransitionRow, ...]]:
    if set(rows_by_split) != set(SPLIT_NAMES):
        raise ValueError("rows must declare every dataset split")
    normalized: dict[SplitName, tuple[TransitionRow, ...]] = {}
    for split in SPLIT_NAMES:
        rows = rows_by_split[split]
        if not isinstance(rows, tuple):
            raise ValueError("partition rows must be tuples")
        normalized[split] = rows
    return normalized


def _validate_dataset_content(
    metadata: DatasetMetadata,
    rows_by_split: Mapping[SplitName, tuple[TransitionRow, ...]],
) -> None:
    _validate_metadata(metadata)
    all_rows = tuple(
        row for split in SPLIT_NAMES for row in rows_by_split[split]
    )
    for split in SPLIT_NAMES:
        if any(row.split != split for row in rows_by_split[split]):
            raise ValueError("row split does not match partition")
    _validate_unique_keys_and_leakage(all_rows)

    episode_rows: dict[str, list[TransitionRow]] = defaultdict(list)
    for row in all_rows:
        _validate_row(row, metadata)
        episode_rows[row.episode_id].append(row)

    declared = {episode.episode_id: episode for episode in metadata.episodes}
    if len(declared) != len(metadata.episodes):
        raise ValueError("episode manifest identifiers must be unique")
    undeclared = set(episode_rows) - set(declared)
    if undeclared:
        raise ValueError(f"undeclared episode: {sorted(undeclared)[0]}")
    if set(declared) - set(episode_rows):
        raise ValueError("episode manifest declares missing rows")

    for episode_id, rows in episode_rows.items():
        _validate_episode(rows)
        _validate_episode_manifest(declared[episode_id], rows, metadata)


def _validate_metadata(metadata: DatasetMetadata) -> None:
    if type(metadata.schema_version) is not int or metadata.schema_version != 1:
        raise ValueError(
            f"unsupported dataset schema version: {metadata.schema_version}"
        )
    _require_nonempty("dataset_version", metadata.dataset_version)
    if not _GIT_SHA_PATTERN.fullmatch(metadata.builder_code_version):
        raise ValueError("builder_code_version must be a full lowercase Git SHA")
    if type(metadata.builder_working_tree_dirty) is not bool:
        raise ValueError("builder_working_tree_dirty must be boolean")
    _require_nonempty("build_command", metadata.build_command)
    _require_sha256("source_evaluation_sha256", metadata.source_evaluation_sha256)
    if not _GIT_SHA_PATTERN.fullmatch(metadata.source_code_version):
        raise ValueError("source_code_version must be a full lowercase Git SHA")
    _require_positive_finite(
        "replay_absolute_tolerance",
        metadata.replay_absolute_tolerance,
    )
    _require_nonempty("split_policy", metadata.split_policy)
    if not metadata.episodes:
        raise ValueError("dataset requires at least one episode")


def _validate_row(row: TransitionRow, metadata: DatasetMetadata) -> None:
    if type(row.schema_version) is not int or row.schema_version != DATASET_SCHEMA_VERSION:
        raise ValueError("unsupported transition row schema version")
    if row.dataset_version != metadata.dataset_version:
        raise ValueError("row dataset version mismatch")
    if row.split not in SPLIT_NAMES:
        raise ValueError("unsupported row split")
    for name in (
        "episode_id",
        "scenario_id",
        "controller_id",
        "environment_id",
    ):
        _require_nonempty(name, getattr(row, name))
    if type(row.seed) is not int or row.seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if type(row.step_index) is not int or row.step_index < 0:
        raise ValueError("step_index must be a non-negative integer")
    for name in (
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
    ):
        _require_finite(name, getattr(row, name))
    _require_positive_finite("control_interval_s", row.control_interval_s)
    _require_positive_finite("torque_limit_nm", row.torque_limit_nm)
    if abs(row.action_applied_torque_nm) > row.torque_limit_nm:
        raise ValueError("action exceeds torque limit")
    for name in ("terminated", "truncated"):
        if type(getattr(row, name)) is not bool:
            raise ValueError(f"{name} must be boolean")
    if row.terminated or row.truncated:
        _require_nonempty("termination_reason", row.termination_reason)
    elif row.termination_reason is not None:
        raise ValueError("termination_reason must be null before final boundary")
    _require_sha256("environment_version", row.environment_version)
    if not _GIT_SHA_PATTERN.fullmatch(row.source_code_version):
        raise ValueError("source_code_version must be a full lowercase Git SHA")
    _require_sha256("config_digest", row.config_digest)
    _require_sha256("source_trajectory_sha256", row.source_trajectory_sha256)


def _validate_unique_keys_and_leakage(rows: tuple[TransitionRow, ...]) -> None:
    episode_splits: dict[str, set[SplitName]] = defaultdict(set)
    scenario_splits: dict[str, set[SplitName]] = defaultdict(set)
    for row in rows:
        episode_splits[row.episode_id].add(row.split)
        scenario_splits[row.scenario_id].add(row.split)
    if any(len(splits) != 1 for splits in episode_splits.values()):
        raise ValueError("episode leakage across splits")
    if any(len(splits) != 1 for splits in scenario_splits.values()):
        raise ValueError("scenario leakage across splits")
    keys = tuple((row.episode_id, row.step_index) for row in rows)
    if len(set(keys)) != len(keys):
        raise ValueError("transition keys must be unique")


def _validate_episode(rows: list[TransitionRow]) -> None:
    first = rows[0]
    identity_fields = (
        "split",
        "scenario_id",
        "controller_id",
        "seed",
        "control_interval_s",
        "target_position_rad",
        "target_velocity_rad_s",
        "torque_limit_nm",
        "environment_id",
        "environment_version",
        "source_code_version",
        "config_digest",
        "source_trajectory_sha256",
    )
    for index, row in enumerate(rows):
        if row.step_index != index:
            raise ValueError("step indices must be contiguous from zero")
        expected_time = index * first.control_interval_s
        if not math.isclose(row.time_s, expected_time, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("times must follow the control schedule")
        if any(getattr(row, field) != getattr(first, field) for field in identity_fields):
            raise ValueError("episode context must remain consistent")
        if index and not (
            math.isclose(
                rows[index - 1].next_state_position_rad,
                row.state_position_rad,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            and math.isclose(
                rows[index - 1].next_state_velocity_rad_s,
                row.state_velocity_rad_s,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("state continuity is broken")
    boundary_count = sum(
        int(row.terminated) + int(row.truncated) for row in rows
    )
    if (
        boundary_count != 1
        or not (rows[-1].terminated or rows[-1].truncated)
    ):
        raise ValueError("episode must have exactly one final boundary")


def _validate_episode_manifest(
    episode: EpisodeManifest,
    rows: list[TransitionRow],
    metadata: DatasetMetadata,
) -> None:
    first = rows[0]
    expected = (
        first.split,
        first.episode_id,
        first.scenario_id,
        first.controller_id,
        first.seed,
        len(rows),
        first.source_trajectory_sha256,
    )
    actual = (
        episode.split,
        episode.episode_id,
        episode.scenario_id,
        episode.controller_id,
        episode.seed,
        episode.transition_count,
        episode.source_trajectory_sha256,
    )
    if actual != expected:
        raise ValueError("episode manifest does not match partition rows")
    if type(episode.replay_verified) is not bool or not episode.replay_verified:
        raise ValueError("episode replay must be verified")
    for name in ("max_position_error_rad", "max_velocity_error_rad_s"):
        value = getattr(episode, name)
        _require_finite(name, value)
        if value < 0 or value > metadata.replay_absolute_tolerance:
            raise ValueError("episode replay error exceeds tolerance")


def _partition_manifest(
    split: SplitName,
    rows: tuple[TransitionRow, ...],
    content: bytes,
) -> PartitionManifest:
    return PartitionManifest(
        split=split,
        file=f"{split}.jsonl",
        sha256=hashlib.sha256(content).hexdigest(),
        row_count=len(rows),
        episode_count=len({row.episode_id for row in rows}),
        scenario_count=len({row.scenario_id for row in rows}),
    )


def _validate_partition_manifest(partition: PartitionManifest) -> None:
    if partition.split not in SPLIT_NAMES:
        raise ValueError("unsupported partition split")
    _require_nonempty("partition file", partition.file)
    _require_sha256("partition sha256", partition.sha256)
    for name in ("row_count", "episode_count", "scenario_count"):
        value = getattr(partition, name)
        if type(value) is not int or value < 0:
            raise ValueError(f"partition {name} must be a non-negative integer")


def _serialize_rows(rows: tuple[TransitionRow, ...]) -> bytes:
    return b"".join(
        _json_bytes(transition_row_to_mapping(row), indent=None) for row in rows
    )


def _json_bytes(payload: Mapping[str, Any], *, indent: int | None) -> bytes:
    separators = None if indent is not None else (",", ":")
    return (
        json.dumps(
            payload,
            allow_nan=False,
            indent=indent,
            separators=separators,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _load_json_object(path: Path) -> Mapping[str, Any]:
    with path.open(encoding="utf-8") as input_file:
        payload = json.load(input_file)
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return payload


def _load_rows(content: bytes) -> tuple[TransitionRow, ...]:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("partition must be UTF-8") from error
    rows: list[TransitionRow] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line:
            raise ValueError(f"partition contains blank line {line_number}")
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError("transition row must be a JSON object")
        rows.append(transition_row_from_mapping(payload))
    return tuple(rows)


def _require_nonempty(name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must not be empty")


def _require_finite(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


def _require_positive_finite(name: str, value: object) -> None:
    _require_finite(name, value)
    if cast(float, value) <= 0:
        raise ValueError(f"{name} must be greater than zero")


def _require_sha256(name: str, value: object) -> None:
    if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
