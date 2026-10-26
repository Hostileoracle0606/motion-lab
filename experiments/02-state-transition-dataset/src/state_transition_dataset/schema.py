"""Schema-v1 records for a versioned state-transition dataset."""

from dataclasses import asdict, dataclass, fields
from typing import Any, Mapping

from state_transition_dataset.config import SplitName


DATASET_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class TransitionRow:
    schema_version: int
    dataset_version: str
    split: SplitName
    episode_id: str
    scenario_id: str
    controller_id: str
    seed: int
    step_index: int
    time_s: float
    control_interval_s: float
    state_position_rad: float
    state_velocity_rad_s: float
    action_applied_torque_nm: float
    next_state_position_rad: float
    next_state_velocity_rad_s: float
    target_position_rad: float
    target_velocity_rad_s: float
    torque_limit_nm: float
    terminated: bool
    truncated: bool
    termination_reason: str | None
    environment_id: str
    environment_version: str
    source_code_version: str
    config_digest: str
    source_trajectory_sha256: str


@dataclass(frozen=True)
class EpisodeManifest:
    split: SplitName
    episode_id: str
    scenario_id: str
    controller_id: str
    seed: int
    transition_count: int
    source_trajectory_sha256: str
    replay_verified: bool
    max_position_error_rad: float
    max_velocity_error_rad_s: float


@dataclass(frozen=True)
class DatasetMetadata:
    schema_version: int
    dataset_version: str
    dataset_config_file: str
    dataset_config_sha256: str
    builder_code_version: str
    builder_working_tree_dirty: bool
    build_command: str
    source_evaluation_sha256: str
    source_code_version: str
    replay_absolute_tolerance: float
    split_policy: str
    episodes: tuple[EpisodeManifest, ...]


@dataclass(frozen=True)
class PartitionManifest:
    split: SplitName
    file: str
    sha256: str
    row_count: int
    episode_count: int
    scenario_count: int


@dataclass(frozen=True)
class DatasetManifest:
    metadata: DatasetMetadata
    partitions: tuple[PartitionManifest, ...]


@dataclass(frozen=True)
class ValidatedDataset:
    manifest: DatasetManifest
    rows_by_split: Mapping[SplitName, tuple[TransitionRow, ...]]


def transition_row_to_mapping(row: TransitionRow) -> dict[str, Any]:
    return asdict(row)


def transition_row_from_mapping(payload: Mapping[str, Any]) -> TransitionRow:
    _require_exact_fields("transition row", payload, TransitionRow)
    return TransitionRow(**payload)


def dataset_manifest_to_mapping(manifest: DatasetManifest) -> dict[str, Any]:
    return asdict(manifest)


def dataset_manifest_from_mapping(payload: Mapping[str, Any]) -> DatasetManifest:
    _require_exact_fields("dataset manifest", payload, DatasetManifest)
    metadata_payload = _require_mapping("metadata", payload["metadata"])
    _require_exact_fields("dataset metadata", metadata_payload, DatasetMetadata)
    episode_payloads = metadata_payload["episodes"]
    if not isinstance(episode_payloads, list):
        raise ValueError("dataset metadata episodes must be a list")
    episodes = tuple(
        _episode_manifest_from_mapping(_require_mapping("episode", episode))
        for episode in episode_payloads
    )
    metadata = DatasetMetadata(
        **{key: value for key, value in metadata_payload.items() if key != "episodes"},
        episodes=episodes,
    )

    partition_payloads = payload["partitions"]
    if not isinstance(partition_payloads, list):
        raise ValueError("dataset manifest partitions must be a list")
    partitions = tuple(
        _partition_manifest_from_mapping(
            _require_mapping("partition", partition)
        )
        for partition in partition_payloads
    )
    return DatasetManifest(metadata=metadata, partitions=partitions)


def _episode_manifest_from_mapping(payload: Mapping[str, Any]) -> EpisodeManifest:
    _require_exact_fields("episode manifest", payload, EpisodeManifest)
    return EpisodeManifest(**payload)


def _partition_manifest_from_mapping(
    payload: Mapping[str, Any],
) -> PartitionManifest:
    _require_exact_fields("partition manifest", payload, PartitionManifest)
    return PartitionManifest(**payload)


def _require_exact_fields(
    description: str,
    payload: Mapping[str, Any],
    record_type: type,
) -> None:
    expected = {field.name for field in fields(record_type)}
    if set(payload) != expected:
        raise ValueError(f"{description} fields do not match schema")


def _require_mapping(description: str, value: Any) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be an object")
    return value
