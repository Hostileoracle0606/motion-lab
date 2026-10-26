"""Materialize dataset rows from validated Experiment 01 evidence."""

from dataclasses import dataclass
import math
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, cast

from feedback_control.report_evidence import (
    EpisodeEvidence,
    load_evaluation_evidence,
)
from feedback_control.trajectory import TransitionRecord
from state_transition_dataset.config import DatasetConfig, SPLIT_NAMES, SplitName
from state_transition_dataset.schema import (
    DATASET_SCHEMA_VERSION,
    TransitionRow,
)


@dataclass(frozen=True)
class RowBuild:
    """Rows plus the source episode objects required for deterministic replay."""

    rows_by_split: Mapping[SplitName, tuple[TransitionRow, ...]]
    episodes: tuple[EpisodeEvidence, ...]
    source_evaluation_sha256: str
    source_code_version: str
    model_path: Path


def build_rows(config: DatasetConfig) -> RowBuild:
    """Convert independently validated source evidence into schema-v1 rows."""
    evidence = load_evaluation_evidence(
        config.source_evaluation_dir,
        config.source_evaluation_config,
    )
    split_by_scenario = _validated_split_policy(config, evidence.episodes)
    partition_rows: dict[SplitName, list[TransitionRow]] = {
        split: [] for split in SPLIT_NAMES
    }
    for episode in evidence.episodes:
        split = split_by_scenario[episode.scenario_id]
        partition_rows[split].extend(
            _rows_for_episode(
                episode,
                split=split,
                dataset_version=config.dataset_version,
            )
        )

    source_code_version = evidence.provenance["code_version"]
    if not isinstance(source_code_version, str):
        raise ValueError("source evaluation code version must be a string")
    rows_by_split = MappingProxyType(
        {
            split: tuple(partition_rows[split])
            for split in SPLIT_NAMES
        }
    )
    return RowBuild(
        rows_by_split=rows_by_split,
        episodes=evidence.episodes,
        source_evaluation_sha256=evidence.evaluation_sha256,
        source_code_version=source_code_version,
        model_path=evidence.config.model.path,
    )


def _validated_split_policy(
    config: DatasetConfig,
    episodes: tuple[EpisodeEvidence, ...],
) -> dict[str, SplitName]:
    scenario_ids = [scenario_id for scenario_id, _ in config.scenario_splits]
    if len(set(scenario_ids)) != len(scenario_ids):
        raise ValueError("scenario split assignments must be unique")
    source_scenarios = {episode.scenario_id for episode in episodes}
    if set(scenario_ids) != source_scenarios:
        raise ValueError(
            "scenario split assignments must exactly match source evaluation scenarios"
        )
    if any(split not in SPLIT_NAMES for _, split in config.scenario_splits):
        raise ValueError("scenario split assignment uses an unsupported split")
    return dict(config.scenario_splits)


def _rows_for_episode(
    episode: EpisodeEvidence,
    *,
    split: SplitName,
    dataset_version: str,
) -> tuple[TransitionRow, ...]:
    trajectory = episode.trajectory
    configuration = trajectory.configuration
    rows: list[TransitionRow] = []
    for transition in trajectory.transitions:
        _require_finite_transition(episode.episode_id, transition)
        rows.append(
            TransitionRow(
                schema_version=DATASET_SCHEMA_VERSION,
                dataset_version=dataset_version,
                split=split,
                episode_id=episode.episode_id,
                scenario_id=episode.scenario_id,
                controller_id=episode.controller,
                seed=episode.seed,
                step_index=transition.step_index,
                time_s=transition.time_s,
                control_interval_s=configuration.control_interval_s,
                state_position_rad=cast(float, transition.state.position_rad),
                state_velocity_rad_s=cast(float, transition.state.velocity_rad_s),
                action_applied_torque_nm=transition.applied_torque_nm,
                next_state_position_rad=cast(
                    float,
                    transition.next_state.position_rad,
                ),
                next_state_velocity_rad_s=cast(
                    float,
                    transition.next_state.velocity_rad_s,
                ),
                target_position_rad=cast(float, transition.target.position_rad),
                target_velocity_rad_s=cast(float, transition.target.velocity_rad_s),
                torque_limit_nm=configuration.torque_limit_nm,
                terminated=transition.terminated,
                truncated=transition.truncated,
                termination_reason=transition.termination_reason,
                environment_id=trajectory.environment_id,
                environment_version=trajectory.environment_version,
                source_code_version=trajectory.code_version,
                config_digest=trajectory.config_digest,
                source_trajectory_sha256=episode.trajectory_sha256,
            )
        )
    return tuple(rows)


def _require_finite_transition(
    episode_id: str,
    transition: TransitionRecord,
) -> None:
    values = (
        transition.time_s,
        transition.state.position_rad,
        transition.state.velocity_rad_s,
        transition.applied_torque_nm,
        transition.next_state.position_rad,
        transition.next_state.velocity_rad_s,
        transition.target.position_rad,
        transition.target.velocity_rad_s,
    )
    if any(
        value is None
        or isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        for value in values
    ):
        raise ValueError(
            f"episode {episode_id} must contain finite source transition values"
        )
