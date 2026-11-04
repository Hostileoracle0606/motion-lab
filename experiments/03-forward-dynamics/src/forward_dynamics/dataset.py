"""Convert independently validated dataset rows into prediction cases."""

from dataclasses import dataclass
from pathlib import Path

from state_transition_dataset.config import SPLIT_NAMES, SplitName
from state_transition_dataset.validation import load_and_validate_dataset

from forward_dynamics.types import DynamicsContext, DynamicsState


@dataclass(frozen=True)
class PredictionCase:
    split: SplitName
    episode_id: str
    scenario_id: str
    controller_id: str
    seed: int
    step_index: int
    state: DynamicsState
    applied_torque_nm: float
    expected_next_state: DynamicsState
    context: DynamicsContext
    terminated: bool
    truncated: bool
    termination_reason: str | None


def load_prediction_cases(
    dataset_dir: Path, expected_dataset_version: str,
) -> tuple[PredictionCase, ...]:
    """Preserve complete partitions and terminal transitions; never fit data."""
    dataset = load_and_validate_dataset(dataset_dir, expected_dataset_version)
    return tuple(
        PredictionCase(
            split=row.split,
            episode_id=row.episode_id,
            scenario_id=row.scenario_id,
            controller_id=row.controller_id,
            seed=row.seed,
            step_index=row.step_index,
            state=DynamicsState(row.state_position_rad, row.state_velocity_rad_s),
            applied_torque_nm=row.action_applied_torque_nm,
            expected_next_state=DynamicsState(
                row.next_state_position_rad, row.next_state_velocity_rad_s,
            ),
            context=DynamicsContext(
                row.environment_id, row.environment_version,
                row.control_interval_s, row.torque_limit_nm,
            ),
            terminated=row.terminated,
            truncated=row.truncated,
            termination_reason=row.termination_reason,
        )
        for split in SPLIT_NAMES
        for row in sorted(dataset.rows_by_split[split], key=lambda r: (r.episode_id, r.step_index))
    )


def validate_case_contexts(
    cases: tuple[PredictionCase, ...], environment_id: str, environment_version: str,
) -> None:
    """Reject a dataset whose plant identity differs from the frozen protocol."""
    for case in cases:
        if (case.context.environment_id != environment_id or
                case.context.environment_version != environment_version):
            raise ValueError("dataset context does not match evaluation protocol")
