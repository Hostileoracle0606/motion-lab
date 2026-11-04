"""Recursive predictions under recorded actions, independent of controllers."""

from collections import defaultdict
from dataclasses import dataclass
import math

from state_transition_dataset.config import SPLIT_NAMES, SplitName

from forward_dynamics.dataset import PredictionCase
from forward_dynamics.metrics import StateResidual, residual
from forward_dynamics.types import (
    DynamicsState, Predictor, validate_finite_scalar, validate_prediction_inputs,
    validate_state,
)


# Match Experiment 02's continuity contract in rad and rad/s, not relative error.
_CONTINUITY_ABSOLUTE_TOLERANCE = 1e-12


@dataclass(frozen=True)
class RolloutPrediction:
    """One window's endpoint; the expected state is never a predictor input."""

    split: SplitName
    episode_id: str
    scenario_id: str
    controller_id: str
    seed: int
    start_step_index: int
    horizon_steps: int
    horizon_s: float
    predictor_id: str
    predicted_endpoint_state: DynamicsState
    expected_endpoint_state: DynamicsState
    residual: StateResidual


@dataclass(frozen=True)
class WindowCount:
    """Predictor-independent starts, including those lacking enough future rows."""

    split: SplitName
    episode_id: str
    horizon_steps: int
    eligible_count: int
    unavailable_count: int


@dataclass(frozen=True)
class RolloutEvaluation:
    """Keep unavailable-window evidence independently of endpoint predictions."""

    predictions: tuple[RolloutPrediction, ...]
    windows: tuple[WindowCount, ...]


def evaluate_rollouts(
    cases: tuple[PredictionCase, ...], predictor: Predictor, horizons: tuple[int, ...],
) -> RolloutEvaluation:
    """Feed predictions forward for every eligible recorded-action window.

    Cases may be contiguous episode slices; counts describe supplied rows only.
    Episodes follow dataset split order and episode ID, then increasing horizon
    and start step. A terminal transition can end a window but cannot precede
    another row. Full-coverage evaluation must supply all validated episodes.
    """
    if (not horizons or any(type(horizon) is not int or horizon <= 0 for horizon in horizons)
            or tuple(sorted(set(horizons))) != horizons):
        raise ValueError("horizons must be increasing unique positive integers")
    if not cases:
        raise ValueError("rollout evaluation requires at least one case")
    if not isinstance(predictor.predictor_id, str) or not predictor.predictor_id.strip():
        raise ValueError("predictor_id must be a nonempty string")
    episodes = _validated_episodes(cases, predictor.predictor_id)
    predictions = []
    windows = []
    for episode in episodes:
        first = episode[0]
        for horizon in horizons:
            eligible = max(len(episode) - horizon + 1, 0)
            windows.append(WindowCount(
                first.split, first.episode_id, horizon, eligible, len(episode) - eligible,
            ))
            for start in range(eligible):
                case = episode[start]
                predicted = _float_state(case.state)
                for action_case in episode[start:start + horizon]:
                    try:
                        validate_prediction_inputs(
                            predicted, action_case.applied_torque_nm, action_case.context,
                        )
                        predicted = predictor.predict(
                            predicted, float(action_case.applied_torque_nm), action_case.context,
                        )
                        validate_state(predicted)
                        predicted = _float_state(predicted)
                    except (ValueError, TypeError, ArithmeticError) as error:
                        raise _contextual_error(predictor.predictor_id, action_case, error) from error
                endpoint = episode[start + horizon - 1]
                try:
                    horizon_s = horizon * float(case.context.control_interval_s)
                    validate_finite_scalar("horizon_s", horizon_s)
                    error = residual(predicted, endpoint.expected_next_state)
                except (ValueError, TypeError, ArithmeticError) as error:
                    raise _contextual_error(predictor.predictor_id, endpoint, error) from error
                predictions.append(RolloutPrediction(
                    case.split, case.episode_id, case.scenario_id, case.controller_id,
                    case.seed, case.step_index, horizon, horizon_s, predictor.predictor_id,
                    predicted, endpoint.expected_next_state, error,
                ))
    return RolloutEvaluation(tuple(predictions), tuple(windows))


def _validated_episodes(
    cases: tuple[PredictionCase, ...], predictor_id: str,
) -> tuple[tuple[PredictionCase, ...], ...]:
    grouped = defaultdict(list)
    for case in cases:
        try:
            validate_prediction_inputs(case.state, case.applied_torque_nm, case.context)
            validate_state(case.expected_next_state)
            if type(case.step_index) is not int or case.step_index < 0:
                raise ValueError("step_index must be a nonnegative integer")
            if case.split not in SPLIT_NAMES:
                raise ValueError("unsupported split")
        except (ValueError, TypeError, ArithmeticError) as error:
            raise _contextual_error(predictor_id, case, error) from error
        grouped[case.episode_id].append(case)
    episodes = []
    for rows in grouped.values():
        episode = tuple(sorted(rows, key=lambda case: case.step_index))
        first = episode[0]
        for index, case in enumerate(episode):
            try:
                if any(getattr(case, field) != getattr(first, field) for field in (
                    "split", "scenario_id", "controller_id", "seed", "context",
                )):
                    raise ValueError("episode identity and context must remain consistent")
                if case.step_index != first.step_index + index:
                    raise ValueError("episode step indices must be unique and contiguous")
                if index:
                    previous = episode[index - 1]
                    if previous.terminated or previous.truncated:
                        raise ValueError("episode contains a row after a terminal boundary")
                    if not all(math.isclose(
                        float(getattr(previous.expected_next_state, field)),
                        float(getattr(case.state, field)), rel_tol=0.0,
                        abs_tol=_CONTINUITY_ABSOLUTE_TOLERANCE,
                    ) for field in ("position_rad", "velocity_rad_s")):
                        raise ValueError("episode state continuity is broken")
            except (ValueError, TypeError, ArithmeticError) as error:
                raise _contextual_error(predictor_id, case, error) from error
        episodes.append(episode)
    return tuple(sorted(episodes, key=lambda rows: (SPLIT_NAMES.index(rows[0].split), rows[0].episode_id)))


def _float_state(state: DynamicsState) -> DynamicsState:
    """Normalize accepted real scalars before recursive predictor arithmetic."""
    return DynamicsState(float(state.position_rad), float(state.velocity_rad_s))


def _contextual_error(predictor_id: str, case: PredictionCase, error: Exception) -> ValueError:
    return ValueError(
        f"predictor {predictor_id}, episode {case.episode_id}, step {case.step_index}: {error}"
    )
