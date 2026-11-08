"""Observed-state one-step predictions and pure grouped error aggregation."""

from collections import defaultdict
from dataclasses import asdict, dataclass

from forward_dynamics.dataset import PredictionCase
from forward_dynamics.metrics import StateResidual, residual, summarize_errors
from forward_dynamics.types import DynamicsState, Predictor, validate_prediction_inputs


# Shared domain identities for both prediction summaries and report coverage.
GROUP_FIELDS = (("episode", "episode_id"), ("scenario", "scenario_id"),
                ("controller", "controller_id"), ("split", "split"))


@dataclass(frozen=True)
class OneStepPrediction:
    case: PredictionCase
    predictor_id: str
    predicted_next_state: DynamicsState
    residual: StateResidual


def evaluate_one_step(
    cases: tuple[PredictionCase, ...], predictor: Predictor,
) -> tuple[OneStepPrediction, ...]:
    """Use each observation independently; never feed a prediction forward."""
    if not cases:
        raise ValueError("one-step evaluation requires at least one case")
    if not isinstance(predictor.predictor_id, str) or not predictor.predictor_id.strip():
        raise ValueError("predictor_id must be a nonempty string")
    rows = []
    seen = set()
    for case in cases:
        key = (case.episode_id, case.step_index)
        if key in seen:
            raise ValueError("duplicate prediction case")
        seen.add(key)
        try:
            validate_prediction_inputs(case.state, case.applied_torque_nm, case.context)
            predicted = predictor.predict(case.state, case.applied_torque_nm, case.context)
            error = residual(predicted, case.expected_next_state)
        except (ValueError, TypeError, ArithmeticError) as error:
            raise ValueError(
                f"predictor {predictor.predictor_id}, episode {case.episode_id}, "
                f"step {case.step_index}: {error}"
            ) from error
        rows.append(OneStepPrediction(case, predictor.predictor_id, predicted, error))
    return tuple(rows)


def summarize_predictions(predictions: tuple[OneStepPrediction, ...]) -> tuple[dict, ...]:
    """Keep episode/scenario/controller/split results, equal weight per row."""
    grouped = defaultdict(list)
    for row in predictions:
        for group, field in GROUP_FIELDS:
            key = (row.predictor_id, group, row.case.split, getattr(row.case, field))
            grouped[key].append(row.residual)
    return tuple(
        {"predictor_id": predictor_id, "group": group, "split": split,
         "key": key, "summary": asdict(summarize_errors(tuple(errors)))}
        for (predictor_id, group, split, key), errors in sorted(grouped.items())
    )
