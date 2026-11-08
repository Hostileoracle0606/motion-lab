"""Dataset-derived window membership, endpoint audit and grouped statistics."""

from collections import defaultdict
from dataclasses import asdict

from forward_dynamics.adapters import make_predictor
from forward_dynamics.config import EvaluationConfig
from forward_dynamics.dataset import PredictionCase
from forward_dynamics.evaluation import GROUP_FIELDS
from forward_dynamics.metrics import residual, summarize_errors
from forward_dynamics.rollouts import RolloutPrediction, WindowCount
from forward_dynamics.serialization import require_equal, require_fields
from forward_dynamics.types import DynamicsState, validate_state


def episode_cases(cases: tuple[PredictionCase, ...]) -> tuple[tuple[PredictionCase, ...], ...]:
    episodes = {}
    for case in cases:
        episodes.setdefault(case.episode_id, []).append(case)
    return tuple(tuple(rows) for rows in episodes.values())


def window_counts(cases: tuple[PredictionCase, ...], horizons: tuple[int, ...]) -> tuple[WindowCount, ...]:
    return tuple(WindowCount(rows[0].split, rows[0].episode_id, horizon,
                             max(len(rows) - horizon + 1, 0),
                             min(horizon - 1, len(rows)))
                 for rows in episode_cases(cases) for horizon in horizons)


def episode_outcomes(cases: tuple[PredictionCase, ...]) -> tuple[dict, ...]:
    return tuple({"split": rows[0].split, "episode_id": rows[0].episode_id,
                  "scenario_id": rows[0].scenario_id, "controller_id": rows[0].controller_id,
                  "seed": rows[0].seed, "transition_count": len(rows),
                  "terminated": rows[-1].terminated, "truncated": rows[-1].truncated,
                  "termination_reason": rows[-1].termination_reason}
                 for rows in episode_cases(cases))


def summarize_rollouts(predictions: tuple[RolloutPrediction, ...]) -> tuple[dict, ...]:
    grouped = defaultdict(list)
    for row in predictions:
        for group, field in GROUP_FIELDS:
            grouped[(row.predictor_id, group, row.split, getattr(row, field), row.horizon_steps)].append(row.residual)
    return tuple({"predictor_id": name, "group": group, "split": split,
                  "key": key, "horizon_steps": horizon,
                  "summary": asdict(summarize_errors(tuple(errors)))}
                 for (name, group, split, key, horizon), errors in sorted(grouped.items()))


def validate_rollout_records(
    records: list[dict], cases: tuple[PredictionCase, ...], config: EvaluationConfig,
) -> tuple[RolloutPrediction, ...]:
    """Reconstruct identities/actions/endpoints from original validated cases.

    Reference endpoints use the recorded simulator trajectory and tolerance.
    Baseline equations are audited separately after metadata/grouping validation.
    """
    expected_count = len(config.predictor_ids) * sum(row.eligible_count for row in window_counts(cases, config.horizons))
    if len(records) != expected_count:
        raise ValueError("rollout membership/count mismatch")
    result = []
    index = 0
    for name in config.predictor_ids:
        for rows in episode_cases(cases):
            first = rows[0]
            for horizon in config.horizons:
                for start in range(max(len(rows) - horizon + 1, 0)):
                    payload = records[index]
                    index += 1
                    require_fields(payload, set(RolloutPrediction.__dataclass_fields__), "rollout")
                    raw = payload["predicted_endpoint_state"]
                    require_fields(raw, {"position_rad", "velocity_rad_s"}, "rollout predicted state")
                    predicted = DynamicsState(**raw)
                    validate_state(predicted)
                    expected = rows[start + horizon - 1].expected_next_state
                    error = residual(predicted, expected)
                    row = RolloutPrediction(first.split, first.episode_id, first.scenario_id,
                        first.controller_id, first.seed, rows[start].step_index, horizon,
                        horizon * float(first.context.control_interval_s), name, predicted, expected, error)
                    require_equal(payload, asdict(row), "rollout identity/endpoint/residual")
                    if name == "mujoco":
                        if max(abs(error.position_error_rad), abs(error.velocity_error_rad_s)) > config.simulator_absolute_tolerance:
                            raise ValueError("simulator rollout exceeds recorded-state tolerance")
                    result.append(row)
    return tuple(result)


def validate_rollout_baselines(
    predictions: tuple[RolloutPrediction, ...], cases: tuple[PredictionCase, ...],
    config: EvaluationConfig,
) -> None:
    """Audit every baseline endpoint using shared original-action prefixes.

    For each original observed start, integrate once to the largest eligible
    horizon and check every requested intermediate endpoint. This is exactly
    the recursive trajectory of each window; saved outputs and expected future
    states never initialize or advance it. Sharing prefixes avoids recomputing
    the same deterministic numerical work for nested horizons.
    """
    saved = {(row.predictor_id, row.episode_id, row.start_step_index, row.horizon_steps): row
             for row in predictions}
    for name in config.predictor_ids:
        if name == "mujoco":
            continue
        predictor = make_predictor(name, config)
        for rows in episode_cases(cases):
            for start, case in enumerate(rows):
                state = case.state
                eligible = [horizon for horizon in config.horizons if horizon <= len(rows) - start]
                if not eligible:
                    continue
                for offset, action in enumerate(rows[start:start + max(eligible)], 1):
                    state = predictor.predict(state, action.applied_torque_nm, action.context)
                    if offset in eligible:
                        row = saved[(name, case.episode_id, case.step_index, offset)]
                        require_equal(asdict(row.predicted_endpoint_state), asdict(state),
                                      "rollout baseline prediction under original actions")
