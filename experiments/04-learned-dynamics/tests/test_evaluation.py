from dataclasses import replace

from forward_dynamics.types import DynamicsState
from learned_dynamics.evaluation import EvaluationCandidate, evaluate, summarize_records


class BiasedPredictor:
    predictor_id = "biased"

    def predict(self, state, applied_torque_nm, context):
        return DynamicsState(state.position_rad + 1.0, state.velocity_rad_s - 2.0)


class DivergingPredictor:
    predictor_id = "diverging"

    def predict(self, state, applied_torque_nm, context):
        raise ValueError("synthetic divergence")


class HistoryPredictor:
    predictor_id = "history"

    def predict_with_history(self, state, applied_torque_nm, previous_state,
                             previous_applied_torque_nm, context):
        return DynamicsState(state.position_rad + previous_state.position_rad + 1.0,
                             state.velocity_rad_s)


def baseline(predictor):
    return EvaluationCandidate.baseline(predictor.predictor_id, predictor)


def learned(predictor, history_steps=0, membership="all"):
    return EvaluationCandidate.learned(
        predictor_id=predictor.predictor_id, predictor=predictor,
        run_spec={"variant_id": "history1" if history_steps else "canonical32",
                  "model": {"hidden_width": 32, "history_steps": history_steps},
                  "loss": "one_step", "membership": membership,
                  "seed": 0, "device": "cpu", "repeat": 0},
        checkpoint_dir="run/checkpoint", checkpoint_sha256="a" * 64,
        selected_epoch=2, selected_validation_mse=0.25,
        training_config_sha256="b" * 64,
    )


def test_recursive_predictions_never_reset_to_future_observations(synthetic_views):
    cases = synthetic_views[0].cases[:3]
    rows = evaluate(cases, (baseline(BiasedPredictor()),), (2,), "all")
    first = next(row for row in rows if row["start_step_index"] == 0)
    assert first["status"] == "success"
    assert first["predicted_endpoint_state"]["position_rad"] == 2.0
    assert first["predicted_endpoint_state"]["velocity_rad_s"] == -4.0
    assert first["applied_actions_nm"] == [0.5, -0.5]


def test_unavailable_and_failed_attempts_remain_in_denominators(synthetic_views):
    cases = synthetic_views[0].cases[:3]
    rows = evaluate(cases, (baseline(BiasedPredictor()), baseline(DivergingPredictor())),
                    (1, 5), "all")
    summary = summarize_records(rows)
    by_key = {(row["predictor_id"], row["horizon_steps"]): row for row in summary}
    assert by_key[("biased", 1)]["counts"] == {
        "attempted": 3, "eligible": 3, "success": 3, "failed": 0, "unavailable": 0}
    assert by_key[("biased", 5)]["counts"] == {
        "attempted": 3, "eligible": 0, "success": 0, "failed": 0, "unavailable": 3}
    assert by_key[("diverging", 1)]["counts"] == {
        "attempted": 3, "eligible": 3, "success": 0, "failed": 3, "unavailable": 0}
    assert all(row["failure_step"] == 1 for row in rows
               if row["predictor_id"] == "diverging" and row["horizon_steps"] == 1)


def test_history_common_excludes_step_zero_and_shifts_predicted_history(synthetic_views):
    cases = synthetic_views[0].cases[:3]
    candidate = learned(HistoryPredictor(), history_steps=1, membership="history_common")
    rows = evaluate(cases, (candidate,), (2,), "history_common")
    assert rows[0]["status"] == "unavailable"
    assert rows[0]["unavailable_reason"] == "history_common requires a previous row"
    start_one = rows[1]
    assert start_one["status"] == "success"
    # step 1: .02 + 0 + 1 = 1.02; step 2 recursively uses 1.02 and .02.
    assert start_one["predicted_endpoint_state"]["position_rad"] == 2.04


def test_loss_common_and_terminal_boundaries_control_membership(synthetic_views):
    cases = tuple(replace(case, truncated=False, termination_reason=None)
                  for case in synthetic_views[0].cases[:6])
    cases = cases[:-1] + (replace(cases[-1], truncated=True,
                                 termination_reason="time_limit"),)
    rows = evaluate(cases, (baseline(BiasedPredictor()),), (1, 5), "loss_common")
    assert sum(row["status"] == "success" and row["horizon_steps"] == 1 for row in rows) == 2
    assert sum(row["status"] == "success" and row["horizon_steps"] == 5 for row in rows) == 2
    assert len(rows) == 12
