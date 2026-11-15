from dataclasses import replace
from types import SimpleNamespace

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


def test_final_test_orchestration_excludes_cpu_and_repeat_diagnostics(
        tmp_path, synthetic_views, monkeypatch):
    from learned_dynamics.config import RunSpec
    from learned_dynamics.evaluation import BASELINE_IDS, run_evaluation
    from learned_dynamics.types import ModelSpec

    variants = (("canonical32", 32, 0, "one_step", "all"),
                ("capacity16", 16, 0, "one_step", "all"),
                ("history0", 32, 0, "one_step", "history_common"),
                ("history1", 32, 1, "one_step", "history_common"),
                ("loss1", 32, 0, "one_step", "loss_common"),
                ("loss5", 32, 0, "rollout_5", "loss_common"))
    runs = tuple(RunSpec(name, ModelSpec(width, history), loss, membership, seed, "mps", 0)
                 for name, width, history, loss, membership in variants for seed in range(5))
    runs += tuple(RunSpec("canonical32", ModelSpec(), "one_step", "all", 0, device, repeat)
                  for device, repeat in (("mps", 1), ("cpu", 0)))
    roster = {"schema_version": 1, "training_config_sha256": "a" * 64,
              "planned_runs": [{"run_id": run.run_id, "run_spec": run.mapping()} for run in runs],
              "results": [{"run_id": run.run_id, "status": "failed",
                           "failure_reason": "synthetic failed fit"} for run in runs]}
    config = SimpleNamespace(training_roster_dir=tmp_path / "training",
        output_dir=tmp_path / "output", dataset_dir=tmp_path / "dataset",
        dataset_version="experiment-02-transition-v1", baseline_ids=BASELINE_IDS,
        memberships=("all", "history_common", "loss_common"), horizons=(1,))
    captured = []
    monkeypatch.setattr("learned_dynamics.evaluation_config.load_evaluation_config", lambda path: config)
    monkeypatch.setattr("learned_dynamics.training_evidence.load_run_manifest", lambda path: roster)
    monkeypatch.setattr("learned_dynamics.data.load_views",
                        lambda *args: SimpleNamespace(test=SimpleNamespace(cases=synthetic_views[2].cases[:3])))
    monkeypatch.setattr("forward_dynamics.adapters.make_predictor", lambda *args: BiasedPredictor())
    monkeypatch.setattr("learned_dynamics.evidence.write_evidence",
                        lambda output, config, roster, rows, cases: captured.extend(rows) or output)
    run_evaluation(tmp_path / "config.toml", config.training_roster_dir, config.output_dir)
    learned_ids = {row["predictor_id"] for row in captured if row["candidate"]["kind"] != "baseline"}
    assert learned_ids == {f"{name}-mps-seed{seed}-repeat0"
                          for name in ("canonical32", "capacity16", "history0", "history1", "loss1", "loss5")
                          for seed in range(5)}


def test_grouped_metrics_keep_episode_denominators_and_training_seed(synthetic_views):
    first = synthetic_views[0].cases[:2]
    second = tuple(replace(case, episode_id="second-episode", scenario_id="second-scenario")
                   for case in first)
    candidate = learned(BiasedPredictor())
    rows = evaluate(first + second, (candidate,), (1,), "all")
    summaries = summarize_records(rows, group_by="episode_id")
    by_episode = {row["group_value"]: row for row in summaries}
    assert set(by_episode) == {"train-episode", "second-episode"}
    assert all(row["counts"] == {"attempted": 2, "eligible": 2,
               "success": 2, "failed": 0, "unavailable": 0} for row in summaries)
    assert all(row["training_seed"] == 0 for row in summaries)


def test_real_final_test_requires_the_complete_predeclared_seed_matrix():
    import pytest
    from learned_dynamics.evaluation import final_test_runs

    with pytest.raises(ValueError, match="all six variants and five MPS seeds"):
        final_test_runs({"planned_runs": []}, require_complete=True)
