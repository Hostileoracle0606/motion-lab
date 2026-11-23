import math
from dataclasses import FrozenInstanceError

import pytest

from forward_dynamics.types import DynamicsState
from model_predictive_control.contracts import EpisodeRecord, Scenario, StepRecord


def state(p, v=0.0):
    return DynamicsState(p, v)


def step(before, after, torque=1.0, physics_steps=5, ns=1_000_000, **kwargs):
    return StepRecord(before, after, torque, torque, physics_steps, ns,
                      kwargs.get('calls', 0), kwargs.get('missed', False),
                      kwargs.get('fallback'), kwargs.get('saturated', False),
                      action_ready_time_ns=kwargs.get('ready_ns'))


def episode(steps, termination='time_limit', seed=1, controller='mpc', target=1.0):
    return EpisodeRecord(Scenario('move', state(0), target, 3), controller,
                         seed, tuple(steps), termination)


def summarize(ep, hold=2):
    from model_predictive_control.metrics import summarize_episode
    return summarize_episode(ep, position_tolerance_rad=0.1,
                             velocity_tolerance_rad_s=0.1, hold_steps=hold)


def test_hand_calculated_trace_and_immutable_result():
    e = episode([step(state(0), state(0.5), torque=2, ns=1_000_000, calls=3),
                 step(state(0.5), state(1.05), torque=-1, ns=2_000_000, missed=True, fallback='deadline'),
                 step(state(1.05), state(1.02), torque=3, physics_steps=2, ns=5_000_000, saturated=True)])
    m = summarize(e)
    assert m.tracking_rmse_rad == pytest.approx(math.sqrt((1 + .25 + .05**2 + .02**2) / 4))
    assert m.effort_nm2_s == pytest.approx(.086)
    assert m.peak_overshoot_rad == pytest.approx(.05)
    assert m.final_error_rad == pytest.approx(-.02)
    assert m.success and m.settling_time_s == pytest.approx(.02)
    assert (m.completed_decisions, m.saturation_count, m.fallback_count, m.deadline_miss_count, m.predictor_calls) == (3, 1, 1, 1, 3)
    assert m.decision_time_p50_ms == 2
    assert m.decision_time_p99_ms == pytest.approx(4.94)
    with pytest.raises(FrozenInstanceError):
        m.success = False


def test_action_ready_timing_includes_fallback_work():
    m = summarize(episode([step(state(0), state(0.5), ns=2_000_000,
                                fallback="deadline", ready_ns=3_000_000)]))
    assert m.decision_time_p99_ms == 2
    assert m.action_ready_time_p50_ms == m.action_ready_time_p99_ms == 3


def test_early_termination_and_missing_terminal_state():
    e = episode([step(state(0), state(1)), step(state(1), None, physics_steps=1)], 'invalid_state')
    m = summarize(e, hold=1)
    assert not m.success and m.settling_time_s is None and m.final_error_rad is None
    assert m.termination == 'invalid_state'
    assert m.tracking_rmse_rad == pytest.approx(math.sqrt(.5))
    m = summarize(episode([step(state(0), state(float('nan')))], 'invalid_state'), hold=1)
    assert m.final_error_rad is None and not m.success


def test_empty_episode_has_no_observation_metrics_or_timings():
    m = summarize(episode([]))
    assert m.tracking_rmse_rad is None and m.peak_overshoot_rad is None
    assert m.final_error_rad is None and m.decision_time_p50_ms is None and m.decision_time_p99_ms is None
    assert m.effort_nm2_s == 0 and not m.success and m.completed_decisions == 0


def test_zero_target_motion_uses_absolute_displacement():
    m = summarize(episode([step(state(0), state(-.2))], target=0), hold=1)
    assert m.peak_overshoot_rad == pytest.approx(.2)


def test_cohort_counts_independent_seeds_and_retains_failures():
    from model_predictive_control.metrics import summarize_cohort
    good = episode([step(state(0), state(1))], seed=1)
    bad = episode([step(state(0), None)], termination='invalid_state', seed=2)
    kwargs = dict(position_tolerance_rad=.1, velocity_tolerance_rad_s=.1, hold_steps=1)
    c = summarize_cohort([good, bad], **kwargs)
    assert (c.episode_count, c.successes, c.success_rate) == (2, 1, .5)
    assert dict(c.terminations) == {'time_limit': 1, 'invalid_state': 1}
    assert [(x.scenario_id, x.seed) for x in c.episodes] == [('move', 1), ('move', 2)]
    with pytest.raises(ValueError):
        summarize_cohort([good, good], **kwargs)
    with pytest.raises(ValueError):
        summarize_cohort([good, episode([], controller='other', seed=2)], **kwargs)
    empty = summarize_cohort([], **kwargs)
    assert empty.episode_count == 0 and empty.success_rate is None and empty.controller_id is None
