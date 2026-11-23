import math
import pytest
from forward_dynamics.types import DynamicsState
from model_predictive_control.contracts import ControlDecision, PlantStep, Scenario
from model_predictive_control.closed_loop import PDController, run_episode

S = DynamicsState(0.0, 0.0)

class Plant:
    def __init__(self, termination=None):
        self.actions = []
        self.termination = termination
    def reset(self, state):
        return DynamicsState(0.1, 0.2)
    def advance(self, torque):
        self.actions.append(torque)
        return PlantStep(DynamicsState(0.1 + len(self.actions), 0.2), 2 if self.termination else 5, self.termination)

def scenario(steps=2):
    return Scenario('s', S, 1.0, steps)

def test_order_clipping_and_reset_state():
    plant = Plant()
    seen = []
    def controller(state, target):
        seen.append((state, target))
        return ControlDecision(4.0, 2)
    times = iter([0, 5, 10, 17])
    result = run_episode(plant, scenario(), controller, controller_id='c', seed=7, fallback=None, clock_ns=lambda: next(times))
    assert seen == [(DynamicsState(0.1, 0.2), 1.0), (DynamicsState(1.1, 0.2), 1.0)]
    assert plant.actions == [3.0, 3.0]
    assert [s.decision_time_ns for s in result.steps] == [5, 7]
    assert result.steps[0].requested_torque_nm == 4.0 and result.steps[0].saturated
    assert result.termination == 'time_limit'

@pytest.mark.parametrize(('command','elapsed','reason','recorded'), [(None, 5, 'no_plan', None), (math.nan, 5, 'invalid_command', None), (2.0, 6, 'deadline', 2.0), (None, 6, 'deadline', None)])
def test_fallback_reason_and_query_accounting(command, elapsed, reason, recorded):
    seen = []
    def fallback(state, target):
        seen.append(state)
        return ControlDecision(-4.0, 3)
    times = iter([0, elapsed])
    result = run_episode(Plant(), scenario(1), lambda s,t: ControlDecision(command, 2), controller_id='c', seed=1, fallback=fallback, deadline_ns=5, clock_ns=lambda: next(times))
    step = result.steps[0]
    assert step.fallback_reason == reason and step.requested_torque_nm == recorded
    assert step.predictor_calls == 5 and step.applied_torque_nm == -3 and step.saturated
    assert step.deadline_missed == (elapsed > 5)
    assert seen == [DynamicsState(0.1, 0.2)]

def test_invalid_fallback_is_error():
    with pytest.raises(ValueError, match='fallback'):
        run_episode(Plant(), scenario(1), lambda s,t: ControlDecision(None), controller_id='c', seed=0, fallback=lambda s,t: ControlDecision(math.inf))

def test_truth_termination_precedes_time_limit():
    result = run_episode(Plant('joint_limit'), scenario(1), lambda s,t: ControlDecision(0), controller_id='c', seed=0, fallback=None)
    assert result.termination == 'joint_limit' and result.steps[0].physics_steps == 2

def test_pd_returns_raw_torque():
    assert PDController(10, 2)(DynamicsState(0, 0.5), 1).requested_torque_nm == 9

def test_selected_torque_preserves_clipped_fallback():
    result = run_episode(Plant(), scenario(1), lambda s,t: ControlDecision(None), controller_id='c', seed=0, fallback=lambda s,t: ControlDecision(-4))
    assert result.steps[0].selected_torque_nm == -4
    assert result.steps[0].applied_torque_nm == -3


def test_missing_fallback_decision_is_configuration_error():
    with pytest.raises(ValueError, match='fallback'):
        run_episode(Plant(), scenario(1), lambda s,t: ControlDecision(None), controller_id='c', seed=0, fallback=lambda s,t: None)
