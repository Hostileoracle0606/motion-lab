import pytest

from forward_dynamics.types import DynamicsState
from model_predictive_control.closed_loop import IntegralFeedbackController, PDController


def test_integral_torque_accumulates_to_the_bound_and_unwinds():
    controller = IntegralFeedbackController(PDController(kp=1.0, kd=0.0), 1.0, 0.025)
    state = DynamicsState(0.0, 0.0)

    requests = [controller(state, 1.0).requested_torque_nm for _ in range(5)]
    reversed_request = controller(state, -1.0).requested_torque_nm

    assert requests == pytest.approx([1.01, 1.02, 1.025, 1.025, 1.025])
    assert reversed_request == pytest.approx(-0.985)


def test_saturation_does_not_build_integral_torque():
    controller = IntegralFeedbackController(PDController(kp=4.0, kd=0.0), 1.0, 1.0)
    for _ in range(20):
        assert controller(DynamicsState(0.0, 0.0), 1.0).requested_torque_nm == 4.0

    assert controller(DynamicsState(1.0, 0.0), 1.0).requested_torque_nm == 0.0


def test_zero_integral_gain_is_the_existing_pd_rule():
    pd = PDController(kp=2.0, kd=1.3)
    controller = IntegralFeedbackController(pd, 0.0, 0.25)
    state = DynamicsState(-0.1, 0.7)
    assert controller(state, 0.57) == pd(state, 0.57)
