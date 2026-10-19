import math

import pytest

from feedback_control.controllers import pd_control, proportional_control


def test_proportional_control_is_zero_at_target() -> None:
    output = proportional_control(
        target_position=1.0,
        position=1.0,
        kp=4.0,
        torque_limit=3.0,
    )

    assert output.raw_torque == pytest.approx(0.0)
    assert output.applied_torque == pytest.approx(0.0)
    assert output.saturated is False


def test_proportional_control_uses_position_error() -> None:
    output = proportional_control(
        target_position=1.0,
        position=0.25,
        kp=2.0,
        torque_limit=3.0,
    )

    assert output.raw_torque == pytest.approx(1.5)
    assert output.applied_torque == pytest.approx(1.5)


def test_pd_control_brakes_positive_velocity_at_target() -> None:
    output = pd_control(
        target_position=1.0,
        position=1.0,
        velocity=0.5,
        kp=4.0,
        kd=2.0,
        torque_limit=3.0,
    )

    assert output.raw_torque == pytest.approx(-1.0)
    assert output.applied_torque == pytest.approx(-1.0)


def test_pd_control_tracks_nonzero_target_velocity() -> None:
    output = pd_control(
        target_position=1.0,
        position=1.0,
        target_velocity=0.75,
        velocity=0.25,
        kp=4.0,
        kd=2.0,
        torque_limit=3.0,
    )

    assert output.raw_torque == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("target_position", "expected_raw", "expected_applied"),
    [(2.0, 8.0, 3.0), (-2.0, -8.0, -3.0)],
)
def test_proportional_control_clips_to_actuator_limit(
    target_position: float,
    expected_raw: float,
    expected_applied: float,
) -> None:
    output = proportional_control(
        target_position=target_position,
        position=0.0,
        kp=4.0,
        torque_limit=3.0,
    )

    assert output.raw_torque == pytest.approx(expected_raw)
    assert output.applied_torque == pytest.approx(expected_applied)
    assert output.saturated is True


def test_pd_with_zero_derivative_gain_matches_proportional_control() -> None:
    p_output = proportional_control(
        target_position=1.0,
        position=0.25,
        kp=2.0,
        torque_limit=3.0,
    )
    pd_output = pd_control(
        target_position=1.0,
        position=0.25,
        velocity=100.0,
        kp=2.0,
        kd=0.0,
        torque_limit=3.0,
    )

    assert pd_output == p_output


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"kp": -1.0}, "kp must be non-negative"),
        ({"torque_limit": 0.0}, "torque_limit must be greater than zero"),
        ({"position": math.nan}, "position must be finite"),
    ],
)
def test_proportional_control_rejects_invalid_parameters(
    kwargs: dict[str, float],
    message: str,
) -> None:
    parameters = {
        "target_position": 1.0,
        "position": 0.0,
        "kp": 2.0,
        "torque_limit": 3.0,
    }
    parameters.update(kwargs)

    with pytest.raises(ValueError, match=message):
        proportional_control(**parameters)
