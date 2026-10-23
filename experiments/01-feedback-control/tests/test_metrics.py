import math

import pytest

from feedback_control.metrics import (
    TimingOutcome,
    integrated_absolute_error,
    normalized_maximum_overshoot,
    rise_time,
    settling_time,
    saturation_fraction,
    squared_control_effort,
    steady_state_absolute_error,
)


def test_rise_time_measures_between_declared_progress_thresholds() -> None:
    result = rise_time(
        positions_rad=(0.0, 0.2, 0.6, 0.95, 1.0),
        initial_position_rad=0.0,
        target_position_rad=1.0,
        control_interval_s=0.1,
        lower_fraction=0.1,
        upper_fraction=0.9,
    )

    assert result.time_s == pytest.approx(0.2)
    assert result.outcome is TimingOutcome.MEASURED


def test_rise_time_reports_when_lower_threshold_is_not_reached() -> None:
    result = rise_time(
        positions_rad=(0.0, 0.02, 0.09),
        initial_position_rad=0.0,
        target_position_rad=1.0,
        control_interval_s=0.1,
        lower_fraction=0.1,
        upper_fraction=0.9,
    )

    assert result.time_s is None
    assert result.outcome is TimingOutcome.LOWER_THRESHOLD_NOT_REACHED


def test_rise_time_reports_when_upper_threshold_is_not_reached() -> None:
    result = rise_time(
        positions_rad=(0.0, 0.1, 0.4, 0.89),
        initial_position_rad=0.0,
        target_position_rad=1.0,
        control_interval_s=0.1,
        lower_fraction=0.1,
        upper_fraction=0.9,
    )

    assert result.time_s is None
    assert result.outcome is TimingOutcome.UPPER_THRESHOLD_NOT_REACHED


def test_rise_time_handles_motion_toward_a_lower_target() -> None:
    result = rise_time(
        positions_rad=(1.0, 0.8, 0.4, 0.05, 0.0),
        initial_position_rad=1.0,
        target_position_rad=0.0,
        control_interval_s=0.1,
        lower_fraction=0.1,
        upper_fraction=0.9,
    )

    assert result.time_s == pytest.approx(0.2)
    assert result.outcome is TimingOutcome.MEASURED


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"positions_rad": ()}, "positions_rad must contain at least one sample"),
        (
            {"positions_rad": (0.0, math.nan)},
            "positions_rad must contain only finite values",
        ),
        ({"initial_position_rad": math.inf}, "initial_position_rad must be finite"),
        ({"target_position_rad": math.nan}, "target_position_rad must be finite"),
        (
            {"target_position_rad": 0.0},
            "target and initial positions must differ",
        ),
        ({"control_interval_s": 0.0}, "control_interval_s must be greater than zero"),
        ({"control_interval_s": math.inf}, "control_interval_s must be finite"),
        ({"lower_fraction": -0.1}, "rise fractions must satisfy"),
        ({"upper_fraction": 1.1}, "rise fractions must satisfy"),
        ({"lower_fraction": 0.9}, "rise fractions must satisfy"),
        ({"lower_fraction": math.nan}, "rise fractions must be finite"),
    ],
)
def test_rise_time_rejects_invalid_inputs(
    kwargs: dict[str, object],
    message: str,
) -> None:
    parameters = {
        "positions_rad": (0.0, 0.5, 1.0),
        "initial_position_rad": 0.0,
        "target_position_rad": 1.0,
        "control_interval_s": 0.1,
        "lower_fraction": 0.1,
        "upper_fraction": 0.9,
    }
    parameters.update(kwargs)

    with pytest.raises(ValueError, match=message):
        rise_time(**parameters)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("positions_rad", "initial_position_rad", "target_position_rad"),
    [
        ((0.0, 0.8, 1.15, 1.02), 0.0, 1.0),
        ((1.0, 0.2, -0.15, -0.02), 1.0, 0.0),
    ],
)
def test_normalized_maximum_overshoot_is_direction_aware(
    positions_rad: tuple[float, ...],
    initial_position_rad: float,
    target_position_rad: float,
) -> None:
    result = normalized_maximum_overshoot(
        positions_rad=positions_rad,
        initial_position_rad=initial_position_rad,
        target_position_rad=target_position_rad,
    )

    assert result == pytest.approx(0.15)


def test_normalized_maximum_overshoot_is_zero_before_target_crossing() -> None:
    result = normalized_maximum_overshoot(
        positions_rad=(0.0, 0.4, 0.9, 1.0),
        initial_position_rad=0.0,
        target_position_rad=1.0,
    )

    assert result == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"positions_rad": ()}, "positions_rad must contain at least one sample"),
        (
            {"positions_rad": (0.0, math.inf)},
            "positions_rad must contain only finite values",
        ),
        ({"initial_position_rad": math.nan}, "initial_position_rad must be finite"),
        ({"target_position_rad": math.inf}, "target_position_rad must be finite"),
        (
            {"target_position_rad": 0.0},
            "target and initial positions must differ",
        ),
    ],
)
def test_normalized_maximum_overshoot_rejects_invalid_inputs(
    kwargs: dict[str, object],
    message: str,
) -> None:
    parameters = {
        "positions_rad": (0.0, 1.0),
        "initial_position_rad": 0.0,
        "target_position_rad": 1.0,
    }
    parameters.update(kwargs)

    with pytest.raises(ValueError, match=message):
        normalized_maximum_overshoot(**parameters)  # type: ignore[arg-type]


def test_settling_time_requires_position_to_remain_inside_band() -> None:
    result = settling_time(
        positions_rad=(0.0, 0.96, 1.06, 1.04, 1.02),
        target_position_rad=1.0,
        control_interval_s=0.1,
        position_tolerance_rad=0.05,
    )

    assert result.time_s == pytest.approx(0.3)
    assert result.outcome is TimingOutcome.MEASURED


def test_settling_time_reports_episode_that_never_settles() -> None:
    result = settling_time(
        positions_rad=(0.0, 0.96, 1.06),
        target_position_rad=1.0,
        control_interval_s=0.1,
        position_tolerance_rad=0.05,
    )

    assert result.time_s is None
    assert result.outcome is TimingOutcome.NOT_SETTLED


def test_settling_time_can_require_velocity_to_remain_inside_band() -> None:
    result = settling_time(
        positions_rad=(0.0, 0.96, 1.02, 1.01, 1.0),
        target_position_rad=1.0,
        control_interval_s=0.1,
        position_tolerance_rad=0.05,
        velocities_rad_s=(1.0, 0.4, -0.2, 0.08, 0.01),
        velocity_tolerance_rad_s=0.1,
    )

    assert result.time_s == pytest.approx(0.3)
    assert result.outcome is TimingOutcome.MEASURED


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"positions_rad": ()}, "positions_rad must contain at least one sample"),
        (
            {"positions_rad": (0.0, math.nan)},
            "positions_rad must contain only finite values",
        ),
        ({"target_position_rad": math.inf}, "target_position_rad must be finite"),
        ({"control_interval_s": 0.0}, "control_interval_s must be greater than zero"),
        ({"control_interval_s": math.nan}, "control_interval_s must be finite"),
        (
            {"position_tolerance_rad": 0.0},
            "position_tolerance_rad must be greater than zero",
        ),
        (
            {"position_tolerance_rad": math.inf},
            "position_tolerance_rad must be finite",
        ),
        (
            {"velocities_rad_s": (0.1, 0.0)},
            "velocities_rad_s and velocity_tolerance_rad_s must be provided together",
        ),
        (
            {"velocity_tolerance_rad_s": 0.1},
            "velocities_rad_s and velocity_tolerance_rad_s must be provided together",
        ),
        (
            {
                "velocities_rad_s": (0.1,),
                "velocity_tolerance_rad_s": 0.1,
            },
            "velocities_rad_s must have the same number of samples",
        ),
        (
            {
                "velocities_rad_s": (0.1, math.nan),
                "velocity_tolerance_rad_s": 0.1,
            },
            "velocities_rad_s must contain only finite values",
        ),
        (
            {
                "velocities_rad_s": (0.1, 0.0),
                "velocity_tolerance_rad_s": 0.0,
            },
            "velocity_tolerance_rad_s must be greater than zero",
        ),
        (
            {
                "velocities_rad_s": (0.1, 0.0),
                "velocity_tolerance_rad_s": math.inf,
            },
            "velocity_tolerance_rad_s must be finite",
        ),
    ],
)
def test_settling_time_rejects_invalid_inputs(
    kwargs: dict[str, object],
    message: str,
) -> None:
    parameters = {
        "positions_rad": (0.0, 1.0),
        "target_position_rad": 1.0,
        "control_interval_s": 0.1,
        "position_tolerance_rad": 0.05,
    }
    parameters.update(kwargs)

    with pytest.raises(ValueError, match=message):
        settling_time(**parameters)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("positions_rad", "target_position_rad"),
    [
        ((0.0, 0.8, 0.9, 1.1, 1.0), 1.0),
        ((1.0, 0.2, -0.4, -0.6, -0.5), -0.5),
    ],
)
def test_steady_state_absolute_error_averages_declared_tail_window(
    positions_rad: tuple[float, ...],
    target_position_rad: float,
) -> None:
    result = steady_state_absolute_error(
        positions_rad=positions_rad,
        target_position_rad=target_position_rad,
        control_interval_s=0.1,
        tail_window_s=0.3,
    )

    assert result == pytest.approx(1.0 / 15.0)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"positions_rad": ()}, "positions_rad must contain at least one sample"),
        (
            {"positions_rad": (0.9, math.nan)},
            "positions_rad must contain only finite values",
        ),
        ({"target_position_rad": math.inf}, "target_position_rad must be finite"),
        ({"control_interval_s": 0.0}, "control_interval_s must be greater than zero"),
        ({"control_interval_s": math.nan}, "control_interval_s must be finite"),
        ({"tail_window_s": 0.0}, "tail_window_s must be greater than zero"),
        ({"tail_window_s": math.inf}, "tail_window_s must be finite"),
        (
            {"tail_window_s": 0.25},
            "tail_window_s must be an integer multiple of control_interval_s",
        ),
        (
            {"tail_window_s": 0.4},
            "tail_window_s cannot exceed the sampled episode duration",
        ),
    ],
)
def test_steady_state_absolute_error_rejects_invalid_inputs(
    kwargs: dict[str, object],
    message: str,
) -> None:
    parameters = {
        "positions_rad": (0.9, 1.0, 1.0),
        "target_position_rad": 1.0,
        "control_interval_s": 0.1,
        "tail_window_s": 0.2,
    }
    parameters.update(kwargs)

    with pytest.raises(ValueError, match=message):
        steady_state_absolute_error(**parameters)  # type: ignore[arg-type]


def test_squared_control_effort_integrates_applied_torque_squared() -> None:
    result = squared_control_effort(
        applied_torques_nm=(2.0, -1.0, 0.5),
        control_interval_s=0.2,
    )

    assert result == pytest.approx(1.05)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"applied_torques_nm": ()},
            "applied_torques_nm must contain at least one sample",
        ),
        (
            {"applied_torques_nm": (1.0, math.inf)},
            "applied_torques_nm must contain only finite values",
        ),
        ({"control_interval_s": 0.0}, "control_interval_s must be greater than zero"),
        ({"control_interval_s": math.nan}, "control_interval_s must be finite"),
    ],
)
def test_squared_control_effort_rejects_invalid_inputs(
    kwargs: dict[str, object],
    message: str,
) -> None:
    parameters = {
        "applied_torques_nm": (1.0, 0.0),
        "control_interval_s": 0.1,
    }
    parameters.update(kwargs)

    with pytest.raises(ValueError, match=message):
        squared_control_effort(**parameters)  # type: ignore[arg-type]


def test_saturation_fraction_counts_steps_at_either_actuator_limit() -> None:
    result = saturation_fraction(
        applied_torques_nm=(-3.0, -2.0, 0.0, 3.0),
        torque_limit_nm=3.0,
    )

    assert result == pytest.approx(0.5)


def test_saturation_fraction_is_zero_when_no_action_reaches_limit() -> None:
    result = saturation_fraction(
        applied_torques_nm=(-2.99, 0.0, 2.99),
        torque_limit_nm=3.0,
    )

    assert result == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"applied_torques_nm": ()},
            "applied_torques_nm must contain at least one sample",
        ),
        (
            {"applied_torques_nm": (0.0, math.nan)},
            "applied_torques_nm must contain only finite values",
        ),
        ({"torque_limit_nm": 0.0}, "torque_limit_nm must be greater than zero"),
        ({"torque_limit_nm": math.inf}, "torque_limit_nm must be finite"),
        (
            {"applied_torques_nm": (0.0, 3.1)},
            "applied_torques_nm must stay within torque_limit_nm",
        ),
    ],
)
def test_saturation_fraction_rejects_invalid_inputs(
    kwargs: dict[str, object],
    message: str,
) -> None:
    parameters = {
        "applied_torques_nm": (-3.0, 0.0, 3.0),
        "torque_limit_nm": 3.0,
    }
    parameters.update(kwargs)

    with pytest.raises(ValueError, match=message):
        saturation_fraction(**parameters)  # type: ignore[arg-type]


def test_integrated_absolute_error_matches_worked_example() -> None:
    result = integrated_absolute_error(
        positions_rad=(0.0, 0.4, 0.8, 1.1),
        target_position_rad=1.0,
        control_interval_s=0.1,
    )

    assert result == pytest.approx(0.19)


def test_integrated_absolute_error_is_zero_at_target() -> None:
    result = integrated_absolute_error(
        positions_rad=(1.0, 1.0, 1.0),
        target_position_rad=1.0,
        control_interval_s=0.1,
    )

    assert result == pytest.approx(0.0)


def test_integrated_absolute_error_handles_negative_direction_motion() -> None:
    result = integrated_absolute_error(
        positions_rad=(1.0, 0.4, 0.0, -0.4),
        target_position_rad=-0.5,
        control_interval_s=0.1,
    )

    assert result == pytest.approx(0.30)


def test_integrated_absolute_error_does_not_cancel_overshoot() -> None:
    result = integrated_absolute_error(
        positions_rad=(0.8, 1.2),
        target_position_rad=1.0,
        control_interval_s=0.1,
    )

    assert result == pytest.approx(0.04)


@pytest.mark.parametrize("control_interval_s", [0.0, -0.1])
def test_integrated_absolute_error_rejects_non_positive_control_interval(
    control_interval_s: float,
) -> None:
    with pytest.raises(ValueError, match="control_interval_s must be greater than zero"):
        integrated_absolute_error(
            positions_rad=(0.0, 0.4),
            target_position_rad=1.0,
            control_interval_s=control_interval_s,
        )


@pytest.mark.parametrize("control_interval_s", [math.nan, math.inf])
def test_integrated_absolute_error_rejects_non_finite_control_interval(
    control_interval_s: float,
) -> None:
    with pytest.raises(ValueError, match="control_interval_s must be finite"):
        integrated_absolute_error(
            positions_rad=(0.0, 0.4),
            target_position_rad=1.0,
            control_interval_s=control_interval_s,
        )


def test_integrated_absolute_error_rejects_empty_trajectory() -> None:
    with pytest.raises(ValueError, match="positions_rad must contain at least one sample"):
        integrated_absolute_error(
            positions_rad=(),
            target_position_rad=1.0,
            control_interval_s=0.1,
        )


@pytest.mark.parametrize("position_rad", [math.nan, math.inf, -math.inf])
def test_integrated_absolute_error_rejects_non_finite_positions(
    position_rad: float,
) -> None:
    with pytest.raises(ValueError, match="positions_rad must contain only finite values"):
        integrated_absolute_error(
            positions_rad=(0.0, position_rad),
            target_position_rad=1.0,
            control_interval_s=0.1,
        )


@pytest.mark.parametrize("target_position_rad", [math.nan, math.inf, -math.inf])
def test_integrated_absolute_error_rejects_non_finite_target(
    target_position_rad: float,
) -> None:
    with pytest.raises(ValueError, match="target_position_rad must be finite"):
        integrated_absolute_error(
            positions_rad=(0.0, 0.4),
            target_position_rad=target_position_rad,
            control_interval_s=0.1,
        )
