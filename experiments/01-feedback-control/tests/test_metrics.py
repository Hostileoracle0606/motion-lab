import math

import pytest

from feedback_control.metrics import integrated_absolute_error


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
