"""Trajectory metrics for Motion Lab Experiment 01."""

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
import math


class TimingOutcome(Enum):
    """Whether a threshold-based duration was observed."""

    MEASURED = "measured"
    LOWER_THRESHOLD_NOT_REACHED = "lower_threshold_not_reached"
    UPPER_THRESHOLD_NOT_REACHED = "upper_threshold_not_reached"
    NOT_SETTLED = "not_settled"


@dataclass(frozen=True)
class TimingResult:
    """A measured time or an explicit reason that no time is defined."""

    time_s: float | None
    outcome: TimingOutcome


def saturation_fraction(
    applied_torques_nm: Iterable[float],
    torque_limit_nm: float,
) -> float:
    """Return the fraction of sampled actions at either actuator limit."""
    torques = tuple(applied_torques_nm)
    if not torques:
        raise ValueError("applied_torques_nm must contain at least one sample")
    if not all(math.isfinite(torque_nm) for torque_nm in torques):
        raise ValueError("applied_torques_nm must contain only finite values")
    if not math.isfinite(torque_limit_nm):
        raise ValueError("torque_limit_nm must be finite")
    if torque_limit_nm <= 0.0:
        raise ValueError("torque_limit_nm must be greater than zero")
    if any(abs(torque_nm) > torque_limit_nm for torque_nm in torques):
        raise ValueError("applied_torques_nm must stay within torque_limit_nm")
    saturated_steps = sum(abs(torque_nm) == torque_limit_nm for torque_nm in torques)
    return saturated_steps / len(torques)


def squared_control_effort(
    applied_torques_nm: Iterable[float],
    control_interval_s: float,
) -> float:
    """Return the applied-torque-squared control-effort proxy."""
    torques = tuple(applied_torques_nm)
    if not torques:
        raise ValueError("applied_torques_nm must contain at least one sample")
    if not all(math.isfinite(torque_nm) for torque_nm in torques):
        raise ValueError("applied_torques_nm must contain only finite values")
    if not math.isfinite(control_interval_s):
        raise ValueError("control_interval_s must be finite")
    if control_interval_s <= 0.0:
        raise ValueError("control_interval_s must be greater than zero")
    return math.fsum(
        torque_nm**2 * control_interval_s for torque_nm in torques
    )


def steady_state_absolute_error(
    positions_rad: Iterable[float],
    target_position_rad: float,
    control_interval_s: float,
    tail_window_s: float,
) -> float:
    """Return mean absolute position error over the final sampled time window."""
    positions = tuple(positions_rad)
    if not positions:
        raise ValueError("positions_rad must contain at least one sample")
    if not all(math.isfinite(position_rad) for position_rad in positions):
        raise ValueError("positions_rad must contain only finite values")
    if not math.isfinite(target_position_rad):
        raise ValueError("target_position_rad must be finite")
    if not math.isfinite(control_interval_s):
        raise ValueError("control_interval_s must be finite")
    if control_interval_s <= 0.0:
        raise ValueError("control_interval_s must be greater than zero")
    if not math.isfinite(tail_window_s):
        raise ValueError("tail_window_s must be finite")
    if tail_window_s <= 0.0:
        raise ValueError("tail_window_s must be greater than zero")
    tail_sample_ratio = tail_window_s / control_interval_s
    tail_sample_count = round(tail_sample_ratio)
    if not math.isclose(
        tail_sample_ratio,
        tail_sample_count,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "tail_window_s must be an integer multiple of control_interval_s"
        )
    if tail_sample_count > len(positions):
        raise ValueError("tail_window_s cannot exceed the sampled episode duration")
    tail_positions = positions[-tail_sample_count:]
    return math.fsum(
        abs(target_position_rad - position_rad) for position_rad in tail_positions
    ) / tail_sample_count


def settling_time(
    positions_rad: Iterable[float],
    target_position_rad: float,
    control_interval_s: float,
    position_tolerance_rad: float,
    velocities_rad_s: Iterable[float] | None = None,
    velocity_tolerance_rad_s: float | None = None,
) -> TimingResult:
    """Return the first sampled time after which all samples stay in band."""
    positions = tuple(positions_rad)
    if not positions:
        raise ValueError("positions_rad must contain at least one sample")
    if not all(math.isfinite(position_rad) for position_rad in positions):
        raise ValueError("positions_rad must contain only finite values")
    if not math.isfinite(target_position_rad):
        raise ValueError("target_position_rad must be finite")
    if not math.isfinite(control_interval_s):
        raise ValueError("control_interval_s must be finite")
    if control_interval_s <= 0.0:
        raise ValueError("control_interval_s must be greater than zero")
    if not math.isfinite(position_tolerance_rad):
        raise ValueError("position_tolerance_rad must be finite")
    if position_tolerance_rad <= 0.0:
        raise ValueError("position_tolerance_rad must be greater than zero")
    if (velocities_rad_s is None) != (velocity_tolerance_rad_s is None):
        raise ValueError(
            "velocities_rad_s and velocity_tolerance_rad_s must be provided together"
        )
    inside_position_band = tuple(
        abs(target_position_rad - position_rad) <= position_tolerance_rad
        for position_rad in positions
    )
    if velocities_rad_s is None:
        inside_band = inside_position_band
    else:
        velocities = tuple(velocities_rad_s)
        if len(velocities) != len(positions):
            raise ValueError(
                "velocities_rad_s must have the same number of samples as positions_rad"
            )
        if not all(math.isfinite(velocity_rad_s) for velocity_rad_s in velocities):
            raise ValueError("velocities_rad_s must contain only finite values")
        assert velocity_tolerance_rad_s is not None
        if not math.isfinite(velocity_tolerance_rad_s):
            raise ValueError("velocity_tolerance_rad_s must be finite")
        if velocity_tolerance_rad_s <= 0.0:
            raise ValueError("velocity_tolerance_rad_s must be greater than zero")
        inside_band = tuple(
            position_inside
            and abs(velocity_rad_s) <= velocity_tolerance_rad_s
            for position_inside, velocity_rad_s in zip(
                inside_position_band,
                velocities,
                strict=True,
            )
        )
    settling_index = next(
        (
            index
            for index in range(len(inside_band))
            if all(inside_band[index:])
        ),
        None,
    )
    if settling_index is None:
        return TimingResult(time_s=None, outcome=TimingOutcome.NOT_SETTLED)
    return TimingResult(
        time_s=settling_index * control_interval_s,
        outcome=TimingOutcome.MEASURED,
    )


def normalized_maximum_overshoot(
    positions_rad: Iterable[float],
    initial_position_rad: float,
    target_position_rad: float,
) -> float:
    """Return maximum overshoot as a fraction of the commanded displacement."""
    positions = tuple(positions_rad)
    if not positions:
        raise ValueError("positions_rad must contain at least one sample")
    if not all(math.isfinite(position_rad) for position_rad in positions):
        raise ValueError("positions_rad must contain only finite values")
    if not math.isfinite(initial_position_rad):
        raise ValueError("initial_position_rad must be finite")
    if not math.isfinite(target_position_rad):
        raise ValueError("target_position_rad must be finite")
    command = target_position_rad - initial_position_rad
    if command == 0.0:
        raise ValueError("target and initial positions must differ")
    direction = math.copysign(1.0, command)
    return max(
        0.0,
        max(
            (position_rad - target_position_rad) * direction / abs(command)
            for position_rad in positions
        ),
    )


def rise_time(
    positions_rad: Iterable[float],
    initial_position_rad: float,
    target_position_rad: float,
    control_interval_s: float,
    lower_fraction: float,
    upper_fraction: float,
) -> TimingResult:
    """Return sampled lower-to-upper threshold rise time in seconds."""
    positions = tuple(positions_rad)
    if not positions:
        raise ValueError("positions_rad must contain at least one sample")
    if not all(math.isfinite(position_rad) for position_rad in positions):
        raise ValueError("positions_rad must contain only finite values")
    if not math.isfinite(initial_position_rad):
        raise ValueError("initial_position_rad must be finite")
    if not math.isfinite(target_position_rad):
        raise ValueError("target_position_rad must be finite")
    if not math.isfinite(control_interval_s):
        raise ValueError("control_interval_s must be finite")
    if control_interval_s <= 0.0:
        raise ValueError("control_interval_s must be greater than zero")
    if not math.isfinite(lower_fraction) or not math.isfinite(upper_fraction):
        raise ValueError("rise fractions must be finite")
    if not 0.0 <= lower_fraction < upper_fraction <= 1.0:
        raise ValueError(
            "rise fractions must satisfy 0 <= lower_fraction < upper_fraction <= 1"
        )
    command = target_position_rad - initial_position_rad
    if command == 0.0:
        raise ValueError("target and initial positions must differ")
    direction = math.copysign(1.0, command)
    amplitude = abs(command)
    progress = tuple(
        (position_rad - initial_position_rad) * direction / amplitude
        for position_rad in positions
    )
    lower_index = next(
        (
            index
            for index, fraction in enumerate(progress)
            if fraction >= lower_fraction
        ),
        None,
    )
    if lower_index is None:
        return TimingResult(
            time_s=None,
            outcome=TimingOutcome.LOWER_THRESHOLD_NOT_REACHED,
        )
    upper_index = next(
        (
            index
            for index, fraction in enumerate(progress[lower_index:], start=lower_index)
            if fraction >= upper_fraction
        ),
        None,
    )
    if upper_index is None:
        return TimingResult(
            time_s=None,
            outcome=TimingOutcome.UPPER_THRESHOLD_NOT_REACHED,
        )
    return TimingResult(
        time_s=(upper_index - lower_index) * control_interval_s,
        outcome=TimingOutcome.MEASURED,
    )


def integrated_absolute_error(
    positions_rad: Iterable[float],
    target_position_rad: float,
    control_interval_s: float,
) -> float:
    """Return accumulated absolute position error in radian-seconds.

    Each position is the pre-action observation for one full control interval.
    A terminal state without a following action must not be included as another
    interval.
    """
    if not math.isfinite(target_position_rad):
        raise ValueError("target_position_rad must be finite")
    if not math.isfinite(control_interval_s):
        raise ValueError("control_interval_s must be finite")
    if control_interval_s <= 0.0:
        raise ValueError("control_interval_s must be greater than zero")
    positions = tuple(positions_rad)
    if not positions:
        raise ValueError("positions_rad must contain at least one sample")
    if not all(math.isfinite(position_rad) for position_rad in positions):
        raise ValueError("positions_rad must contain only finite values")
    return math.fsum(
        abs(target_position_rad - position_rad) * control_interval_s
        for position_rad in positions
    )
