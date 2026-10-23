"""Trajectory metrics for Motion Lab Experiment 01."""

from collections.abc import Iterable
import math


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
