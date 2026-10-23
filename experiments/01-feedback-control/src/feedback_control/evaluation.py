"""Episode-level evaluation for Motion Lab Experiment 01."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import TYPE_CHECKING

from feedback_control.config import MetricThresholds
from feedback_control.metrics import (
    TimingResult,
    integrated_absolute_error,
    normalized_maximum_overshoot,
    rise_time,
    saturation_fraction,
    settling_time,
    squared_control_effort,
    steady_state_absolute_error,
)

if TYPE_CHECKING:
    from feedback_control.simulation import EpisodeResult


class MetricAvailability(Enum):
    """Whether an episode contains enough samples for a windowed metric."""

    MEASURED = "measured"
    INSUFFICIENT_DURATION = "insufficient_duration"


@dataclass(frozen=True)
class EpisodeMetrics:
    integrated_absolute_error_rad_s: float
    rise_time: TimingResult
    normalized_maximum_overshoot: float
    settling_time: TimingResult
    steady_state_absolute_error_rad: float | None
    steady_state_error_availability: MetricAvailability
    squared_control_effort_nm2_s: float
    saturation_fraction: float


def evaluate_episode(
    result: EpisodeResult,
    *,
    control_interval_s: float,
    thresholds: MetricThresholds,
    torque_limit_nm: float,
) -> EpisodeMetrics:
    """Measure one static-target episode from its pre-action samples."""
    if not result.samples:
        raise ValueError("episode must contain at least one sample")
    target_position_rad = result.samples[0].target_position_rad
    if any(
        sample.target_position_rad != target_position_rad
        for sample in result.samples[1:]
    ):
        raise ValueError("episode must use one target position")
    positions = tuple(sample.position_rad for sample in result.samples)
    velocities = tuple(sample.velocity_rad_s for sample in result.samples)
    applied_torques = tuple(
        sample.applied_torque_nm for sample in result.samples
    )
    iae = integrated_absolute_error(
        positions_rad=positions,
        target_position_rad=target_position_rad,
        control_interval_s=control_interval_s,
    )
    observed_intervals = [
        later.time_s - earlier.time_s
        for earlier, later in zip(result.samples, result.samples[1:])
    ]
    observed_intervals.append(result.final_time_s - result.samples[-1].time_s)
    if any(
        not math.isclose(
            observed_interval_s,
            control_interval_s,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for observed_interval_s in observed_intervals
    ):
        raise ValueError("control_interval_s does not match the episode schedule")
    initial_position_rad = positions[0]
    sampled_duration_s = len(positions) * control_interval_s
    if thresholds.steady_state_window_s > sampled_duration_s:
        steady_state_error = None
        steady_state_availability = MetricAvailability.INSUFFICIENT_DURATION
    else:
        steady_state_error = steady_state_absolute_error(
            positions,
            target_position_rad,
            control_interval_s,
            thresholds.steady_state_window_s,
        )
        steady_state_availability = MetricAvailability.MEASURED
    return EpisodeMetrics(
        integrated_absolute_error_rad_s=iae,
        rise_time=rise_time(
            positions,
            initial_position_rad,
            target_position_rad,
            control_interval_s,
            thresholds.rise_lower_fraction,
            thresholds.rise_upper_fraction,
        ),
        normalized_maximum_overshoot=normalized_maximum_overshoot(
            positions,
            initial_position_rad,
            target_position_rad,
        ),
        settling_time=settling_time(
            positions,
            target_position_rad,
            control_interval_s,
            thresholds.settling_position_tolerance_rad,
            velocities,
            thresholds.settling_velocity_tolerance_rad_s,
        ),
        steady_state_absolute_error_rad=steady_state_error,
        steady_state_error_availability=steady_state_availability,
        squared_control_effort_nm2_s=squared_control_effort(
            applied_torques,
            control_interval_s,
        ),
        saturation_fraction=saturation_fraction(
            applied_torques,
            torque_limit_nm,
        ),
    )
