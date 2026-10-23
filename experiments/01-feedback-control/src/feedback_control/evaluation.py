"""Episode-level evaluation for Motion Lab Experiment 01."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

from feedback_control.metrics import integrated_absolute_error

if TYPE_CHECKING:
    from feedback_control.simulation import EpisodeResult


@dataclass(frozen=True)
class EpisodeMetrics:
    integrated_absolute_error_rad_s: float


def evaluate_episode(
    result: EpisodeResult,
    *,
    control_interval_s: float,
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
    iae = integrated_absolute_error(
        positions_rad=(sample.position_rad for sample in result.samples),
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
    return EpisodeMetrics(integrated_absolute_error_rad_s=iae)
