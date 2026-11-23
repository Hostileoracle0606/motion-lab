"""Closed-loop metrics computed only from recorded episode data."""

from dataclasses import dataclass
import math

from .contracts import EpisodeRecord, PHYSICS_INTERVAL_S, Termination


@dataclass(frozen=True)
class EpisodeMetrics:
    scenario_id: str
    controller_id: str
    seed: int
    tracking_rmse_rad: float | None
    effort_nm2_s: float
    peak_overshoot_rad: float | None
    final_error_rad: float | None
    success: bool
    settling_time_s: float | None
    termination: Termination
    completed_decisions: int
    saturation_count: int
    fallback_count: int
    deadline_miss_count: int
    predictor_calls: int
    decision_time_p50_ms: float | None
    decision_time_p99_ms: float | None
    action_ready_time_p50_ms: float | None
    action_ready_time_p99_ms: float | None
    candidate_slots: int
    rejected_candidates: int


@dataclass(frozen=True)
class CohortMetrics:
    controller_id: str | None
    episode_count: int
    successes: int
    success_rate: float | None
    terminations: tuple[tuple[Termination, int], ...]
    episodes: tuple[EpisodeMetrics, ...]


def _finite_state(state):
    return state is not None and math.isfinite(state.position_rad) and math.isfinite(state.velocity_rad_s)


def _quantile(values, q):
    if not values:
        return None
    values = sorted(values)
    index = (len(values) - 1) * q
    lower = math.floor(index)
    upper = math.ceil(index)
    return values[lower] + (values[upper] - values[lower]) * (index - lower)


def _validate_options(position_tolerance_rad, velocity_tolerance_rad_s, hold_steps):
    for value in (position_tolerance_rad, velocity_tolerance_rad_s):
        if not math.isfinite(value) or value < 0:
            raise ValueError('tolerances must be finite and nonnegative')
    if isinstance(hold_steps, bool) or not isinstance(hold_steps, int) or hold_steps < 1:
        raise ValueError('hold_steps must be a positive integer')


def summarize_episode(episode: EpisodeRecord, *, position_tolerance_rad: float,
                      velocity_tolerance_rad_s: float, hold_steps: int) -> EpisodeMetrics:
    """Summarize a record; missing or nonfinite states cannot establish success."""
    _validate_options(position_tolerance_rad, velocity_tolerance_rad_s, hold_steps)
    steps = episode.steps
    target = episode.scenario.target_position_rad
    initial = steps[0].state if steps else None
    observations = ([initial] if _finite_state(initial) else [])
    observations += [s.next_state for s in steps if _finite_state(s.next_state)]
    errors = [s.position_rad - target for s in observations]
    rmse = math.sqrt(sum(e * e for e in errors) / len(errors)) if errors else None
    overshoot = None
    if _finite_state(initial) and errors:
        movement = target - initial.position_rad
        if movement == 0:
            overshoot = max(abs(e) for e in errors)
        else:
            direction = 1 if movement > 0 else -1
            overshoot = max(0.0, *(direction * e for e in errors))
    final = steps[-1].next_state if steps else None
    final_error = target - final.position_rad if _finite_state(final) else None
    elapsed = 0.0
    suffix_start = None
    suffix_length = 0
    for step in steps:
        elapsed += step.physics_steps * PHYSICS_INTERVAL_S
        state = step.next_state
        inside = (_finite_state(state) and abs(state.position_rad - target) <= position_tolerance_rad
                  and abs(state.velocity_rad_s) <= velocity_tolerance_rad_s)
        if inside:
            if suffix_length == 0:
                suffix_start = elapsed
            suffix_length += 1
        else:
            suffix_start = None
            suffix_length = 0
    success = episode.termination == 'time_limit' and suffix_length >= hold_steps
    times = [s.decision_time_ns / 1_000_000 for s in steps]
    action_ready_times = [
        s.action_ready_time_ns / 1_000_000
        for s in steps if s.action_ready_time_ns is not None
    ]
    return EpisodeMetrics(
        scenario_id=episode.scenario.scenario_id, controller_id=episode.controller_id,
        seed=episode.seed, tracking_rmse_rad=rmse,
        effort_nm2_s=sum(s.applied_torque_nm ** 2 * s.physics_steps * PHYSICS_INTERVAL_S for s in steps),
        peak_overshoot_rad=overshoot, final_error_rad=final_error, success=success,
        settling_time_s=suffix_start if success else None, termination=episode.termination,
        completed_decisions=len(steps), saturation_count=sum(s.saturated for s in steps),
        fallback_count=sum(s.fallback_reason is not None for s in steps),
        deadline_miss_count=sum(s.deadline_missed for s in steps),
        predictor_calls=sum(s.predictor_calls for s in steps),
        decision_time_p50_ms=_quantile(times, .5), decision_time_p99_ms=_quantile(times, .99),
        action_ready_time_p50_ms=_quantile(action_ready_times, .5),
        action_ready_time_p99_ms=_quantile(action_ready_times, .99),
        candidate_slots=sum(s.candidate_slots for s in steps),
        rejected_candidates=sum(s.rejected_candidates for s in steps),
    )


def summarize_cohort(episodes, *, position_tolerance_rad: float,
                     velocity_tolerance_rad_s: float, hold_steps: int) -> CohortMetrics:
    """Summarize one controller in input order; empty input has no rate or controller."""
    _validate_options(position_tolerance_rad, velocity_tolerance_rad_s, hold_steps)
    records = tuple(episodes)
    controllers = {e.controller_id for e in records}
    if len(controllers) > 1:
        raise ValueError('cohort must contain only one controller')
    keys = [(e.scenario.scenario_id, e.seed) for e in records]
    if len(set(keys)) != len(keys):
        raise ValueError('scenario and seed keys must be unique')
    summaries = tuple(summarize_episode(e, position_tolerance_rad=position_tolerance_rad,
                                       velocity_tolerance_rad_s=velocity_tolerance_rad_s,
                                       hold_steps=hold_steps) for e in records)
    counts = {}
    for summary in summaries:
        counts[summary.termination] = counts.get(summary.termination, 0) + 1
    successes = sum(s.success for s in summaries)
    return CohortMetrics(next(iter(controllers), None), len(summaries), successes,
                         successes / len(summaries) if summaries else None,
                         tuple(counts.items()), summaries)
