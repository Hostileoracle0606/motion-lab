"""Episode-local history and rollout windows with disjoint exclusion counts."""

from learned_dynamics.types import (
    SplitView, TrainingView, Window, WindowCoverage, checked_episodes,
)


def make_windows(
    view: SplitView | TrainingView, horizon: int, history_steps: int = 0,
) -> tuple[tuple[Window, ...], tuple[WindowCoverage, ...]]:
    """Count unavailable future first, then missing history; all starts accounted for."""
    if not isinstance(view, (SplitView, TrainingView)):
        raise ValueError("windows require a checked split/training view")
    if type(horizon) is not int or horizon <= 0:
        raise ValueError("horizon must be a positive integer")
    if type(history_steps) is not int or history_steps not in (0, 1):
        raise ValueError("history_steps must be zero or one")
    windows, coverage = [], []
    for rows in checked_episodes(view.cases):
        eligible, unavailable, missing_history = 0, 0, 0
        for start in range(len(rows)):
            if start + horizon > len(rows):
                unavailable += 1
            elif history_steps and start == 0:
                missing_history += 1
            else:
                eligible += 1
                windows.append(Window(rows[start:start + horizon],
                                      rows[start - 1] if history_steps else None))
        first = rows[0]
        coverage.append(WindowCoverage(first.split, first.episode_id, horizon,
                                       eligible, unavailable, missing_history))
    return tuple(windows), tuple(coverage)
