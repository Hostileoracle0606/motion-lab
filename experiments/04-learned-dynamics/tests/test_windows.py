from dataclasses import replace
from pathlib import Path

import pytest

from forward_dynamics.types import DynamicsState


DATA = Path(__file__).resolve().parents[3] / "experiments/02-state-transition-dataset/data/dataset-v1"


def episode(case, count=3):
    return tuple(replace(case, step_index=i, state=DynamicsState(i, 2),
                         expected_next_state=DynamicsState(i + 1, 2),
                         truncated=i == count - 1,
                         termination_reason="time_limit" if i == count - 1 else None)
                 for i in range(count))


def test_terminal_ending_window_allowed_but_crossing_rejected(case, split_view):
    from learned_dynamics.windows import make_windows
    cases = episode(case)
    windows, counts = make_windows(split_view(cases), 2)
    assert [w.cases[0].step_index for w in windows] == [0, 1]
    assert windows[-1].cases[-1].truncated is True
    assert (counts[0].eligible, counts[0].unavailable, counts[0].missing_history) == (2, 1, 0)
    with pytest.raises(ValueError, match="terminal"):
        make_windows(split_view(cases + (replace(cases[-1], step_index=3,
                                                 state=DynamicsState(3, 2)),)), 2)


def test_history_removes_only_starts_without_previous_row(case, split_view):
    from learned_dynamics.windows import make_windows
    windows, counts = make_windows(split_view(episode(case)), 2, 1)
    assert len(windows) == 1
    assert windows[0].previous.step_index == 0
    assert windows[0].cases[0].step_index == 1
    assert (counts[0].eligible, counts[0].unavailable, counts[0].missing_history) == (1, 1, 1)


def test_missing_future_takes_priority_over_history(case, split_view):
    from learned_dynamics.windows import make_windows
    windows, counts = make_windows(split_view(episode(case)), 100, 1)
    assert windows == ()
    assert (counts[0].eligible, counts[0].unavailable, counts[0].missing_history) == (0, 3, 0)


def test_windows_never_join_different_episodes(case, split_view):
    from learned_dynamics.windows import make_windows
    cases = episode(case, 2)
    other = tuple(replace(c, episode_id="b") for c in cases)
    windows, counts = make_windows(split_view(cases + other), 3)
    assert not windows
    assert len(counts) == 2
    assert all(c.eligible == 0 and c.unavailable == 2 for c in counts)


@pytest.mark.parametrize("change", [
    {"step_index": 4}, {"state": DynamicsState(999, 2)}, {"scenario_id": "different"},
    {"context": None},
])
def test_broken_episode_is_rejected(case, split_view, change):
    from learned_dynamics.windows import make_windows
    cases = episode(case)
    with pytest.raises(ValueError):
        make_windows(split_view((cases[0], replace(cases[1], **change), cases[2])), 1)


@pytest.mark.parametrize("horizon,history", [(0, 0), (True, 0), (1.5, 0), (1, True), (1, 2)])
def test_window_arguments_are_strict(case, split_view, horizon, history):
    from learned_dynamics.windows import make_windows
    with pytest.raises(ValueError):
        make_windows(split_view(), horizon, history)


def test_committed_memberships_match_protocol():
    from learned_dynamics.data import load_views
    from learned_dynamics.windows import make_windows
    views = load_views(DATA, "experiment-02-transition-v1")
    for view, expected in [(views.train, (940, 936, 924)),
                           (views.validation, (800, 798, 792)),
                           (views.test, (800, 798, 792))]:
        assert tuple(len(make_windows(view, h, history)[0]) for h, history in
                     [(1, 0), (1, 1), (5, 0)]) == expected
        for h, history in [(1, 0), (5, 1), (100, 1)]:
            _, counts = make_windows(view, h, history)
            assert sum(c.eligible + c.unavailable + c.missing_history for c in counts) == len(view.cases)
