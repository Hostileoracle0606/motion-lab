from dataclasses import FrozenInstanceError, replace

import pytest


def test_views_are_immutable(split_view):
    view = split_view()
    with pytest.raises(FrozenInstanceError):
        view.split = "test"


def test_view_rejects_mixed_split_membership(split_view, case):
    with pytest.raises(ValueError, match="split"):
        split_view((case, replace(case, split="test", episode_id="b")))


@pytest.mark.parametrize("change", [
    {"step_index": True}, {"seed": False}, {"applied_torque_nm": float("nan")},
    {"terminated": 1}, {"truncated": 1}, {"termination_reason": None},
])
def test_view_rejects_invalid_case_fields(split_view, case, change):
    with pytest.raises(ValueError):
        split_view((replace(case, **change),))


def test_view_rejects_bad_provenance(split_view):
    with pytest.raises(ValueError, match="sha256"):
        replace(split_view(), train_sha256="invalid")


def test_training_wrapper_preserves_case_identity(split_view):
    from learned_dynamics.types import TrainingView
    view = split_view()
    assert TrainingView(view).cases == view.cases
