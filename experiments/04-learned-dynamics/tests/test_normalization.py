from dataclasses import FrozenInstanceError, replace
import json

import numpy as np
import pytest

from forward_dynamics.types import DynamicsState
from learned_dynamics.types import DataViews, TrainingView


def training_fixture(case, split_view):
    first = replace(case, state=DynamicsState(0, 2), expected_next_state=DynamicsState(1, 2.5))
    second = replace(case, episode_id="b", state=DynamicsState(2, 2),
                     expected_next_state=DynamicsState(3, 2.5))
    return TrainingView(split_view((first, second)))


def test_train_only_statistics_and_round_trip(case, split_view):
    from learned_dynamics.normalization import fit_normalizer, transform_inputs, inverse_delta
    normalizer = fit_normalizer(training_fixture(case, split_view))
    assert normalizer.input_mean == (1, 2, 3)
    assert normalizer.input_scale == (1, 1, 1)
    assert normalizer.input_constant == (False, True, True)
    assert normalizer.delta_mean == (1, 0.5)
    assert normalizer.delta_constant == (True, True)
    assert normalizer.fit_keys == (("a", 0), ("b", 0))
    np.testing.assert_array_equal(transform_inputs(np.array([[0, 2, 3], [2, 2, 3]]), normalizer),
                                  [[-1, 0, 0], [1, 0, 0]])
    np.testing.assert_array_equal(inverse_delta(np.array([[0, 0], [1, -1]]), normalizer),
                                  [[1, .5], [2, -.5]])
    with pytest.raises(FrozenInstanceError):
        normalizer.fit_row_count = 999


def test_heldout_mutation_cannot_change_fitted_statistics(case, split_view):
    from learned_dynamics.normalization import fit_normalizer
    train = training_fixture(case, split_view)
    validation = split_view((replace(case, split="validation", episode_id="v", scenario_id="v"),), "validation")
    test = split_view((replace(case, split="test", episode_id="t", scenario_id="t"),), "test")
    before = DataViews(train, validation, test)
    after = DataViews(train, replace(validation, cases=(
        replace(validation.cases[0], state=DynamicsState(-100, 500)),)),
        replace(test, cases=(replace(test.cases[0], state=DynamicsState(800, -900)),)))
    assert fit_normalizer(before.train) == fit_normalizer(after.train)


@pytest.mark.parametrize("split", ["train", "validation", "test"])
def test_fitting_rejects_bare_views(split_view, split):
    from learned_dynamics.normalization import fit_normalizer
    with pytest.raises(ValueError, match="TrainingView"):
        fit_normalizer(split_view(split=split))


def test_serialization_is_deterministic_and_refuses_overwrite(tmp_path, case, split_view):
    from learned_dynamics.normalization import fit_normalizer, write_normalizer, load_normalizer
    norm = fit_normalizer(training_fixture(case, split_view))
    first, second = tmp_path / "first.json", tmp_path / "second.json"
    write_normalizer(norm, first)
    write_normalizer(norm, second)
    assert first.read_bytes() == second.read_bytes()
    assert load_normalizer(first) == norm
    with pytest.raises(ValueError, match="exist"):
        write_normalizer(norm, first)


@pytest.mark.parametrize("field,value", [
    ("schema_version", True), ("schema_version", 2), ("fit_row_count", True),
    ("fit_row_count", 3), ("input_scale", [0, 1, 1]),
    ("delta_scale", [-1, 1]), ("input_mean", [1, 2]),
    ("feature_order", ["velocity_rad_s", "position_rad", "applied_torque_nm"]),
    ("feature_units", ["m", "rad/s", "N m"]),
    ("input_constant", [False, 1, True]), ("train_sha256", "wrong"),
    ("input_mean", [float("inf"), 2, 3]), ("delta_mean", [True, .5]),
    ("fit_keys", [["a", True], ["b", 0]]),
    ("fit_keys", [["a", 0], ["a", 0]]), ("unexpected", 1),
])
def test_loader_rejects_invalid_normalizer(tmp_path, case, split_view, field, value):
    from learned_dynamics.normalization import fit_normalizer, write_normalizer, load_normalizer
    file = tmp_path / "norm.json"
    write_normalizer(fit_normalizer(training_fixture(case, split_view)), file)
    data = json.loads(file.read_text())
    data[field] = value
    file.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_normalizer(file)


def test_loader_rejects_duplicate_fields(tmp_path):
    from learned_dynamics.normalization import load_normalizer
    file = tmp_path / "norm.json"
    file.write_text('{"schema_version":1,"schema_version":1}')
    with pytest.raises(ValueError):
        load_normalizer(file)


@pytest.mark.parametrize("values", [
    np.array([1, 2, 3]), np.ones((1, 2)), np.array([[True, False, True]]),
    np.array([[float("nan"), 2, 3]]), np.array([["1", "2", "3"]]),
])
def test_transform_rejects_invalid_arrays(case, split_view, values):
    from learned_dynamics.normalization import fit_normalizer, transform_inputs
    with pytest.raises(ValueError):
        transform_inputs(values, fit_normalizer(training_fixture(case, split_view)))


def test_history_scaling_reuses_training_features_without_clipping(case, split_view):
    from learned_dynamics.normalization import fit_normalizer, transform_inputs
    norm = fit_normalizer(training_fixture(case, split_view))
    np.testing.assert_array_equal(transform_inputs(np.array([[5, 2, 3, -3, 2, 3]]), norm),
                                  [[4, 0, 0, -4, 0, 0]])


def test_inverse_delta_rejects_bad_shape(case, split_view):
    from learned_dynamics.normalization import fit_normalizer, inverse_delta
    with pytest.raises(ValueError):
        inverse_delta(np.ones((1, 3)), fit_normalizer(training_fixture(case, split_view)))


def test_normalizer_rejects_scale_below_constant_threshold(case, split_view):
    from learned_dynamics.normalization import fit_normalizer
    norm = fit_normalizer(training_fixture(case, split_view))
    with pytest.raises(ValueError, match="threshold"):
        replace(norm, input_scale=(1e-15, 1, 1))
