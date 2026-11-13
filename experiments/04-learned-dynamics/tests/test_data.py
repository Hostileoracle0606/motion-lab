from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pytest


DATA = Path(__file__).resolve().parents[3] / "experiments/02-state-transition-dataset/data/dataset-v1"


def test_views_preserve_scenario_membership_and_failures():
    from learned_dynamics.data import load_views

    views = load_views(DATA, "experiment-02-transition-v1")
    assert tuple(len(v.cases) for v in (views.train, views.validation, views.test)) == (940, 800, 800)
    assert sum(c.terminated for c in views.train.cases) == 2
    assert {c.scenario_id for c in views.validation.cases} == {"negative-to-positive-narrow"}
    assert {c.scenario_id for c in views.test.cases} == {"positive-to-negative-narrow"}
    assert views.train.view.manifest_sha256 == hashlib.sha256((DATA / "manifest.json").read_bytes()).hexdigest()


def test_features_exclude_targets_and_raw_torque(split_view):
    from learned_dynamics.data import training_arrays
    from learned_dynamics.types import TrainingView

    x, y = training_arrays(TrainingView(split_view()))
    np.testing.assert_array_equal(x, [[1, 2, 3]])
    np.testing.assert_allclose(y, [[0.02, 0.1]], rtol=0, atol=1e-14)
    assert x.dtype == y.dtype == np.float64


@pytest.mark.parametrize("split", ["validation", "test"])
def test_training_view_rejects_validation_and_test(split_view, split):
    from learned_dynamics.types import TrainingView
    with pytest.raises(ValueError, match="train"):
        TrainingView(split_view(split=split))


def test_arrays_require_training_wrapper(split_view):
    from learned_dynamics.data import training_arrays
    with pytest.raises(ValueError, match="TrainingView"):
        training_arrays(split_view())


@pytest.mark.parametrize("corruption", ["checksum", "membership"])
def test_load_views_rejects_corrupt_dataset(tmp_path, corruption):
    from learned_dynamics.data import load_views
    copied = tmp_path / "data"
    shutil.copytree(DATA, copied)
    if corruption == "checksum":
        (copied / "train.jsonl").write_text("{}\n")
    else:
        path = copied / "manifest.json"
        payload = json.loads(path.read_text())
        # Partition declarations are a genuine consumer input, not a loader mock.
        payload["partitions"][0]["split"] = "test"
        path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        load_views(copied, "experiment-02-transition-v1")


def test_wrong_dataset_version_is_rejected():
    from learned_dynamics.data import load_views
    with pytest.raises(ValueError, match="version"):
        load_views(DATA, "wrong")
