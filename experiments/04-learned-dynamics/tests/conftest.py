from dataclasses import replace
from pathlib import Path

import pytest

from forward_dynamics.dataset import PredictionCase
from forward_dynamics.types import DynamicsContext, DynamicsState


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "experiments/02-state-transition-dataset/data/dataset-v1"


@pytest.fixture
def case():
    return PredictionCase(
        split="train", episode_id="a", scenario_id="scenario-a",
        controller_id="p", seed=0, step_index=0,
        state=DynamicsState(1.0, 2.0), applied_torque_nm=3.0,
        expected_next_state=DynamicsState(1.02, 2.1),
        context=DynamicsContext("one-link", "a" * 64, 0.01, 3.0),
        terminated=False, truncated=True, termination_reason="time_limit",
    )


@pytest.fixture
def split_view(case):
    from learned_dynamics.types import SplitView

    def build(cases=None, split="train"):
        selected = cases if cases is not None else (replace(case, split=split),)
        return SplitView(split, tuple(selected), "experiment-02-transition-v1", "b" * 64, "c" * 64)
    return build
