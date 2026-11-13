from dataclasses import replace
from pathlib import Path

import pytest

from forward_dynamics.dataset import PredictionCase
from forward_dynamics.types import DynamicsContext, DynamicsState


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "experiments/02-state-transition-dataset/data/dataset-v1"


def write_training_config(path: Path, *, output_dir: str = "results", epochs=3,
                          seeds=(0, 1)) -> Path:
    """Write a short explicit roster; tests derive expected runs by hand."""
    dataset = path.parent / "dataset"
    dataset.mkdir(exist_ok=True)
    run_blocks = []
    for seed in seeds:
        run_blocks.append(f"""
[[runs]]
variant_id = "canonical32"
hidden_width = 32
history_steps = 0
loss = "one_step"
membership = "all"
seed = {str(seed).lower() if isinstance(seed, bool) else seed}
device = "cpu"
repeat = 0
""")
    path.write_text(f"""schema_version = 1
dataset_dir = "dataset"
dataset_version = "experiment-02-transition-v1"
output_dir = "{output_dir}"
artifact_budget_bytes = 524288000
resume = "reject"

[training]
epochs = {str(epochs).lower() if isinstance(epochs, bool) else epochs}
batch_size = 64
sampler_seed_offset = 100000
validation_interval_epochs = 1
selection_metric = "validation_normalized_one_step_mse"
tie_break = "earliest_epoch"

[optimizer]
name = "adam"
learning_rate = 0.001
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8
weight_decay = 0.0

[normalization]
statistics = "float64_population_std"
constant_threshold = 1e-12
target_transform = "next_minus_current"
{''.join(run_blocks)}""")
    return path


@pytest.fixture
def training_config_writer():
    return write_training_config


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
