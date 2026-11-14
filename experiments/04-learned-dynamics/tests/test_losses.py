from dataclasses import replace

import pytest
import torch

from forward_dynamics.dataset import PredictionCase
from forward_dynamics.types import DynamicsState
from learned_dynamics.losses import OneStepBatch, one_step_loss, rollout_loss
from learned_dynamics.model import ResidualMLP
from learned_dynamics.predictor import SUPPORTED_CONTEXT
from learned_dynamics.types import FEATURE_ORDER, FEATURE_UNITS, ModelSpec, Normalizer, Window


def unit_normalizer():
    return Normalizer(1, FEATURE_ORDER, FEATURE_UNITS, (0.0, 0.0, 0.0),
                      (1.0, 1.0, 1.0), (False, False, False), (0.0, 0.0),
                      (1.0, 1.0), (False, False), 1,
                      "experiment-02-transition-v1", "b" * 64, "c" * 64,
                      (("episode", 0),))


def zero_model_with_delta(delta=1.0):
    model = ResidualMLP(ModelSpec(), seed=0)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.network[-1].bias[0] = delta
    return model


def test_one_step_loss_uses_normalized_physical_delta_components():
    model = zero_model_with_delta(1.0)
    batch = OneStepBatch(torch.tensor([[0.0, 0.0, 0.0]]),
                         torch.tensor([[2.0, 0.0]]))
    assert one_step_loss(model, batch, unit_normalizer()).item() == pytest.approx(0.5)


def test_rollout_loss_recurses_on_prediction_and_preserves_gradients():
    first = PredictionCase("train", "episode", "scenario", "synthetic", 0, 0,
                           DynamicsState(0.0, 0.0), 0.0, DynamicsState(10.0, 0.0),
                           SUPPORTED_CONTEXT, False, False, None)
    second = replace(first, step_index=1, state=DynamicsState(10.0, 0.0),
                     expected_next_state=DynamicsState(20.0, 0.0), truncated=True,
                     termination_reason="time_limit")
    model = zero_model_with_delta(1.0)
    loss = rollout_loss(model, (Window((first, second), None),), unit_normalizer(), 2)
    assert loss.item() == pytest.approx(101.25)
    loss.backward()
    assert model.network[-1].bias.grad is not None
    assert torch.isfinite(model.network[-1].bias.grad).all()


@pytest.mark.parametrize("horizon", [0, True, 3])
def test_rollout_loss_rejects_invalid_or_incomplete_horizon(horizon):
    case = PredictionCase("train", "episode", "scenario", "synthetic", 0, 0,
                          DynamicsState(0.0, 0.0), 0.0, DynamicsState(1.0, 0.0),
                          SUPPORTED_CONTEXT, False, True, "time_limit")
    with pytest.raises(ValueError):
        rollout_loss(zero_model_with_delta(), (Window((case,), None),), unit_normalizer(), horizon)
