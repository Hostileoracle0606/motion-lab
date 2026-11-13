"""Differentiable normalized one-step and recursive physical-state losses."""

from dataclasses import dataclass

import torch

from learned_dynamics.model import ResidualMLP
from learned_dynamics.types import Normalizer, Window


@dataclass(frozen=True)
class OneStepBatch:
    inputs: torch.Tensor
    target_deltas: torch.Tensor

    def __post_init__(self) -> None:
        if (not isinstance(self.inputs, torch.Tensor) or self.inputs.ndim != 2 or
                self.inputs.shape[0] == 0 or self.inputs.shape[1] not in (3, 6) or
                self.inputs.dtype != torch.float32 or
                not isinstance(self.target_deltas, torch.Tensor) or
                self.target_deltas.shape != (self.inputs.shape[0], 2) or
                self.target_deltas.dtype != torch.float32):
            raise ValueError("one-step batch must be float32 [N,3|6] inputs and [N,2] deltas")
        if not torch.isfinite(self.inputs).all().item() or not torch.isfinite(self.target_deltas).all().item():
            raise ValueError("one-step batch values must be finite")


def _constants(normalizer: Normalizer, device: torch.device, input_width: int):
    if not isinstance(normalizer, Normalizer):
        raise ValueError("loss requires a Normalizer")
    repeats = input_width // 3
    input_mean = torch.tensor(normalizer.input_mean * repeats, dtype=torch.float32, device=device)
    input_scale = torch.tensor(normalizer.input_scale * repeats, dtype=torch.float32, device=device)
    delta_mean = torch.tensor(normalizer.delta_mean, dtype=torch.float32, device=device)
    delta_scale = torch.tensor(normalizer.delta_scale, dtype=torch.float32, device=device)
    return input_mean, input_scale, delta_mean, delta_scale


def one_step_loss(model: ResidualMLP, batch: OneStepBatch,
                  normalizer: Normalizer) -> torch.Tensor:
    """Average squared error over examples and two normalized delta coordinates."""
    if not isinstance(model, ResidualMLP) or not isinstance(batch, OneStepBatch):
        raise ValueError("one-step loss requires ResidualMLP and OneStepBatch")
    if batch.inputs.shape[1] != 3 * (1 + model.spec.history_steps):
        raise ValueError("batch width does not match model history")
    device = next(model.parameters()).device
    inputs = batch.inputs.to(device)
    targets = batch.target_deltas.to(device)
    x_mean, x_scale, y_mean, y_scale = _constants(normalizer, device, inputs.shape[1])
    predicted = model((inputs - x_mean) / x_scale)
    target = (targets - y_mean) / y_scale
    loss = torch.mean((predicted - target) ** 2)
    if not torch.isfinite(loss).item():
        raise ValueError("one-step loss is non-finite")
    return loss


def rollout_loss(model: ResidualMLP, windows: tuple[Window, ...],
                 normalizer: Normalizer, horizon: int = 5) -> torch.Tensor:
    """Recurse on predictions; future measured states are labels, never inputs."""
    if (not isinstance(model, ResidualMLP) or not isinstance(windows, tuple) or not windows or
            type(horizon) is not int or horizon <= 0 or
            any(not isinstance(window, Window) or len(window.cases) != horizon for window in windows)):
        raise ValueError("rollout loss requires complete immutable horizon windows")
    if model.spec.history_steps and any(window.previous is None for window in windows):
        raise ValueError("history rollout requires an explicit previous case")
    device = next(model.parameters()).device
    _, _, delta_mean, delta_scale = _constants(
        normalizer, device, 3 * (1 + model.spec.history_steps))
    current = torch.tensor([[window.cases[0].state.position_rad,
                             window.cases[0].state.velocity_rad_s]
                            for window in windows], dtype=torch.float32, device=device)
    previous = None
    previous_action = None
    if model.spec.history_steps:
        previous = torch.tensor([[window.previous.state.position_rad,
                                  window.previous.state.velocity_rad_s]
                                 for window in windows], dtype=torch.float32, device=device)
        previous_action = torch.tensor([[window.previous.applied_torque_nm]
                                        for window in windows], dtype=torch.float32, device=device)
    squared = []
    for step in range(horizon):
        action = torch.tensor([[window.cases[step].applied_torque_nm] for window in windows],
                              dtype=torch.float32, device=device)
        physical_inputs = torch.cat((current, action), dim=1)
        if model.spec.history_steps:
            physical_inputs = torch.cat((physical_inputs, previous, previous_action), dim=1)
        input_mean, input_scale, _, _ = _constants(normalizer, device, physical_inputs.shape[1])
        normalized_delta = model((physical_inputs - input_mean) / input_scale)
        predicted_next = current + normalized_delta * delta_scale + delta_mean
        target = torch.tensor([[window.cases[step].expected_next_state.position_rad,
                                window.cases[step].expected_next_state.velocity_rad_s]
                               for window in windows], dtype=torch.float32, device=device)
        squared.append(((predicted_next - target) / delta_scale) ** 2)
        if model.spec.history_steps:
            previous, previous_action = current, action
        current = predicted_next
    loss = torch.stack(squared).mean()
    if not torch.isfinite(loss).item():
        raise ValueError("rollout loss is non-finite")
    return loss
