"""Small float32 residual networks, initialized on CPU with a declared seed."""

import torch
from torch import nn

from learned_dynamics.types import ModelSpec


class ResidualMLP(nn.Module):
    def __init__(self, spec: ModelSpec, seed: int = 0):
        super().__init__()
        if not isinstance(spec, ModelSpec):
            raise ValueError("model requires a ModelSpec")
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError("seed must be a nonnegative 63-bit integer")
        self.spec = spec
        width = spec.hidden_width
        # Do not contaminate the caller's random state or initialize on a
        # global default GPU/device/dtype. Training owns data-order randomness.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.network = nn.Sequential(
                nn.Linear(3 * (1 + spec.history_steps), width, device="cpu", dtype=torch.float32),
                nn.Tanh(),
                nn.Linear(width, width, device="cpu", dtype=torch.float32),
                nn.Tanh(),
                nn.Linear(width, 2, device="cpu", dtype=torch.float32),
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if (not isinstance(inputs, torch.Tensor) or inputs.ndim != 2 or
                inputs.shape[0] == 0 or inputs.shape[1] != 3 * (1 + self.spec.history_steps) or
                inputs.dtype != torch.float32):
            raise ValueError("model inputs must be nonempty float32 [N,input_width] tensors")
        if not torch.isfinite(inputs).all().item():
            raise ValueError("model inputs must be finite")
        return self.network(inputs)
