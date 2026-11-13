"""Convert physical state/action to normalized residual inference and back."""

from copy import deepcopy

import numpy as np
import torch

from forward_dynamics.types import (
    DynamicsContext, DynamicsState, validate_prediction_inputs, validate_state,
)
from learned_dynamics.devices import require_device
from learned_dynamics.model import ResidualMLP
from learned_dynamics.normalization import inverse_delta, transform_inputs
from learned_dynamics.types import Normalizer


SUPPORTED_CONTEXT = DynamicsContext(
    "motion-lab/experiment-01/one-link",
    "8612091d73babb1a0122cb5163ea744183486cd8cc6fbae2813f45818be57263",
    0.010, 3.0,
)


def validate_context(context: DynamicsContext) -> None:
    validate_prediction_inputs(DynamicsState(0.0, 0.0), 0.0, context)
    if context != SUPPORTED_CONTEXT:
        raise ValueError("context does not match the supported one-link environment/timing/limits")


class LearnedPredictor:
    def __init__(self, model: ResidualMLP, normalizer: Normalizer,
                 context: DynamicsContext, predictor_id: str, device: str):
        validate_context(context)
        if not isinstance(model, ResidualMLP) or not isinstance(normalizer, Normalizer):
            raise ValueError("predictor requires ResidualMLP and Normalizer")
        if not isinstance(predictor_id, str) or not predictor_id.strip():
            raise ValueError("predictor_id must be nonempty")
        self.predictor_id, self.normalizer, self.context = predictor_id, normalizer, SUPPORTED_CONTEXT
        self.device = require_device(device)
        if any(p.dtype != torch.float32 or not torch.isfinite(p).all().item()
               for p in model.parameters()):
            raise ValueError("model parameters must be finite float32")
        try:
            self.model = deepcopy(model).to(self.device).eval()
        except (RuntimeError, NotImplementedError) as error:
            raise ValueError(f"predictor {predictor_id}: device setup failed: {error}") from error

    def _infer(self, values: np.ndarray, states: np.ndarray) -> np.ndarray:
        try:
            normalized = transform_inputs(values, self.normalizer)
            with np.errstate(over="ignore", invalid="ignore"):
                inputs = normalized.astype(np.float32)
            if not np.isfinite(inputs).all():
                raise ValueError("normalized values exceed float32 range")
            with torch.inference_mode():
                output = self.model(torch.from_numpy(inputs).to(self.device)).cpu().numpy()
            delta = inverse_delta(output, self.normalizer)
            with np.errstate(over="ignore", invalid="ignore"):
                result = states + delta
            if not np.isfinite(result).all():
                raise ValueError("predicted states must be finite")
            return result
        except (ValueError, RuntimeError, NotImplementedError, ArithmeticError) as error:
            raise ValueError(f"predictor {self.predictor_id}: {error}") from error

    def predict(self, state: DynamicsState, applied_torque_nm: float,
                context: DynamicsContext) -> DynamicsState:
        if self.model.spec.history_steps:
            raise ValueError("history model requires explicit previous state/action")
        validate_context(context)
        validate_prediction_inputs(state, applied_torque_nm, context)
        states = np.array([[state.position_rad, state.velocity_rad_s]], dtype=np.float64)
        result = self._infer(np.column_stack((states, [float(applied_torque_nm)])), states)
        return DynamicsState(float(result[0, 0]), float(result[0, 1]))

    def predict_batch(self, states: np.ndarray, actions: np.ndarray,
                      context: DynamicsContext) -> np.ndarray:
        if self.model.spec.history_steps:
            raise ValueError("history model requires explicit previous state/action")
        validate_context(context)
        if (not isinstance(states, np.ndarray) or states.ndim != 2 or states.shape[1] != 2 or
                states.shape[0] == 0 or states.dtype.kind not in "iuf" or
                not isinstance(actions, np.ndarray) or actions.shape != (states.shape[0],) or
                actions.dtype.kind not in "iuf"):
            raise ValueError("states/actions must be real numeric [N,2]/[N] arrays")
        if not np.isfinite(states).all() or not np.isfinite(actions).all() or np.any(np.abs(actions.astype(np.float64)) > context.torque_limit_nm):
            raise ValueError("batch values must be finite and actions within the torque limit")
        physical = states.astype(np.float64)
        return self._infer(np.column_stack((physical, actions)), physical)

    def predict_with_history(self, state: DynamicsState, applied_torque_nm: float,
                             previous_state: DynamicsState, previous_applied_torque_nm: float,
                             context: DynamicsContext) -> DynamicsState:
        if self.model.spec.history_steps != 1:
            raise ValueError("predict_with_history requires an explicit history model")
        validate_context(context)
        validate_prediction_inputs(state, applied_torque_nm, context)
        validate_prediction_inputs(previous_state, previous_applied_torque_nm, context)
        values = np.array([[state.position_rad, state.velocity_rad_s, applied_torque_nm,
                            previous_state.position_rad, previous_state.velocity_rad_s,
                            previous_applied_torque_nm]], dtype=np.float64)
        result = self._infer(values, values[:, :2])
        answer = DynamicsState(float(result[0, 0]), float(result[0, 1]))
        validate_state(answer)
        return answer
