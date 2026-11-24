"""Constant-torque hinge prediction with ideal dry friction and viscous damping."""

import numpy as np

from forward_dynamics.plant import PlantParameters
from forward_dynamics.types import (
    DynamicsContext, DynamicsState, validate_finite_scalar, validate_prediction_inputs,
)


class AnalyticalDryFrictionPredictor:
    """Integrate slip exactly and resolve the single possible stop or reversal."""

    def __init__(
        self, parameters: PlantParameters, context: DynamicsContext, friction_torque_nm: float,
    ) -> None:
        if not isinstance(parameters, PlantParameters):
            raise ValueError("parameters must be PlantParameters")
        validate_prediction_inputs(DynamicsState(0.0, 0.0), 0.0, context)
        if (context.environment_id, context.environment_version) != (
            parameters.environment_id, parameters.environment_version,
        ):
            raise ValueError("prediction context does not match plant parameters")
        validate_finite_scalar("friction_torque_nm", friction_torque_nm)
        if friction_torque_nm < 0:
            raise ValueError("friction torque must be nonnegative")
        self.parameters = parameters
        self.context = context
        self.friction_torque_nm = float(friction_torque_nm)
        self.predictor_id = f"analytical_dry_friction_{self.friction_torque_nm:g}Nm"

    def _slip(self, v, acceleration, duration):
        decay = self.parameters.damping_nm_s_per_rad / self.parameters.inertia_kg_m2
        if decay == 0.0:
            return v * duration + 0.5 * acceleration * duration ** 2, v + acceleration * duration
        x = decay * duration
        exp_delta = np.expm1(-x)
        phi1 = np.divide(-exp_delta, x, out=np.ones_like(x), where=x != 0)
        # Evaluate (x + expm1(-x))/x² without cancellation near zero.
        phi2 = np.array(0.5 - x / 6)
        np.divide(x + exp_delta, x ** 2, out=phi2, where=x > 1e-4)
        delta = v * duration + acceleration * duration ** 2 * phi2
        return delta, v + acceleration * duration * phi1

    def predict(
        self, state: DynamicsState, applied_torque_nm: float, context: DynamicsContext,
    ) -> DynamicsState:
        validate_prediction_inputs(state, applied_torque_nm, context)
        result = self.predict_batch(
            np.array([[state.position_rad, state.velocity_rad_s]], dtype=float),
            np.array([applied_torque_nm], dtype=float), context,
        )
        return DynamicsState(*map(float, result[0]))

    def predict_batch(self, states, actions, context: DynamicsContext) -> np.ndarray:
        if context != self.context:
            raise ValueError("prediction context does not match plant parameters")
        if (not isinstance(states, np.ndarray) or states.ndim != 2 or states.shape[1] != 2
                or not len(states) or states.dtype.kind not in "iuf"
                or not isinstance(actions, np.ndarray) or actions.shape != (len(states),)
                or actions.dtype.kind not in "iuf"):
            raise ValueError("states/actions must be real numeric [N,2]/[N] arrays")
        q, v = states.astype(float, copy=False).T
        torque = actions.astype(float, copy=False)
        if (not np.isfinite(states).all() or not np.isfinite(torque).all()
                or np.any(np.abs(torque) > context.torque_limit_nm)):
            raise ValueError("batch values must be finite and actions within the torque limit")
        inertia = self.parameters.inertia_kg_m2
        damping = self.parameters.damping_nm_s_per_rad
        friction = self.friction_torque_nm
        duration = float(context.control_interval_s)
        breakaway = torque - np.clip(torque, -friction, friction)
        slip_torque = np.where(v == 0, breakaway, torque - friction * np.sign(v))
        acceleration = (slip_torque - damping * v) / inertia
        displacement, end_velocity = self._slip(v, acceleration, duration)
        crossing = ((v > 0) & (end_velocity <= 0)) | ((v < 0) & (end_velocity >= 0))
        if crossing.any():
            initial_velocity = v[crossing]
            before_stop = acceleration[crossing]
            stop_time = -initial_velocity / before_stop
            if damping:
                decay = damping / inertia
                stop_time = -np.log1p(-decay * stop_time) / decay
            stop_time = np.minimum(stop_time, duration)
            to_stop, _ = self._slip(initial_velocity, before_stop, stop_time)
            from_stop, end_velocity[crossing] = self._slip(
                np.zeros_like(stop_time), breakaway[crossing] / inertia, duration - stop_time,
            )
            displacement[crossing] = to_stop + from_stop
        result = np.column_stack((q + displacement, end_velocity))
        if not np.isfinite(result).all():
            raise ValueError("predicted states must be finite")
        return result
