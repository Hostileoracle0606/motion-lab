"""Pure fixed-step classical RK4 for an unconstrained damped rotor.

Known coefficients, constant applied torque, no soft-limit/contact forces.
Integration accuracy for this ODE is distinct from agreement with MuJoCo.
"""

import math

from forward_dynamics.plant import PlantParameters
from forward_dynamics.types import (
    DynamicsContext, DynamicsState, validate_finite_scalar,
    validate_prediction_inputs, validate_state,
)


MAX_SUBSTEPS = 10000


class AnalyticalRK4Predictor:
    """Integrate qdot=v, vdot=(u-b*v)/J without simulator queries or state."""

    def __init__(self, parameters: PlantParameters, integration_timestep_s: float) -> None:
        if not isinstance(parameters, PlantParameters):
            raise ValueError("parameters must be PlantParameters")
        validate_finite_scalar("integration_timestep_s", integration_timestep_s)
        if integration_timestep_s <= 0:
            raise ValueError("integration_timestep_s must be positive")
        self.parameters = parameters
        self.integration_timestep_s = float(integration_timestep_s)
        self.predictor_id = f"analytical_rk4_{self.integration_timestep_s:g}"

    def predict(
        self, state: DynamicsState, applied_torque_nm: float, context: DynamicsContext,
    ) -> DynamicsState:
        validate_prediction_inputs(state, applied_torque_nm, context)
        if (context.environment_id != self.parameters.environment_id
                or context.environment_version != self.parameters.environment_version):
            raise ValueError("prediction context does not match plant parameters")
        ratio = float(context.control_interval_s) / self.integration_timestep_s
        if not math.isfinite(ratio) or ratio > MAX_SUBSTEPS:
            raise ValueError(f"control interval exceeds supported {MAX_SUBSTEPS} substeps")
        substeps = round(ratio)
        if substeps <= 0 or not math.isclose(ratio, substeps, rel_tol=0, abs_tol=1e-12):
            raise ValueError("control interval must be a positive integer number of substeps")
        q, v = float(state.position_rad), float(state.velocity_rad_s)
        torque = float(applied_torque_nm)
        inertia = self.parameters.inertia_kg_m2
        damping = self.parameters.damping_nm_s_per_rad
        h = self.integration_timestep_s
        for _ in range(substeps):
            k1q, k1v = v, (torque - damping * v) / inertia
            k2q = v + h * k1v / 2
            k2v = (torque - damping * k2q) / inertia
            k3q = v + h * k2v / 2
            k3v = (torque - damping * k3q) / inertia
            k4q = v + h * k3v
            k4v = (torque - damping * k4q) / inertia
            q += h * (k1q + 2 * k2q + 2 * k3q + k4q) / 6
            v += h * (k1v + 2 * k2v + 2 * k3v + k4v) / 6
            validate_state(DynamicsState(q, v))
        return DynamicsState(q, v)
