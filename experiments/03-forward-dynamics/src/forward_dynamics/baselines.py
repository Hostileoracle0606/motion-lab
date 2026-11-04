"""Simple deterministic adapters at the forward-dynamics prediction seam."""

from forward_dynamics.types import (
    DynamicsContext,
    DynamicsState,
    validate_prediction_inputs,
    validate_state,
)


class NoChangePredictor:
    """Assume both position and velocity remain unchanged over the interval."""

    predictor_id = "no_change"

    def predict(
        self,
        state: DynamicsState,
        applied_torque_nm: float,
        context: DynamicsContext,
    ) -> DynamicsState:
        """Validate the applied action and context, then return the current state."""
        validate_prediction_inputs(state, applied_torque_nm, context)
        return DynamicsState(float(state.position_rad), float(state.velocity_rad_s))


class ConstantVelocityPredictor:
    """Advance position at the current velocity; ignore torque after validation."""

    predictor_id = "constant_velocity"

    def predict(
        self,
        state: DynamicsState,
        applied_torque_nm: float,
        context: DynamicsContext,
    ) -> DynamicsState:
        """Predict q + v*dt without wrapping or clamping the limited hinge."""
        validate_prediction_inputs(state, applied_torque_nm, context)
        predicted = DynamicsState(
            position_rad=(
                float(state.position_rad)
                + float(state.velocity_rad_s) * float(context.control_interval_s)
            ),
            velocity_rad_s=float(state.velocity_rad_s),
        )
        validate_state(predicted)
        return predicted
