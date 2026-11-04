"""Prediction records and validation shared by dynamics adapters."""

from dataclasses import dataclass
import math
from numbers import Real
from typing import Protocol


@dataclass(frozen=True)
class DynamicsState:
    """Scalar limited-hinge position (rad) and velocity (rad/s)."""

    position_rad: float
    velocity_rad_s: float


@dataclass(frozen=True)
class DynamicsContext:
    """Environment identity, held-action duration, and applied-torque limit."""

    environment_id: str
    environment_version: str
    control_interval_s: float
    torque_limit_nm: float


class Predictor(Protocol):
    """Predict one next state without access to a target or expected outcome."""

    predictor_id: str

    def predict(
        self,
        state: DynamicsState,
        applied_torque_nm: float,
        context: DynamicsContext,
    ) -> DynamicsState:
        """Apply a previously limited torque for the context's interval."""
        ...


def validate_finite_scalar(name: str, value: float) -> None:
    """Reject non-real, boolean, or non-finite numeric scalar inputs."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real scalar")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError(f"{name} must be a finite real scalar")


def validate_state(state: DynamicsState) -> None:
    """Validate at the prediction/metric seam, without imposing joint limits."""
    if not isinstance(state, DynamicsState):
        raise ValueError("state must be a DynamicsState")
    validate_finite_scalar("position_rad", state.position_rad)
    validate_finite_scalar("velocity_rad_s", state.velocity_rad_s)


def validate_prediction_inputs(
    state: DynamicsState,
    applied_torque_nm: float,
    context: DynamicsContext,
) -> None:
    """Validate shared inputs; model-specific identity checks belong to adapters."""
    validate_state(state)
    if not isinstance(context, DynamicsContext):
        raise ValueError("context must be a DynamicsContext")
    for name in ("environment_id", "environment_version"):
        value = getattr(context, name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a nonempty string")
    for name in ("control_interval_s", "torque_limit_nm"):
        value = getattr(context, name)
        validate_finite_scalar(name, value)
        if value <= 0.0:
            raise ValueError(f"{name} must be greater than zero")
    validate_finite_scalar("applied_torque_nm", applied_torque_nm)
    # Real includes fixed-width NumPy scalars: normalize before abs/arithmetic
    # so the minimum signed integer cannot overflow and bypass this limit.
    if abs(float(applied_torque_nm)) > float(context.torque_limit_nm):
        raise ValueError("applied_torque_nm exceeds torque_limit_nm")
