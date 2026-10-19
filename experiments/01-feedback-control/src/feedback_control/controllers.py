"""Pure feedback-controller equations for the one-link experiment.

These functions do not know about MuJoCo. They transform measured state and a
target into a bounded torque command, which keeps the control law independently
testable from the simulator.
"""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ControllerOutput:
    """A controller's requested torque and the command the actuator receives."""

    raw_torque: float
    applied_torque: float
    saturated: bool


def proportional_control(
    *,
    target_position: float,
    position: float,
    kp: float,
    torque_limit: float,
) -> ControllerOutput:
    """Compute position-error feedback: u = Kp * (target - position)."""
    _validate_gain("kp", kp)
    _validate_finite("target_position", target_position)
    _validate_finite("position", position)

    position_error = target_position - position
    raw_torque = kp * position_error
    return _apply_symmetric_limit(raw_torque, torque_limit)


def pd_control(
    *,
    target_position: float,
    position: float,
    velocity: float,
    kp: float,
    kd: float,
    torque_limit: float,
    target_velocity: float = 0.0,
) -> ControllerOutput:
    """Compute position and velocity feedback.

    u = Kp * (target_position - position)
        + Kd * (target_velocity - velocity)
    """
    _validate_gain("kp", kp)
    _validate_gain("kd", kd)
    _validate_finite("target_position", target_position)
    _validate_finite("position", position)
    _validate_finite("target_velocity", target_velocity)
    _validate_finite("velocity", velocity)

    position_error = target_position - position
    velocity_error = target_velocity - velocity
    raw_torque = kp * position_error + kd * velocity_error
    return _apply_symmetric_limit(raw_torque, torque_limit)


def _apply_symmetric_limit(raw_torque: float, torque_limit: float) -> ControllerOutput:
    _validate_finite("raw_torque", raw_torque)
    _validate_finite("torque_limit", torque_limit)
    if torque_limit <= 0:
        raise ValueError("torque_limit must be greater than zero")

    applied_torque = max(-torque_limit, min(torque_limit, raw_torque))
    return ControllerOutput(
        raw_torque=raw_torque,
        applied_torque=applied_torque,
        saturated=applied_torque != raw_torque,
    )


def _validate_gain(name: str, gain: float) -> None:
    _validate_finite(name, gain)
    if gain < 0:
        raise ValueError(f"{name} must be non-negative")


def _validate_finite(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
