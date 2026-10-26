"""Deterministic replay of recorded applied-action sequences in MuJoCo."""

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path

import mujoco

from feedback_control.trajectory import TrajectoryRecord


@dataclass(frozen=True)
class ReplayResult:
    max_position_error_rad: float
    max_velocity_error_rad_s: float


def verify_action_replay(
    trajectory: TrajectoryRecord,
    model_path: Path,
    absolute_tolerance: float,
) -> ReplayResult:
    """Replay saved applied torques and reject state drift beyond tolerance."""
    if not math.isfinite(absolute_tolerance) or absolute_tolerance <= 0:
        raise ValueError(
            "absolute_tolerance must be finite and greater than zero"
        )
    resolved_model = model_path.resolve()
    model_sha256 = hashlib.sha256(resolved_model.read_bytes()).hexdigest()
    if model_sha256 != trajectory.environment_version:
        raise ValueError("model checksum mismatch")
    if not trajectory.transitions:
        raise ValueError("cannot replay an empty trajectory")

    model = mujoco.MjModel.from_xml_path(str(resolved_model))
    if (model.nq, model.nv, model.nu) != (1, 1, 1):
        raise ValueError(
            "replay requires exactly one position, velocity, and actuator"
        )
    if not math.isclose(
        model.opt.timestep,
        trajectory.configuration.simulation_timestep_s,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("model timestep does not match trajectory")
    substep_ratio = (
        trajectory.configuration.control_interval_s / model.opt.timestep
    )
    substeps = round(substep_ratio)
    if substeps <= 0 or not math.isclose(
        substep_ratio,
        substeps,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("control interval must be an integer number of substeps")

    keyframe_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_KEY,
        trajectory.configuration.model_keyframe,
    )
    if keyframe_id < 0:
        raise ValueError("trajectory model keyframe does not exist")
    first_state = trajectory.transitions[0].state
    if not _state_is_finite(first_state.position_rad, first_state.velocity_rad_s):
        raise ValueError("initial replay state must be finite")

    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, keyframe_id)
    data.qpos[0] = first_state.position_rad
    data.qvel[0] = first_state.velocity_rad_s
    mujoco.mj_forward(model, data)

    maximum_position_error = 0.0
    maximum_velocity_error = 0.0
    for transition in trajectory.transitions:
        if not math.isfinite(transition.applied_torque_nm):
            raise ValueError("applied replay action must be finite")
        expected = transition.next_state
        if not _state_is_finite(expected.position_rad, expected.velocity_rad_s):
            raise ValueError("saved replay next state must be finite")

        data.ctrl[0] = transition.applied_torque_nm
        for _ in range(substeps):
            mujoco.mj_step(model, data)
        actual_position = float(data.qpos[0])
        actual_velocity = float(data.qvel[0])
        if not _state_is_finite(actual_position, actual_velocity):
            raise ValueError("replayed state must be finite")
        position_error = abs(actual_position - expected.position_rad)
        velocity_error = abs(actual_velocity - expected.velocity_rad_s)
        maximum_position_error = max(maximum_position_error, position_error)
        maximum_velocity_error = max(maximum_velocity_error, velocity_error)
        if (
            position_error > absolute_tolerance
            or velocity_error > absolute_tolerance
        ):
            raise ValueError(
                "replay state error exceeds tolerance at "
                f"transition {transition.step_index}"
            )

    return ReplayResult(
        max_position_error_rad=maximum_position_error,
        max_velocity_error_rad_s=maximum_velocity_error,
    )


def _state_is_finite(
    position_rad: float | None,
    velocity_rad_s: float | None,
) -> bool:
    return (
        position_rad is not None
        and velocity_rad_s is not None
        and math.isfinite(position_rad)
        and math.isfinite(velocity_rad_s)
    )
