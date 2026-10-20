"""Closed-loop MuJoCo execution for the one-link control experiment."""

from dataclasses import dataclass
import math
from typing import Literal

import mujoco

from feedback_control.config import ExperimentConfig, validate_experiment_config
from feedback_control.controllers import (
    ControllerOutput,
    pd_control,
    proportional_control,
)


ControllerName = Literal["p", "pd"]


@dataclass(frozen=True)
class TrajectorySample:
    step_index: int
    time_s: float
    position_rad: float
    velocity_rad_s: float
    target_position_rad: float
    target_velocity_rad_s: float
    raw_torque_nm: float
    applied_torque_nm: float
    saturated: bool


@dataclass(frozen=True)
class EpisodeResult:
    controller: ControllerName
    samples: tuple[TrajectorySample, ...]
    final_time_s: float
    final_position_rad: float
    final_velocity_rad_s: float


def run_episode(
    config: ExperimentConfig,
    controller: ControllerName,
) -> EpisodeResult:
    """Run one deterministic observe-control-step episode."""
    validate_experiment_config(config)
    if controller not in ("p", "pd"):
        raise ValueError(f"unknown controller: {controller!r}")

    model = mujoco.MjModel.from_xml_path(str(config.model.path))
    _validate_model_contract(model, config)
    data = mujoco.MjData(model)

    keyframe_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_KEY,
        config.model.keyframe,
    )
    if keyframe_id < 0:
        raise ValueError(f"unknown model keyframe: {config.model.keyframe!r}")

    mujoco.mj_resetDataKeyframe(model, data, keyframe_id)
    mujoco.mj_forward(model, data)
    if not math.isclose(
        float(data.qpos[0]),
        config.episode.start_position_rad,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError("model keyframe does not match configured start position")

    simulation_substeps = round(
        config.episode.control_interval_s / config.model.simulation_timestep_s
    )
    control_steps = round(
        config.episode.duration_s / config.episode.control_interval_s
    )

    samples: list[TrajectorySample] = []
    for step_index in range(control_steps):
        position = float(data.qpos[0])
        velocity = float(data.qvel[0])
        output = _compute_control(config, controller, position, velocity)

        # Record the observation and the action selected from that observation
        # before the action changes the simulation state.
        samples.append(
            TrajectorySample(
                step_index=step_index,
                time_s=float(data.time),
                position_rad=position,
                velocity_rad_s=velocity,
                target_position_rad=config.episode.target_position_rad,
                target_velocity_rad_s=config.episode.target_velocity_rad_s,
                raw_torque_nm=output.raw_torque,
                applied_torque_nm=output.applied_torque,
                saturated=output.saturated,
            )
        )

        data.ctrl[0] = output.applied_torque
        for _ in range(simulation_substeps):
            mujoco.mj_step(model, data)

    return EpisodeResult(
        controller=controller,
        samples=tuple(samples),
        final_time_s=float(data.time),
        final_position_rad=float(data.qpos[0]),
        final_velocity_rad_s=float(data.qvel[0]),
    )


def _compute_control(
    config: ExperimentConfig,
    controller: ControllerName,
    position: float,
    velocity: float,
) -> ControllerOutput:
    common_arguments = {
        "target_position": config.episode.target_position_rad,
        "position": position,
        "torque_limit": config.episode.torque_limit_nm,
    }
    if controller == "p":
        return proportional_control(kp=config.p.kp, **common_arguments)
    return pd_control(
        velocity=velocity,
        target_velocity=config.episode.target_velocity_rad_s,
        kp=config.pd.kp,
        kd=config.pd.kd,
        **common_arguments,
    )


def _validate_model_contract(
    model: mujoco.MjModel,
    config: ExperimentConfig,
) -> None:
    if (model.nq, model.nv, model.nu) != (1, 1, 1):
        raise ValueError("experiment requires exactly one position, velocity, and actuator")
    if not math.isclose(
        model.opt.timestep,
        config.model.simulation_timestep_s,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("MuJoCo timestep does not match the experiment configuration")

    control_min, control_max = model.actuator_ctrlrange[0]
    expected_limit = config.episode.torque_limit_nm
    if not (
        math.isclose(control_min, -expected_limit, rel_tol=0.0, abs_tol=1e-12)
        and math.isclose(control_max, expected_limit, rel_tol=0.0, abs_tol=1e-12)
    ):
        raise ValueError("MuJoCo actuator limits do not match the experiment configuration")

    joint_min, joint_max = model.jnt_range[0]
    for name, position in (
        ("start", config.episode.start_position_rad),
        ("target", config.episode.target_position_rad),
    ):
        if not joint_min <= position <= joint_max:
            raise ValueError(f"configured {name} position is outside the joint range")
