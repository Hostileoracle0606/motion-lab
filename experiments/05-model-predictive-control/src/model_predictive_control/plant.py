"""Persistent MuJoCo truth plant for the checksummed one-link model."""
import hashlib
import math
from pathlib import Path

import mujoco

from forward_dynamics.mujoco_predictor import MODEL_SHA256, MUJOCO_VERSION
from forward_dynamics.types import DynamicsState, validate_finite_scalar, validate_state
from model_predictive_control.contracts import PHYSICS_INTERVAL_S, POSITION_LIMIT_RAD, TORQUE_LIMIT_NM, PlantStep


class MujocoPlant:
    def __init__(self, model_path: Path) -> None:
        content = model_path.resolve().read_bytes()
        if hashlib.sha256(content).hexdigest() != MODEL_SHA256:
            raise ValueError("unsupported model checksum")
        if mujoco.__version__ != MUJOCO_VERSION:
            raise ValueError("unsupported MuJoCo version")
        self._model = mujoco.MjModel.from_xml_string(content.decode("utf-8"))
        if (self._model.nq, self._model.nv, self._model.nu, self._model.na) != (1, 1, 1, 0):
            raise ValueError("unsupported model state dimensions")
        if self._model.opt.timestep != PHYSICS_INTERVAL_S:
            raise ValueError("unsupported physics interval")
        self._data = mujoco.MjData(self._model)

    def reset(self, state: DynamicsState) -> DynamicsState:
        validate_state(state)
        if abs(float(state.position_rad)) > POSITION_LIMIT_RAD:
            raise ValueError("initial state exceeds joint limit")
        mujoco.mj_resetData(self._model, self._data)
        self._data.qpos[0] = float(state.position_rad)
        self._data.qvel[0] = float(state.velocity_rad_s)
        mujoco.mj_forward(self._model, self._data)
        return DynamicsState(float(self._data.qpos[0]), float(self._data.qvel[0]))

    def advance(self, applied_torque_nm: float) -> PlantStep:
        validate_finite_scalar("applied_torque_nm", applied_torque_nm)
        if abs(float(applied_torque_nm)) > TORQUE_LIMIT_NM:
            raise ValueError("applied_torque_nm exceeds torque limit")
        self._data.ctrl[0] = float(applied_torque_nm)
        for count in range(1, 6):
            mujoco.mj_step(self._model, self._data)
            position, velocity = float(self._data.qpos[0]), float(self._data.qvel[0])
            if not math.isfinite(position) or not math.isfinite(velocity):
                return PlantStep(None, count, "invalid_state")
            state = DynamicsState(position, velocity)
            if abs(position) > POSITION_LIMIT_RAD:
                return PlantStep(state, count, "joint_limit")
        return PlantStep(state, 5)
