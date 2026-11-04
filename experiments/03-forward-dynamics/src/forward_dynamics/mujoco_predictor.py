"""Stateless reference queries for the checksummed Experiment 01 plant."""

import hashlib
import math
from pathlib import Path

import mujoco

from forward_dynamics.types import (
    DynamicsContext, DynamicsState, validate_prediction_inputs, validate_state,
)


ENVIRONMENT_ID = "motion-lab/experiment-01/one-link"
MODEL_SHA256 = "8612091d73babb1a0122cb5163ea744183486cd8cc6fbae2813f45818be57263"
MUJOCO_VERSION = "3.16.0"
MAX_SUBSTEPS = 10000


class MujocoPredictor:
    """Use fresh simulator data for every observed-state prediction.

    Only the known self-contained XML is supported: changing its dynamics or
    adding hidden state requires a new model/state contract, not just nq=nv=1.
    """

    predictor_id = "mujoco"

    def __init__(self, model_path: Path) -> None:
        content = model_path.resolve().read_bytes()
        if hashlib.sha256(content).hexdigest() != MODEL_SHA256:
            raise ValueError("unsupported model checksum")
        if mujoco.__version__ != MUJOCO_VERSION:
            raise ValueError("unsupported MuJoCo version")
        self._model = mujoco.MjModel.from_xml_string(content.decode("utf-8"))
        if (self._model.nq, self._model.nv, self._model.nu, self._model.na) != (1, 1, 1, 0):
            raise ValueError("unsupported model state dimensions")

    def predict(
        self, state: DynamicsState, applied_torque_nm: float, context: DynamicsContext,
    ) -> DynamicsState:
        validate_prediction_inputs(state, applied_torque_nm, context)
        if context.environment_id != ENVIRONMENT_ID or context.environment_version != MODEL_SHA256:
            raise ValueError("prediction context does not match the supported model")
        if context.torque_limit_nm != 3.0:
            raise ValueError("context torque limit does not match the model")
        ratio = float(context.control_interval_s) / self._model.opt.timestep
        if not math.isfinite(ratio) or ratio > MAX_SUBSTEPS:
            raise ValueError(f"control interval exceeds supported {MAX_SUBSTEPS} substeps")
        substeps = round(ratio)
        if substeps <= 0 or not math.isclose(ratio, substeps, abs_tol=1e-12, rel_tol=0):
            raise ValueError("control interval must be an integer number of substeps")
        data = mujoco.MjData(self._model)
        data.qpos[0] = float(state.position_rad)
        data.qvel[0] = float(state.velocity_rad_s)
        mujoco.mj_forward(self._model, data)
        data.ctrl[0] = float(applied_torque_nm)
        for _ in range(substeps):
            mujoco.mj_step(self._model, data)
        predicted = DynamicsState(float(data.qpos[0]), float(data.qvel[0]))
        validate_state(predicted)
        return predicted
