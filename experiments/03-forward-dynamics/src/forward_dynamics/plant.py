"""Static parameters for a zero-gravity, directly actuated damped hinge.

The record and its validation are simulator independent. MuJoCo is imported
only by extraction, which checks model bytes before compiling them.
"""

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import xml.etree.ElementTree as ET

from forward_dynamics.types import validate_finite_scalar


ENVIRONMENT_ID = "motion-lab/experiment-01/one-link"


@dataclass(frozen=True)
class PlantParameters:
    """Known physical coefficients and source SHA-256, never fitted values."""

    environment_id: str
    environment_version: str
    inertia_kg_m2: float
    damping_nm_s_per_rad: float

    def __post_init__(self) -> None:
        for name in ("environment_id", "environment_version"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty string")
        for name in ("inertia_kg_m2", "damping_nm_s_per_rad"):
            value = getattr(self, name)
            validate_finite_scalar(name, value)
            object.__setattr__(self, name, float(value))
        if self.inertia_kg_m2 <= 0 or self.damping_nm_s_per_rad < 0:
            raise ValueError("inertia must be positive and damping nonnegative")


def expected_plant_parameters(environment_version: str) -> PlantParameters:
    """Committed narrow-plant contract, independently readable without MuJoCo.

    Effective capsule inertia plus 0.01 kg m² armature, cross-checked against
    compiled extraction at generation. The checksum binds all geometry, mass,
    damping and actuator settings; no other model can inherit these constants.
    """
    if environment_version != "8612091d73babb1a0122cb5163ea744183486cd8cc6fbae2813f45818be57263":
        raise ValueError("no independent parameter contract for model checksum")
    return PlantParameters(ENVIRONMENT_ID, environment_version, 0.23242512112403102, 0.05)


def load_plant_parameters(
    model_path: Path, expected_environment_version: str,
) -> PlantParameters:
    """Extract J = aᵀ R diag(I) Rᵀ a + m |r × a|² + armature.

    The axis a, COM offset r and principal-frame rotation R are all expressed
    in the moving body's frame. Only one rigid moving body and one direct
    torque motor are supported. Soft limits and contact forces are explicitly
    omitted by the analytical ODE; no simulator force is used as a coefficient.
    """
    content = model_path.resolve().read_bytes()
    checksum = hashlib.sha256(content).hexdigest()
    if checksum != expected_environment_version:
        raise ValueError("environment_version does not match model checksum")
    root = ET.fromstring(content)
    if any(node.tag == "include" or "file" in node.attrib for node in root.iter()):
        raise ValueError("unsupported external model assets: checksum must cover the plant")

    import mujoco
    import numpy as np

    if mujoco.__version__ != "3.16.0":
        raise ValueError("unsupported MuJoCo version")
    model = mujoco.MjModel.from_xml_string(content.decode("utf-8"))
    if ((model.nq, model.nv, model.nu, model.na, model.njnt) != (1, 1, 1, 0, 1)
            or model.jnt_type[0] != mujoco.mjtJoint.mjJNT_HINGE
            or any((model.neq, model.ntendon, model.nplugin, model.nmocap, model.nflex))):
        raise ValueError("unsupported model state or force features")
    body = int(model.jnt_bodyid[0])
    if np.any(model.body_parentid[1:] == body):
        raise ValueError("unsupported attached bodies: expected one rigid moving body")
    if (np.any(model.opt.gravity) or np.any(model.opt.wind)
            or model.opt.density != 0 or model.opt.viscosity != 0
            or model.jnt_stiffness[0] != 0 or model.dof_frictionloss[0] != 0
            or model.opt.disableflags != 0 or model.opt.enableflags != 0):
        raise ValueError("unsupported passive forces or simulator flags")
    if (model.actuator_trntype[0] != mujoco.mjtTrn.mjTRN_JOINT
            or model.actuator_trnid[0, 0] != 0
            or model.actuator_dyntype[0] != mujoco.mjtDyn.mjDYN_NONE
            or model.actuator_gaintype[0] != mujoco.mjtGain.mjGAIN_FIXED
            or model.actuator_biastype[0] != mujoco.mjtBias.mjBIAS_NONE
            or model.actuator_gainprm[0, 0] != 1
            or not np.array_equal(model.actuator_gear[0], [1, 0, 0, 0, 0, 0])
            or model.actuator_forcelimited[0] or model.jnt_actfrclimited[0]):
        raise ValueError("unsupported actuator: expected a direct unit-gain torque motor")

    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, model.body_iquat[body])
    rotation = rotation.reshape(3, 3)
    axis = model.jnt_axis[0]
    principal_axis = rotation.T @ axis
    offset = model.body_ipos[body] - model.jnt_pos[0]
    perpendicular = np.cross(offset, axis)
    rotational = float(np.dot(model.body_inertia[body], principal_axis**2))
    shifted = float(model.body_mass[body]) * float(np.dot(perpendicular, perpendicular))
    inertia = rotational + shifted + float(model.dof_armature[0])
    parameters = PlantParameters(ENVIRONMENT_ID, checksum, inertia, float(model.dof_damping[0]))

    # Independent compiled generalized-inertia check. Kinematics/CRB only:
    # no forward force, constraint solve, or simulator stepping is needed.
    data = mujoco.MjData(model)
    matrix = np.empty((1, 1))
    for angle in (-2.0, -1.0, 0.0, 0.7, 2.0):
        data.qpos[0] = angle
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        mujoco.mj_crb(model, data)
        mujoco.mj_fullM(model, data, matrix)
        if not math.isclose(inertia, float(matrix[0, 0]), rel_tol=1e-12, abs_tol=1e-14):
            raise ValueError("unsupported model: effective inertia disagrees with compiled mass matrix")
    return parameters
