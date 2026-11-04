"""Physical-parameter extraction, independently checked against compiled inertia."""

import hashlib
from pathlib import Path

import mujoco
import numpy as np
import pytest

from forward_dynamics.plant import PlantParameters, load_plant_parameters


ROOT = Path(__file__).resolve().parents[3]
MODEL = ROOT / "experiments/01-feedback-control/models/one_link.xml"
VERSION = hashlib.sha256(MODEL.read_bytes()).hexdigest()
ENVIRONMENT = "motion-lab/experiment-01/one-link"
FIXTURE = """<mujoco>
  <option gravity="0 0 0"/>
  <worldbody><body>
    <inertial pos="0.2 0.3 0.4" mass="2" diaginertia="0.1 0.2 0.25"
              quat="0.7071067811865476 0 0 0.7071067811865476"/>
    <joint name="hinge" type="hinge" pos="0.1 0.3 0.1" axis="0 1 0"
           damping="0.05" armature="0.01"/>
  </body></worldbody>
  <actuator><motor joint="hinge"/></actuator>
</mujoco>"""


def write_model(tmp_path, text=FIXTURE):
    path = tmp_path / "plant.xml"
    path.write_text(text)
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_extracts_original_damping_and_armature_in_generalized_inertia() -> None:
    parameters = load_plant_parameters(MODEL, VERSION)
    assert parameters.environment_id == ENVIRONMENT
    assert parameters.environment_version == VERSION
    assert parameters.damping_nm_s_per_rad == 0.05
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    assert model.dof_armature[0] == 0.01
    for angle in (-2.0, -1.0, 0.0, 0.7, 2.0):
        data = mujoco.MjData(model)
        data.qpos[0] = angle
        mujoco.mj_forward(model, data)
        matrix = np.empty((1, 1))
        mujoco.mj_fullM(model, data, matrix)
        assert parameters.inertia_kg_m2 == pytest.approx(matrix[0, 0], abs=1e-14)
        assert parameters.inertia_kg_m2 - model.dof_armature[0] > 0


def test_rotates_principal_inertia_and_shifts_from_com_to_hinge(tmp_path) -> None:
    path, checksum = write_model(tmp_path)
    # Rotation makes I_yy = 0.1; perpendicular offset is (0.1, 0, 0.3).
    # J = 0.1 + 2 * (0.1**2 + 0.3**2) + 0.01 = 0.31.
    assert load_plant_parameters(path, checksum).inertia_kg_m2 == pytest.approx(0.31)


def test_rejects_checksum_mismatch() -> None:
    with pytest.raises(ValueError, match="checksum"):
        load_plant_parameters(MODEL, "0" * 64)


@pytest.mark.parametrize("old,new", [
    ('gravity="0 0 0"', 'gravity="0 0 -9.81"'),
    ('damping="0.05"', 'damping="0.05" stiffness="1"'),
    ('damping="0.05"', 'damping="0.05" frictionloss="0.1"'),
    ('<motor joint="hinge"/>', '<motor joint="hinge" gear="2"/>'),
    ('<motor joint="hinge"/>', '<position joint="hinge"/>'),
    ('type="hinge"', 'type="slide"'),
    ('</body>', '<body><geom size="0.1" mass="1"/></body></body>'),
])
def test_rejects_physics_outside_the_damped_rotor_contract(tmp_path, old, new) -> None:
    path, checksum = write_model(tmp_path, FIXTURE.replace(old, new))
    with pytest.raises(ValueError, match="unsupported"):
        load_plant_parameters(path, checksum)


@pytest.mark.parametrize("field,value", [
    ("inertia_kg_m2", 0), ("inertia_kg_m2", -1),
    ("inertia_kg_m2", float("inf")), ("inertia_kg_m2", True),
    ("damping_nm_s_per_rad", -1), ("damping_nm_s_per_rad", float("nan")),
    ("damping_nm_s_per_rad", False), ("environment_id", ""),
    ("environment_version", " "),
])
def test_rejects_invalid_parameters(field, value) -> None:
    values = dict(environment_id=ENVIRONMENT, environment_version=VERSION,
                  inertia_kg_m2=2.0, damping_nm_s_per_rad=0.05)
    values[field] = value
    with pytest.raises(ValueError):
        PlantParameters(**values)
