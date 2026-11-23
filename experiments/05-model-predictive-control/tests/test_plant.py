from pathlib import Path
import mujoco
import pytest
from forward_dynamics.types import DynamicsState
from model_predictive_control.plant import MujocoPlant

XML = Path(__file__).resolve().parents[2] / '01-feedback-control/models/one_link.xml'

def test_direct_mujoco_interval_and_reset():
    plant = MujocoPlant(XML)
    state = DynamicsState(0.2, -0.3)
    assert plant.reset(state) == state
    step = plant.advance(1.2)
    model = mujoco.MjModel.from_xml_path(str(XML))
    data = mujoco.MjData(model)
    data.qpos[0], data.qvel[0], data.ctrl[0] = 0.2, -0.3, 1.2
    mujoco.mj_forward(model, data)
    for _ in range(5):
        mujoco.mj_step(model, data)
    assert step.physics_steps == 5 and step.termination is None
    assert step.next_state.position_rad == pytest.approx(data.qpos[0])
    assert step.next_state.velocity_rad_s == pytest.approx(data.qvel[0])
    plant.reset(state)
    assert plant.advance(1.2) == step

def test_plant_rejects_invalid_action_and_xml(tmp_path):
    plant = MujocoPlant(XML)
    plant.reset(DynamicsState(0, 0))
    for value in (3.1, float('nan')):
        with pytest.raises(ValueError): plant.advance(value)
    bad = tmp_path / 'bad.xml'
    bad.write_text('<mujoco/>')
    with pytest.raises(ValueError, match='checksum'): MujocoPlant(bad)

def test_joint_violation_stops_early():
    plant = MujocoPlant(XML)
    plant.reset(DynamicsState(1.5707, 2.0))
    step = plant.advance(0)
    assert step.termination == 'joint_limit' and step.physics_steps < 5

def test_unstable_velocity_cannot_silently_reset_to_success(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    plant = MujocoPlant(XML)
    plant.reset(DynamicsState(0, 1e200))
    step = plant.advance(0)
    assert step.termination == 'invalid_state'
    assert step.next_state is None
    assert step.physics_steps == 1


def test_model_checksum_and_perturbed_dynamics():
    import hashlib

    nominal = MujocoPlant(XML)
    explicit = MujocoPlant(XML, link_mass_kg=1.0, frictionloss_nm=0.0)
    changed = MujocoPlant(XML, link_mass_kg=1.5, frictionloss_nm=0.2)
    repeated = MujocoPlant(XML, link_mass_kg=1.5, frictionloss_nm=0.2)
    assert nominal.model_sha256 == hashlib.sha256(XML.read_bytes()).hexdigest()
    assert explicit.model_sha256 == nominal.model_sha256
    assert changed.model_sha256 == repeated.model_sha256 != nominal.model_sha256
    for plant in (nominal, explicit, changed):
        plant.reset(DynamicsState(0, 0))
    nominal_step = nominal.advance(1.2)
    assert explicit.advance(1.2) == nominal_step
    assert changed.advance(1.2).next_state != nominal_step.next_state


@pytest.mark.parametrize('parameters', [
    {'link_mass_kg': 1.5}, {'frictionloss_nm': 0.2},
])
def test_each_perturbation_changes_dynamics(parameters):
    nominal = MujocoPlant(XML)
    changed = MujocoPlant(XML, **parameters)
    for plant in (nominal, changed):
        plant.reset(DynamicsState(0, 0))
    assert changed.advance(1.2).next_state != nominal.advance(1.2).next_state


@pytest.mark.parametrize('name, value', [
    ('link_mass_kg', 0), ('link_mass_kg', -1), ('link_mass_kg', True),
    ('link_mass_kg', float('nan')), ('link_mass_kg', float('inf')),
    ('frictionloss_nm', -1), ('frictionloss_nm', False),
    ('frictionloss_nm', float('nan')), ('frictionloss_nm', float('inf')),
])
def test_rejects_invalid_perturbations(name, value):
    with pytest.raises(ValueError):
        MujocoPlant(XML, **{name: value})


def test_perturbation_does_not_bypass_input_checksum(tmp_path):
    altered = tmp_path / 'altered.xml'
    altered.write_bytes(XML.read_bytes() + b'\n')
    with pytest.raises(ValueError, match='checksum'):
        MujocoPlant(altered, link_mass_kg=1.5)
