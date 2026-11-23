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

def test_unstable_velocity_cannot_silently_reset_to_success():
    plant = MujocoPlant(XML)
    plant.reset(DynamicsState(0, 1e200))
    step = plant.advance(0)
    assert step.termination == 'invalid_state'
    assert step.next_state is None
    assert step.physics_steps == 1
