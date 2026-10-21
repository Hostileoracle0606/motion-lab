from pathlib import Path

import mujoco
import numpy as np


MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "one_link.xml"
COMPARISON_MODEL_PATH = (
    Path(__file__).resolve().parents[1] / "models" / "one_link_comparison.xml"
)


def test_model_has_one_joint_and_one_actuator() -> None:
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))

    assert model.nq == 1
    assert model.nv == 1
    assert model.nu == 1
    np.testing.assert_allclose(model.actuator_ctrlrange[0], [-3.0, 3.0])


def test_positive_torque_moves_joint_in_positive_direction() -> None:
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)

    initial_angle = float(data.qpos[0])
    data.ctrl[0] = 1.0

    for _ in range(200):
        mujoco.mj_step(model, data)

    assert float(data.qpos[0]) > initial_angle


def test_comparison_model_has_two_independent_display_arms() -> None:
    model = mujoco.MjModel.from_xml_path(str(COMPARISON_MODEL_PATH))

    assert (model.nq, model.nv, model.nu) == (2, 2, 2)
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "p_hinge") >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "pd_hinge") >= 0
    assert (
        mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_CAMERA,
            "comparison_camera",
        )
        >= 0
    )
    np.testing.assert_allclose(
        model.actuator_ctrlrange,
        [[-3.0, 3.0], [-3.0, 3.0]],
    )
