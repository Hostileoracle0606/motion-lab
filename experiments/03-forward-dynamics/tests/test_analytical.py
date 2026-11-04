"""Independent exact-solution tests for the simplified constant-torque rotor."""

from dataclasses import replace
import math
from pathlib import Path
import subprocess
import sys

import mujoco
import numpy as np
import pytest

from forward_dynamics.analytical import AnalyticalRK4Predictor
from forward_dynamics.dataset import load_prediction_cases
from forward_dynamics.evaluation import evaluate_one_step
from forward_dynamics.plant import PlantParameters, load_plant_parameters
from forward_dynamics.types import DynamicsContext, DynamicsState


PARAMETERS = PlantParameters("rotor", "fixture-v1", 2.0, 0.0)
CONTEXT = DynamicsContext("rotor", "fixture-v1", 0.1, 10.0)
ROOT = Path(__file__).resolve().parents[3]


def test_zero_damping_exact_constant_acceleration_and_torque_sign() -> None:
    predictor = AnalyticalRK4Predictor(PARAMETERS, 0.01)
    actual = predictor.predict(DynamicsState(1.0, 3.0), 4.0, CONTEXT)
    assert actual.position_rad == pytest.approx(1.31, abs=1e-14)
    assert actual.velocity_rad_s == pytest.approx(3.2, abs=1e-14)
    negative = predictor.predict(DynamicsState(1.0, 3.0), -4.0, CONTEXT)
    assert negative.position_rad == pytest.approx(1.29, abs=1e-14)
    assert negative.velocity_rad_s == pytest.approx(2.8, abs=1e-14)


@pytest.mark.parametrize("duration", [0.01, 1.0])
def test_damped_rotor_converges_to_independent_exponential_solution(duration) -> None:
    parameters = replace(PARAMETERS, damping_nm_s_per_rad=8.0)
    context = replace(CONTEXT, control_interval_s=duration)
    # J=2, b=8, u=4 => v_inf=0.5, decay=exp(-4t).
    decay = math.exp(-4 * duration)
    exact_q = 1 + 0.5 * duration + 2.5 * (1 - decay) / 4
    exact_v = 0.5 + 2.5 * decay
    errors = []
    for step in (0.01, 0.005):
        actual = AnalyticalRK4Predictor(parameters, step).predict(DynamicsState(1, 3), 4, context)
        errors.append((abs(actual.position_rad - exact_q), abs(actual.velocity_rad_s - exact_v)))
    assert errors[0][0] > 1e-12 and errors[0][1] > 1e-12
    assert errors[1][0] < errors[0][0] / 10
    assert errors[1][1] < errors[0][1] / 10
    assert errors[1][0] < 1e-8 and errors[1][1] < 1e-8


@pytest.mark.parametrize("step,identifier", [
    (0.002, "analytical_rk4_0.002"), (0.001, "analytical_rk4_0.001"),
    (0.0005, "analytical_rk4_0.0005"),
])
def test_frozen_steps_have_stable_ids_and_calls_are_order_independent(step, identifier) -> None:
    predictor = AnalyticalRK4Predictor(PARAMETERS, step)
    expected = predictor.predict(DynamicsState(1, 3), 4, CONTEXT)
    predictor.predict(DynamicsState(-2, -4), -2, CONTEXT)
    assert predictor.predict(DynamicsState(1, 3), 4, CONTEXT) == expected
    assert predictor.predictor_id == identifier


@pytest.mark.parametrize("step", [True, 0, -1, float("nan"), float("inf"), "0.01"])
def test_rejects_invalid_integration_steps(step) -> None:
    with pytest.raises(ValueError):
        AnalyticalRK4Predictor(PARAMETERS, step)


@pytest.mark.parametrize("duration,step", [(0.1, 1e100), (0.1, 0.03), (0.1, 1e-100)])
def test_requires_a_positive_bounded_integer_substep_count(duration, step) -> None:
    predictor = AnalyticalRK4Predictor(PARAMETERS, step)
    with pytest.raises(ValueError, match="substeps"):
        predictor.predict(DynamicsState(1, 3), 4, replace(CONTEXT, control_interval_s=duration))


@pytest.mark.parametrize("field,value", [
    ("environment_id", "other"), ("environment_version", "other"),
    ("control_interval_s", True), ("control_interval_s", 0),
    ("torque_limit_nm", float("inf")),
])
def test_rejects_invalid_or_mismatched_context(field, value) -> None:
    with pytest.raises(ValueError):
        AnalyticalRK4Predictor(PARAMETERS, 0.01).predict(
            DynamicsState(1, 3), 4, replace(CONTEXT, **{field: value}),
        )


@pytest.mark.parametrize("state,torque", [
    (DynamicsState(True, 0), 0), (DynamicsState(0, float("inf")), 0),
    (DynamicsState(0, 0), True), (DynamicsState(0, 0), 11),
    (DynamicsState(0, 0), np.int64(-9223372036854775808)),
])
def test_rejects_invalid_state_and_actions(state, torque) -> None:
    with pytest.raises(ValueError):
        AnalyticalRK4Predictor(PARAMETERS, 0.01).predict(state, torque, CONTEXT)


def test_normalizes_fixed_width_scalars_before_dynamics_arithmetic() -> None:
    parameters = replace(PARAMETERS, inertia_kg_m2=np.int64(2))
    context = replace(CONTEXT, torque_limit_nm=1e20)
    actual = AnalyticalRK4Predictor(parameters, 0.01).predict(
        DynamicsState(np.int64(0), np.int64(0)), np.int64(-9223372036854775808), context,
    )
    assert actual.position_rad == pytest.approx(-23058430092136940)
    assert actual.velocity_rad_s == pytest.approx(-461168601842738800)
    assert type(actual.position_rad) is float and type(actual.velocity_rad_s) is float


def test_prediction_never_calls_simulator_dynamics(monkeypatch) -> None:
    def forbidden(*args, **kwargs):
        pytest.fail("pure RK4 prediction called simulator dynamics")
    for name in ("mj_step", "mj_forward", "mj_fwdAcceleration", "mj_fwdVelocity", "mj_inverse"):
        monkeypatch.setattr(mujoco, name, forbidden)
    assert AnalyticalRK4Predictor(PARAMETERS, 0.01).predict(DynamicsState(1, 3), 4, CONTEXT).velocity_rad_s == pytest.approx(3.2)


def test_ordinary_import_and_prediction_work_when_mujoco_import_is_forbidden() -> None:
    program = """
import sys
sys.modules['mujoco'] = None
from forward_dynamics.plant import PlantParameters
from forward_dynamics.analytical import AnalyticalRK4Predictor
from forward_dynamics.types import DynamicsContext, DynamicsState
p = AnalyticalRK4Predictor(PlantParameters('rotor', 'v1', 2, 0), 0.01)
assert abs(p.predict(DynamicsState(1, 3), 4, DynamicsContext('rotor', 'v1', 0.1, 4)).position_rad - 1.31) < 1e-14
"""
    subprocess.run([sys.executable, "-c", program], check=True, cwd=ROOT)


def test_joint_limit_failure_remains_in_common_evaluation_without_clamping() -> None:
    cases = load_prediction_cases(ROOT / "experiments/02-state-transition-dataset/data/dataset-v1", "experiment-02-transition-v1")
    boundary = next(case for case in cases if case.termination_reason == "joint_limit_violation")
    model = ROOT / "experiments/01-feedback-control/models/one_link.xml"
    predictor = AnalyticalRK4Predictor(load_plant_parameters(model, boundary.context.environment_version), 0.002)
    rows = evaluate_one_step(cases, predictor)
    assert len(rows) == len(cases) == 2540
    boundary_row = next(row for row in rows if row.case == boundary)
    assert math.isfinite(boundary_row.predicted_next_state.position_rad)
    beyond_limit = predictor.predict(DynamicsState(2, 1), 0, boundary.context)
    assert beyond_limit.position_rad > 2
