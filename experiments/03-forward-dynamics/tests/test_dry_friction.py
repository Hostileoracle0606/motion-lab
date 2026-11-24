from math import exp

import numpy as np
import pytest

from forward_dynamics.dry_friction import AnalyticalDryFrictionPredictor
from forward_dynamics.plant import PlantParameters
from forward_dynamics.types import DynamicsContext, DynamicsState


CONTEXT = DynamicsContext("rotor", "fixture-v1", 1.0, 4.0)


def test_dry_friction_batch_handles_stick_breakaway_stop_and_reverse():
    predictor = AnalyticalDryFrictionPredictor(PlantParameters("rotor", "fixture-v1", 2.0, 0.0), CONTEXT, 1.0)
    states = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.25], [1.0, -0.25]])
    torques = np.array([0.5, 3.0, 0.0, 3.0])

    actual = predictor.predict_batch(states, torques, CONTEXT)

    np.testing.assert_allclose(actual, [[1.0, 0.0], [1.5, 1.0], [1.0625, 0.0], [1.3671875, 0.875]])
    assert predictor.predict(DynamicsState(1.0, -0.25), 3.0, CONTEXT) == DynamicsState(1.3671875, 0.875)


def test_dry_friction_integrates_viscous_damping_during_slip():
    predictor = AnalyticalDryFrictionPredictor(PlantParameters("rotor", "fixture-v1", 2.0, 2.0), CONTEXT, 1.0)

    actual = predictor.predict(DynamicsState(1.0, 2.0), 3.0, CONTEXT)

    assert actual.position_rad == pytest.approx(3.0 - exp(-1))
    assert actual.velocity_rad_s == pytest.approx(1.0 + exp(-1))


def test_dry_friction_solves_damped_stop_and_sticks():
    predictor = AnalyticalDryFrictionPredictor(PlantParameters("rotor", "fixture-v1", 2.0, 2.0), CONTEXT, 1.0)

    actual = predictor.predict(DynamicsState(0.0, 0.5), 0.0, CONTEXT)

    assert actual.position_rad == pytest.approx(0.15342640972002736)
    assert actual.velocity_rad_s == 0.0
