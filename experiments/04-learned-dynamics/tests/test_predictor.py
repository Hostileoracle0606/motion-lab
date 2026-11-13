from dataclasses import replace
import math

import numpy as np
import pytest

from forward_dynamics.types import DynamicsContext, DynamicsState
from learned_dynamics.types import FEATURE_ORDER, FEATURE_UNITS, Normalizer


CONTEXT = DynamicsContext("motion-lab/experiment-01/one-link",
    "8612091d73babb1a0122cb5163ea744183486cd8cc6fbae2813f45818be57263", .01, 3.0)


@pytest.fixture
def norm():
    return Normalizer(1, FEATURE_ORDER, FEATURE_UNITS, (0., 0., 0.), (1., 1., 1.),
                      (False, False, False), (.02, -.1), (1., 1.), (False, False),
                      1, "experiment-02-transition-v1", "b" * 64, "c" * 64, (("a", 0),))


@pytest.fixture
def make_predictor(norm):
    def make(history=0, device="cpu", torque=False):
        import torch
        from learned_dynamics.model import ResidualMLP
        from learned_dynamics.predictor import LearnedPredictor
        from learned_dynamics.types import ModelSpec
        model = ResidualMLP(ModelSpec(history_steps=history))
        with torch.no_grad():
            for p in model.parameters():
                p.zero_()
            if torque:
                model.network[0].weight[0, 2] = 1
                model.network[2].weight[0, 0] = 1
                model.network[4].weight[0, 0] = 1
            if history:
                model.network[0].weight[0, 3] = 1
                model.network[2].weight[0, 0] = 1
                model.network[4].weight[0, 0] = 1
        return LearnedPredictor(model, norm, CONTEXT, "synthetic", device)
    return make


def test_zero_normalized_output_means_saved_mean_delta(make_predictor):
    p = make_predictor()
    assert p.predict(DynamicsState(1, 2), 0., CONTEXT) == DynamicsState(1.02, 1.9)


def test_predict_a_b_a_is_stateless(make_predictor):
    p = make_predictor(torque=True)
    a = p.predict(DynamicsState(1, 2), 1., CONTEXT)
    p.predict(DynamicsState(-1, -2), -1., CONTEXT)
    assert p.predict(DynamicsState(1, 2), 1., CONTEXT) == a
    assert a.position_rad == pytest.approx(1.02 + math.tanh(math.tanh(1)), abs=1e-6)


def test_torque_sign_reaches_network(make_predictor):
    p = make_predictor(torque=True)
    assert p.predict(DynamicsState(0, 0), 1., CONTEXT).position_rad > .02
    assert p.predict(DynamicsState(0, 0), -1., CONTEXT).position_rad < .02


@pytest.mark.parametrize("context", [
    replace(CONTEXT, environment_version="d" * 64),
    replace(CONTEXT, control_interval_s=.02), replace(CONTEXT, torque_limit_nm=2),
    replace(CONTEXT, environment_id="different"),
])
def test_wrong_context_rejected(make_predictor, context):
    with pytest.raises(ValueError, match="context"):
        make_predictor().predict(DynamicsState(1, 2), 0., context)


def test_invalid_inputs_and_unclipped_outputs(make_predictor):
    p = make_predictor()
    for state, action in [(DynamicsState(True, 0), 0), (DynamicsState(float("nan"), 0), 0),
                          (DynamicsState(0, 0), 4), (DynamicsState(0, 0), True)]:
        with pytest.raises(ValueError):
            p.predict(state, action, CONTEXT)
    assert p.predict(DynamicsState(10, 0), 0., CONTEXT).position_rad == 10.02


def test_batch_equals_single(make_predictor):
    p = make_predictor(torque=True)
    states, actions = np.array([[1., 2.], [-1., -2.]]), np.array([1., -1.])
    batch = p.predict_batch(states, actions, CONTEXT)
    expected = [p.predict(DynamicsState(*s), float(a), CONTEXT) for s, a in zip(states, actions)]
    np.testing.assert_allclose(batch, [[s.position_rad, s.velocity_rad_s] for s in expected], atol=1e-6)


def test_history_is_explicit_and_canonical_call_rejects_it(make_predictor):
    p = make_predictor(history=1)
    with pytest.raises(ValueError, match="history"):
        p.predict(DynamicsState(0, 0), 0., CONTEXT)
    result = p.predict_with_history(DynamicsState(0, 0), 0., DynamicsState(1, 0), -1., CONTEXT)
    assert result.position_rad == pytest.approx(.02 + math.tanh(math.tanh(1)), abs=1e-6)


def test_nonfinite_model_output_is_contextual_failure(make_predictor):
    import torch
    p = make_predictor()
    with torch.no_grad():
        p.model.network[4].bias.fill_(float("inf"))
    with pytest.raises(ValueError, match="synthetic"):
        p.predict(DynamicsState(0, 0), 0., CONTEXT)


def test_actual_mps_prediction_repeatability_and_batch(make_predictor):
    import torch
    if not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable; GPU prediction not established")
    p = make_predictor(device="mps", torque=True)
    a = p.predict(DynamicsState(1, 2), 1., CONTEXT)
    p.predict(DynamicsState(-1, 0), -1., CONTEXT)
    assert p.predict(DynamicsState(1, 2), 1., CONTEXT) == a
    actual = p.predict_batch(np.array([[1., 2.]]), np.array([1.]), CONTEXT)
    np.testing.assert_allclose(actual, [[a.position_rad, a.velocity_rad_s]], atol=1e-6, rtol=0)
