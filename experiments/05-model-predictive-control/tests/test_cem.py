import numpy as np
import pytest

from forward_dynamics.types import DynamicsState
from model_predictive_control.contracts import NOMINAL_CONTEXT
from model_predictive_control.planner import CostWeights


def run(predictor, **kwargs):
    from model_predictive_control.cem import plan_cem
    options = dict(horizon=1, candidate_count=7, iterations=3, elite_count=2,
                   min_std_nm=0.1, rng=np.random.default_rng(42))
    options.update(kwargs)
    return plan_cem(predictor, NOMINAL_CONTEXT, DynamicsState(0, 0), 0,
                    CostWeights(0, 0, 1, 0), **options)


class RecordingPredictor:
    def __init__(self, reject=lambda torque: False):
        self.torques = []
        self.reject = reject

    def predict(self, state, torque, context):
        self.torques.append(torque)
        if self.reject(torque):
            raise ValueError('infeasible')
        return DynamicsState(0, 0)


def test_unequal_batches_rank_feasible_elites_and_clip():
    p = RecordingPredictor(lambda x: x > 0)
    result = run(p)
    rng = np.random.default_rng(42)
    first = rng.uniform(-3, 3, (3, 1))
    elites = first[first[:, 0] <= 0][:, 0]
    second = np.clip(rng.normal(elites.mean(), max(elites.std(), 0.1), (2, 1)), -3, 3)
    feasible = second[second[:, 0] <= 0][:, 0]
    third = np.clip(rng.normal(feasible.mean(), max(feasible.std(), 0.1), (2, 1)), -3, 3)
    expected = np.concatenate((first, second, third)).ravel()
    assert p.torques == pytest.approx(expected)
    valid = expected[expected <= 0]
    assert result.actions_nm == (valid[np.argmin(valid ** 2)],)
    assert (result.candidates, result.feasible_candidates, result.predictor_calls) == (7, len(valid), 7)


def test_all_infeasible_preserves_uniform_and_counts_early_rejections():
    p = RecordingPredictor(lambda x: True)
    result = run(p, horizon=3)
    rng = np.random.default_rng(42)
    expected = np.concatenate([rng.uniform(-3, 3, (n, 3))[:, 0] for n in (3, 2, 2)])
    assert p.torques == pytest.approx(expected)
    assert result.actions_nm is None and result.cost is None
    assert (result.candidates, result.feasible_candidates, result.predictor_calls) == (7, 0, 7)


def test_seed_determinism_and_bounds():
    a, b = RecordingPredictor(), RecordingPredictor()
    assert run(a, min_std_nm=100) == run(b, min_std_nm=100)
    assert a.torques == b.torques
    assert all(-3 <= x <= 3 for x in a.torques)


@pytest.mark.parametrize('override', [dict(horizon=0), dict(candidate_count=True), dict(iterations=0),
    dict(iterations=8), dict(elite_count=0), dict(elite_count=3), dict(min_std_nm=0),
    dict(min_std_nm=float('nan')), dict(min_std_nm=True)])
def test_invalid_settings(override):
    with pytest.raises(ValueError):
        run(RecordingPredictor(), **override)


def test_unexpected_predictor_exception_propagates():
    class Broken:
        def predict(self, *args):
            raise RuntimeError('bug')
    with pytest.raises(RuntimeError, match='bug'):
        run(Broken())


def test_equal_cost_keeps_first_candidate_across_rounds():
    from model_predictive_control.cem import plan_cem
    p = RecordingPredictor()
    result = plan_cem(p, NOMINAL_CONTEXT, DynamicsState(0, 0), 0,
                      CostWeights(0, 0, 0, 0), horizon=1, candidate_count=7,
                      iterations=3, elite_count=2, min_std_nm=0.1,
                      rng=np.random.default_rng(42))
    assert result.actions_nm == (p.torques[0],)
    rng = np.random.default_rng(42)
    first = rng.uniform(-3, 3, (3, 1))
    second = np.clip(rng.normal(first[:2].mean(), first[:2].std(), (2, 1)), -3, 3)
    assert p.torques[3:5] == pytest.approx(second.ravel())


def test_empty_round_retains_previous_distribution():
    class RejectSecondRound(RecordingPredictor):
        def predict(self, state, torque, context):
            self.torques.append(torque)
            if 3 < len(self.torques) <= 5:
                raise ValueError('infeasible')
            return DynamicsState(0, 0)

    p = RejectSecondRound()
    result = run(p)
    rng = np.random.default_rng(42)
    first = rng.uniform(-3, 3, (3, 1))
    elites = first[np.argsort(first[:, 0] ** 2, kind='stable')[:2], 0]
    mean, std = elites.mean(), max(elites.std(), 0.1)
    second = np.clip(rng.normal(mean, std, (2, 1)), -3, 3)
    third = np.clip(rng.normal(mean, std, (2, 1)), -3, 3)
    assert p.torques == pytest.approx(np.concatenate((first, second, third)).ravel())
    assert (result.feasible_candidates, result.predictor_calls) == (5, 7)
