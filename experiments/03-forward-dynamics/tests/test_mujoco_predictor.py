from dataclasses import replace
from pathlib import Path

import pytest

from forward_dynamics.dataset import load_prediction_cases
from forward_dynamics.mujoco_predictor import MujocoPredictor
from forward_dynamics.types import DynamicsState


ROOT = Path(__file__).resolve().parents[3]
MODEL = ROOT / "experiments/01-feedback-control/models/one_link.xml"
DATA = ROOT / "experiments/02-state-transition-dataset/data/dataset-v1"


def test_fresh_predictions_reproduce_every_recorded_next_state() -> None:
    predictor = MujocoPredictor(MODEL)
    cases = load_prediction_cases(DATA, "experiment-02-transition-v1")
    for case in cases:
        actual = predictor.predict(case.state, case.applied_torque_nm, case.context)
        assert actual.position_rad == pytest.approx(case.expected_next_state.position_rad, abs=1e-12, rel=0)
        assert actual.velocity_rad_s == pytest.approx(case.expected_next_state.velocity_rad_s, abs=1e-12, rel=0)
    first = cases[0]
    assert predictor.predict(first.state, first.applied_torque_nm, first.context) == cases[0].expected_next_state


@pytest.mark.parametrize(
    "changes",
    [{"environment_id": "wrong"}, {"environment_version": "0" * 64},
     {"control_interval_s": 0.003}, {"control_interval_s": 1e308}, {"torque_limit_nm": 2.0}],
)
def test_rejects_incompatible_prediction_context(changes) -> None:
    case = load_prediction_cases(DATA, "experiment-02-transition-v1")[0]
    with pytest.raises(ValueError):
        MujocoPredictor(MODEL).predict(DynamicsState(0.0, 0.0), 0.0, replace(case.context, **changes))


def test_rejects_changed_model_even_when_dimensions_match(tmp_path: Path) -> None:
    changed = tmp_path / "changed.xml"
    changed.write_text(MODEL.read_text().replace('damping="0.05"', 'damping="0.1"'))
    with pytest.raises(ValueError, match="model"):
        MujocoPredictor(changed)
