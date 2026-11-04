"""Public baseline behavior and shared prediction-input contracts."""

from dataclasses import replace

import pytest

from forward_dynamics.baselines import ConstantVelocityPredictor, NoChangePredictor
from forward_dynamics.types import DynamicsContext, DynamicsState


CONTEXT = DynamicsContext("one-link", "a" * 64, 0.01, 3.0)
PREDICTORS = (NoChangePredictor(), ConstantVelocityPredictor())


@pytest.mark.parametrize(
    ("predictor", "position", "velocity", "interval", "expected_position"),
    [
        (NoChangePredictor(), 1.0, 2.0, 0.01, 1.0),
        (ConstantVelocityPredictor(), 1.0, 2.0, 0.01, 1.02),
        (ConstantVelocityPredictor(), 1.0, -2.0, 0.1, 0.8),
        (ConstantVelocityPredictor(), 2.0, 2.0, 0.1, 2.2),
    ],
)
def test_baseline_equations_preserve_velocity_and_do_not_clamp_position(
    predictor, position, velocity, interval, expected_position,
) -> None:
    state = DynamicsState(position, velocity)
    result = predictor.predict(state, 0.0, replace(CONTEXT, control_interval_s=interval))

    assert result.position_rad == pytest.approx(expected_position)
    assert result.velocity_rad_s == velocity
    assert state == DynamicsState(position, velocity)


@pytest.mark.parametrize("predictor", PREDICTORS)
def test_valid_applied_torque_does_not_change_simple_prediction(predictor) -> None:
    results = [
        predictor.predict(DynamicsState(1.0, 2.0), torque, CONTEXT)
        for torque in (-3.0, 0.0, 3.0)
    ]

    assert results[0] == results[1] == results[2]


@pytest.mark.parametrize("predictor", PREDICTORS)
@pytest.mark.parametrize("field", ["position_rad", "velocity_rad_s"])
@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf"), True, "1", None])
def test_rejects_invalid_state_at_prediction_seam(predictor, field, invalid) -> None:
    state = replace(DynamicsState(1.0, 2.0), **{field: invalid})

    with pytest.raises(ValueError, match=field):
        predictor.predict(state, 0.0, CONTEXT)


@pytest.mark.parametrize("predictor", PREDICTORS)
@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), True, "1", None, -3.01, 3.01])
def test_rejects_invalid_or_unapplied_torque(predictor, invalid) -> None:
    with pytest.raises(ValueError, match="torque"):
        predictor.predict(DynamicsState(1.0, 2.0), invalid, CONTEXT)


@pytest.mark.parametrize("predictor", PREDICTORS)
@pytest.mark.parametrize("field", ["control_interval_s", "torque_limit_nm"])
@pytest.mark.parametrize("invalid", [0.0, -1.0, float("nan"), float("inf"), True, "1", None])
def test_rejects_invalid_context_scalars(predictor, field, invalid) -> None:
    with pytest.raises(ValueError, match=field):
        predictor.predict(
            DynamicsState(1.0, 2.0), 0.0, replace(CONTEXT, **{field: invalid}),
        )


@pytest.mark.parametrize("predictor", PREDICTORS)
@pytest.mark.parametrize("field", ["environment_id", "environment_version"])
@pytest.mark.parametrize("invalid", ["", " ", None, True])
def test_rejects_missing_environment_identity(predictor, field, invalid) -> None:
    with pytest.raises(ValueError, match=field):
        predictor.predict(
            DynamicsState(1.0, 2.0), 0.0, replace(CONTEXT, **{field: invalid}),
        )


def test_constant_velocity_rejects_non_finite_computed_position() -> None:
    with pytest.raises(ValueError, match="position_rad"):
        ConstantVelocityPredictor().predict(
            DynamicsState(1e308, 1e308), 0.0,
            replace(CONTEXT, control_interval_s=2.0),
        )
