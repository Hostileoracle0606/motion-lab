"""Physical-unit errors with independent, hand-computable expectations."""

import math
from pathlib import Path
import subprocess
import sys

import pytest

from forward_dynamics.metrics import StateResidual, residual, summarize_errors
from forward_dynamics.types import DynamicsState


def test_residual_is_signed_predicted_minus_expected_without_wrapping() -> None:
    result = residual(DynamicsState(3.0, -2.0), DynamicsState(-3.0, 1.0))

    assert result.position_error_rad == 6.0
    assert result.velocity_error_rad_s == -3.0


def test_summarizes_components_independently_in_physical_units() -> None:
    result = summarize_errors((StateResidual(-1.0, 0.0), StateResidual(3.0, -4.0)))

    assert result.sample_count == 2
    assert result.position_mae_rad == 2.0
    assert result.position_rmse_rad == pytest.approx(math.sqrt(5.0))
    assert result.position_max_abs_rad == 3.0
    assert result.velocity_mae_rad_s == 2.0
    assert result.velocity_rmse_rad_s == pytest.approx(math.sqrt(8.0))
    assert result.velocity_max_abs_rad_s == 4.0


def test_summary_weights_each_transition_equally() -> None:
    # One episode supplies one error of 4; another supplies three errors of 0.
    result = summarize_errors((StateResidual(4.0, 0.0),) + (StateResidual(0.0, 0.0),) * 3)

    assert result.sample_count == 4
    assert result.position_mae_rad == 1.0
    assert result.position_rmse_rad == 2.0


@pytest.mark.parametrize("value", [0.0, 1e308, 1e-300])
def test_summary_remains_finite_without_squaring_overflow_or_underflow(value) -> None:
    result = summarize_errors((StateResidual(value, -value),) * 2)

    assert result.position_mae_rad == value
    assert result.position_rmse_rad == value
    assert result.velocity_rmse_rad_s == value


def test_rejects_empty_errors() -> None:
    with pytest.raises(ValueError, match="at least one"):
        summarize_errors(())


@pytest.mark.parametrize("field", ["position_error_rad", "velocity_error_rad_s"])
@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf"), True, "1", None])
def test_summary_rejects_invalid_error_fields(field, invalid) -> None:
    values = {"position_error_rad": 0.0, "velocity_error_rad_s": 0.0, field: invalid}

    with pytest.raises(ValueError, match=field):
        summarize_errors((StateResidual(**values),))


@pytest.mark.parametrize("bad_state", [DynamicsState(float("nan"), 0.0), DynamicsState(0.0, True)])
@pytest.mark.parametrize("bad_is_predicted", [True, False])
def test_residual_validates_both_states(bad_state, bad_is_predicted) -> None:
    valid = DynamicsState(0.0, 0.0)
    args = (bad_state, valid) if bad_is_predicted else (valid, bad_state)

    with pytest.raises(ValueError):
        residual(*args)


def test_residual_rejects_overflow_from_finite_states() -> None:
    with pytest.raises(ValueError, match="position_error_rad"):
        residual(DynamicsState(1e308, 0.0), DynamicsState(-1e308, 0.0))


def test_ordinary_metrics_import_does_not_import_mujoco(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-c", "import sys; import forward_dynamics.metrics; assert 'mujoco' not in sys.modules"],
        cwd=tmp_path, capture_output=True, text=True,
    )

    assert result.returncode == 0, result.stderr
