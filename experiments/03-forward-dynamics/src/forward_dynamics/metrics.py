"""Pure physical-unit residuals and transition-weighted error summaries."""

from dataclasses import dataclass
import math

from forward_dynamics.types import DynamicsState, validate_finite_scalar, validate_state


@dataclass(frozen=True)
class StateResidual:
    """Signed predicted-minus-expected error for each state coordinate."""

    position_error_rad: float
    velocity_error_rad_s: float


@dataclass(frozen=True)
class ErrorSummary:
    """Equal weight per supplied transition, with separate physical units."""

    sample_count: int
    position_mae_rad: float
    position_rmse_rad: float
    position_max_abs_rad: float
    velocity_mae_rad_s: float
    velocity_rmse_rad_s: float
    velocity_max_abs_rad_s: float


def residual(predicted: DynamicsState, expected: DynamicsState) -> StateResidual:
    """Subtract directly for a limited joint; never wrap an angular excursion."""
    validate_state(predicted)
    validate_state(expected)
    result = StateResidual(
        position_error_rad=float(predicted.position_rad) - float(expected.position_rad),
        velocity_error_rad_s=float(predicted.velocity_rad_s) - float(expected.velocity_rad_s),
    )
    _validate_residual(result)
    return result


def summarize_errors(errors: tuple[StateResidual, ...]) -> ErrorSummary:
    """Return per-component MAE, RMSE and max absolute error; reject empty input.

    Each residual has one vote. Group by episode before calling this function
    to retain episode-level results; combining episodes weights their lengths,
    not their episode means. No sample is filtered out.
    """
    if not errors:
        raise ValueError("errors must contain at least one residual")
    for error in errors:
        _validate_residual(error)
    position = _component_summary(tuple(float(error.position_error_rad) for error in errors))
    velocity = _component_summary(tuple(float(error.velocity_error_rad_s) for error in errors))
    return ErrorSummary(len(errors), *position, *velocity)


def _validate_residual(error: StateResidual) -> None:
    if not isinstance(error, StateResidual):
        raise ValueError("errors must contain StateResidual records")
    validate_finite_scalar("position_error_rad", error.position_error_rad)
    validate_finite_scalar("velocity_error_rad_s", error.velocity_error_rad_s)


def _component_summary(values: tuple[float, ...]) -> tuple[float, float, float]:
    scale = max(abs(value) for value in values)
    if scale == 0.0:
        return 0.0, 0.0, 0.0
    # Scaling keeps finite large errors from overflowing when summed/squared,
    # and preserves tiny errors that would underflow if squared directly.
    normalized = tuple(abs(value) / scale for value in values)
    mae = scale * (math.fsum(normalized) / len(values))
    rmse = scale * math.sqrt(math.fsum(value * value for value in normalized) / len(values))
    return mae, rmse, scale
