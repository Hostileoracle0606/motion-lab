"""Declared gain sweeps and response diagnostics for Experiment 01.

The diagnostics in this module support gain selection and demonstration design.
They are not a replacement for the experiment's formal evaluation metrics.
"""

from dataclasses import dataclass, replace
import math
from pathlib import Path
import tomllib

import mujoco

from feedback_control.config import ExperimentConfig
from feedback_control.simulation import ControllerName, EpisodeResult, run_episode


@dataclass(frozen=True)
class GainSweepConfig:
    p_kp_values: tuple[float, ...]
    pd_kp: float
    pd_kd_values: tuple[float, ...]


@dataclass(frozen=True)
class ResponseDiagnostics:
    controller: ControllerName
    kp: float
    kd: float
    final_abs_error_rad: float
    final_abs_velocity_rad_s: float
    overshoot_rad: float
    max_abs_velocity_rad_s: float
    saturated_steps: int
    target_crossings: int


def load_gain_sweep_config(path: Path) -> GainSweepConfig:
    """Load and validate the declared P and PD gain candidates."""
    with path.resolve().open("rb") as config_file:
        raw = tomllib.load(config_file)

    sweep = GainSweepConfig(
        p_kp_values=tuple(float(value) for value in raw["p"]["kp_values"]),
        pd_kp=float(raw["pd"]["kp"]),
        pd_kd_values=tuple(float(value) for value in raw["pd"]["kd_values"]),
    )
    _validate_gain_candidates("p.kp_values", sweep.p_kp_values)
    _validate_gain_candidates("pd.kd_values", sweep.pd_kd_values)
    _validate_nonnegative_finite("pd.kp", sweep.pd_kp)
    return sweep


def run_gain_sweep(
    base_config: ExperimentConfig,
    sweep: GainSweepConfig,
) -> tuple[ResponseDiagnostics, ...]:
    """Run every declared candidate under the same base scenario."""
    rows: list[ResponseDiagnostics] = []

    for kp in sweep.p_kp_values:
        candidate = replace(base_config, p=replace(base_config.p, kp=kp))
        result = run_episode(candidate, "p")
        rows.append(summarize_response(result, kp=kp, kd=0.0))

    for kd in sweep.pd_kd_values:
        candidate = replace(
            base_config,
            pd=replace(base_config.pd, kp=sweep.pd_kp, kd=kd),
        )
        result = run_episode(candidate, "pd")
        rows.append(summarize_response(result, kp=sweep.pd_kp, kd=kd))

    return tuple(rows)


def summarize_response(
    result: EpisodeResult,
    *,
    kp: float,
    kd: float,
) -> ResponseDiagnostics:
    """Summarize visible response traits without claiming formal evaluation."""
    if not result.samples:
        raise ValueError("cannot summarize an episode with no samples")

    start_position = result.samples[0].position_rad
    target_position = result.samples[0].target_position_rad
    commanded_change = target_position - start_position
    if math.isclose(commanded_change, 0.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("gain tuning requires a nonzero commanded position change")

    positions = [sample.position_rad for sample in result.samples]
    positions.append(result.final_position_rad)
    velocities = [sample.velocity_rad_s for sample in result.samples]
    velocities.append(result.final_velocity_rad_s)
    errors = [target_position - position for position in positions]

    command_direction = math.copysign(1.0, commanded_change)
    overshoot_rad = max(
        0.0,
        max((position - target_position) * command_direction for position in positions),
    )

    return ResponseDiagnostics(
        controller=result.controller,
        kp=kp,
        kd=kd,
        final_abs_error_rad=abs(errors[-1]),
        final_abs_velocity_rad_s=abs(velocities[-1]),
        overshoot_rad=overshoot_rad,
        max_abs_velocity_rad_s=max(abs(velocity) for velocity in velocities),
        saturated_steps=sum(sample.saturated for sample in result.samples),
        target_crossings=_count_sign_changes(errors),
    )


def estimate_critical_derivative_gain(
    config: ExperimentConfig,
    *,
    kp: float,
) -> float:
    """Estimate Kd for critical damping of the local linear one-DOF model.

    This estimate assumes continuous time, zero gravity, no saturation, and a
    fixed local inertia: Kd = 2 * sqrt(inertia * Kp) - passive_damping.
    """
    _validate_nonnegative_finite("kp", kp)
    model = mujoco.MjModel.from_xml_path(str(config.model.path))
    if model.nv != 1:
        raise ValueError("critical-damping estimate requires a one-DOF model")

    data = mujoco.MjData(model)
    keyframe_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_KEY,
        config.model.keyframe,
    )
    if keyframe_id < 0:
        raise ValueError(f"unknown model keyframe: {config.model.keyframe!r}")
    mujoco.mj_resetDataKeyframe(model, data, keyframe_id)
    mujoco.mj_forward(model, data)

    effective_inertia = float(data.M[0])
    passive_damping = float(model.dof_damping[0])
    return max(0.0, 2.0 * math.sqrt(effective_inertia * kp) - passive_damping)


def nearest_candidate(candidates: tuple[float, ...], target: float) -> float:
    """Choose the declared candidate nearest a theory-derived target."""
    if not candidates:
        raise ValueError("at least one candidate is required")
    return min(candidates, key=lambda candidate: (abs(candidate - target), candidate))


def _count_sign_changes(values: list[float]) -> int:
    nonzero_signs = [math.copysign(1.0, value) for value in values if value != 0.0]
    return sum(
        current != previous
        for previous, current in zip(nonzero_signs, nonzero_signs[1:])
    )


def _validate_gain_candidates(name: str, candidates: tuple[float, ...]) -> None:
    if not candidates:
        raise ValueError(f"{name} must not be empty")
    if len(set(candidates)) != len(candidates):
        raise ValueError(f"{name} must not contain duplicates")
    for candidate in candidates:
        _validate_nonnegative_finite(name, candidate)


def _validate_nonnegative_finite(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
