"""Reproducible multi-scenario orchestration for Experiment 01."""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
import hashlib
import json
import math
from importlib.metadata import version
from pathlib import Path
import platform
from typing import Any, Literal, cast

from feedback_control.config import (
    EpisodeConfig,
    EvaluationConfig,
    EvaluationPlanCase,
    ExperimentConfig,
    build_evaluation_plan,
    load_evaluation_config,
)
from feedback_control.evaluation import EpisodeMetrics, evaluate_episode
from feedback_control.metrics import TimingOutcome
from feedback_control.simulation import EpisodeResult, TrajectorySample, run_episode
from feedback_control.trajectory import (
    TrajectoryRecord,
    load_trajectory,
    save_trajectory,
)


EVALUATION_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class EvaluationProvenance:
    code_version: str
    working_tree_dirty: bool
    config_file: str
    config_sha256: str
    command: str
    python_version: str
    platform: str
    mujoco_version: str


@dataclass(frozen=True)
class EvaluatedEpisode:
    episode_id: str
    controller: Literal["p", "pd"]
    scenario_id: str
    seed: int
    status: Literal["passed", "failed"]
    failure_reasons: tuple[str, ...]
    included_in_aggregate: bool
    exclusion_reason: str | None
    trajectory_file: str
    trajectory_sha256: str
    config_digest: str
    metrics: EpisodeMetrics


@dataclass(frozen=True)
class ControllerAggregate:
    controller: Literal["p", "pd"]
    episode_count: int
    included_episode_count: int
    failed_episode_count: int
    excluded_episode_count: int
    mean_integrated_absolute_error_rad_s: float
    mean_normalized_maximum_overshoot: float
    mean_measured_steady_state_absolute_error_rad: float | None
    measured_steady_state_error_count: int
    unavailable_steady_state_error_count: int
    mean_squared_control_effort_nm2_s: float
    mean_saturation_fraction: float
    mean_measured_rise_time_s: float | None
    measured_rise_time_count: int
    undefined_rise_time_count: int
    mean_measured_settling_time_s: float | None
    measured_settling_time_count: int
    unsettled_episode_count: int


@dataclass(frozen=True)
class EvaluationResult:
    schema_version: int
    provenance: EvaluationProvenance
    episodes: tuple[EvaluatedEpisode, ...]
    aggregates: tuple[ControllerAggregate, ...]


def run_evaluation(
    *,
    config_path: Path,
    output_dir: Path,
    code_version: str,
    command: str,
    working_tree_dirty: bool = False,
) -> EvaluationResult:
    """Execute, validate, measure, and persist one complete evaluation plan."""
    if not command.strip():
        raise ValueError("command must not be empty")
    resolved_config_path = config_path.resolve()
    config = load_evaluation_config(resolved_config_path)
    resolved_output_dir = output_dir.resolve()
    if resolved_output_dir.exists() and not resolved_output_dir.is_dir():
        raise ValueError("output_dir must be a directory")
    trajectory_dir = resolved_output_dir / "trajectories"
    if trajectory_dir.exists() and not trajectory_dir.is_dir():
        raise ValueError("output trajectory path must be a directory")
    trajectory_dir.mkdir(parents=True, exist_ok=True)

    evaluated_episodes = tuple(
        _evaluate_plan_case(
            case,
            config=config,
            trajectory_dir=trajectory_dir,
            case_index=case_index,
            code_version=code_version,
        )
        for case_index, case in enumerate(build_evaluation_plan(config))
    )
    result = EvaluationResult(
        schema_version=EVALUATION_SCHEMA_VERSION,
        provenance=EvaluationProvenance(
            code_version=code_version,
            working_tree_dirty=working_tree_dirty,
            config_file=resolved_config_path.name,
            config_sha256=hashlib.sha256(
                resolved_config_path.read_bytes()
            ).hexdigest(),
            command=command,
            python_version=platform.python_version(),
            platform=platform.platform(),
            mujoco_version=version("mujoco"),
        ),
        episodes=evaluated_episodes,
        aggregates=tuple(
            _aggregate_controller(evaluated_episodes, controller)
            for controller in ("p", "pd")
        ),
    )
    with (resolved_output_dir / "evaluation.json").open(
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            _jsonable(result),
            output_file,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        output_file.write("\n")
    return result


def _evaluate_plan_case(
    case: EvaluationPlanCase,
    *,
    config: EvaluationConfig,
    trajectory_dir: Path,
    case_index: int,
    code_version: str,
) -> EvaluatedEpisode:
    experiment_config = _experiment_config(config, case)
    result = run_episode(
        experiment_config,
        case.controller,
        terminate_on_joint_limit=(
            config.failure_rules.fail_on_joint_limit_violation
        ),
        terminate_on_non_finite_state=(
            config.failure_rules.fail_on_non_finite_state
        ),
    )
    episode_id = f"{case.controller}-{case.scenario.id}-seed-{case.seed}"
    trajectory_relative_path = Path("trajectories") / f"{case_index:04d}.json"
    trajectory_path = trajectory_dir / trajectory_relative_path.name
    saved = save_trajectory(
        trajectory_path,
        episode_id=episode_id,
        seed=case.seed,
        code_version=code_version,
        result=result,
        config=experiment_config,
    )
    validated = load_trajectory(trajectory_path)
    if validated != saved:
        raise ValueError("persisted trajectory does not match validated record")
    metrics = evaluate_episode(
        _episode_result(validated),
        control_interval_s=config.episode.control_interval_s,
        thresholds=config.metrics,
        torque_limit_nm=config.episode.torque_limit_nm,
    )
    failure_reasons: list[str] = []
    if result.terminated:
        failure_reasons.append(result.termination_reason)
    if (
        config.failure_rules.fail_if_not_settled
        and metrics.settling_time.outcome is not TimingOutcome.MEASURED
    ):
        failure_reasons.append("not_settled")
    return EvaluatedEpisode(
        episode_id=episode_id,
        controller=case.controller,
        scenario_id=case.scenario.id,
        seed=case.seed,
        status="failed" if failure_reasons else "passed",
        failure_reasons=tuple(failure_reasons),
        included_in_aggregate=True,
        exclusion_reason=None,
        trajectory_file=trajectory_relative_path.as_posix(),
        trajectory_sha256=hashlib.sha256(trajectory_path.read_bytes()).hexdigest(),
        config_digest=validated.config_digest,
        metrics=metrics,
    )


def _episode_result(record: TrajectoryRecord) -> EpisodeResult:
    """Reconstruct the metric input from the validated durable record."""
    final_transition = record.transitions[-1]
    final_state = final_transition.next_state
    return EpisodeResult(
        controller=cast(Literal["p", "pd"], record.controller.name),
        samples=tuple(
            TrajectorySample(
                step_index=transition.step_index,
                time_s=transition.time_s,
                position_rad=transition.state.position_rad,
                velocity_rad_s=transition.state.velocity_rad_s,
                target_position_rad=transition.target.position_rad,
                target_velocity_rad_s=transition.target.velocity_rad_s,
                raw_torque_nm=transition.raw_torque_nm,
                applied_torque_nm=transition.applied_torque_nm,
                saturated=transition.saturated,
            )
            for transition in record.transitions
        ),
        final_time_s=(
            final_transition.time_s + record.configuration.control_interval_s
        ),
        final_position_rad=final_state.position_rad,
        final_velocity_rad_s=final_state.velocity_rad_s,
        terminated=final_transition.terminated,
        truncated=final_transition.truncated,
        termination_reason=cast(str, final_transition.termination_reason),
        non_finite_state_fields=final_transition.non_finite_state_fields,
    )


def _experiment_config(
    config: EvaluationConfig,
    case: EvaluationPlanCase,
) -> ExperimentConfig:
    return ExperimentConfig(
        model=config.model,
        episode=EpisodeConfig(
            duration_s=config.episode.duration_s,
            control_interval_s=config.episode.control_interval_s,
            start_position_rad=case.scenario.start_position_rad,
            initial_velocity_rad_s=case.scenario.initial_velocity_rad_s,
            target_position_rad=case.scenario.target_position_rad,
            target_velocity_rad_s=config.episode.target_velocity_rad_s,
            torque_limit_nm=config.episode.torque_limit_nm,
        ),
        p=config.p,
        pd=config.pd,
    )


def _aggregate_controller(
    episodes: tuple[EvaluatedEpisode, ...],
    controller: Literal["p", "pd"],
) -> ControllerAggregate:
    controller_episodes = tuple(
        episode for episode in episodes if episode.controller == controller
    )
    included = tuple(
        episode for episode in controller_episodes if episode.included_in_aggregate
    )
    rise_times = tuple(
        episode.metrics.rise_time.time_s
        for episode in included
        if episode.metrics.rise_time.time_s is not None
    )
    settling_times = tuple(
        episode.metrics.settling_time.time_s
        for episode in included
        if episode.metrics.settling_time.time_s is not None
    )
    steady_state_errors = tuple(
        episode.metrics.steady_state_absolute_error_rad
        for episode in included
        if episode.metrics.steady_state_absolute_error_rad is not None
    )
    return ControllerAggregate(
        controller=controller,
        episode_count=len(controller_episodes),
        included_episode_count=len(included),
        failed_episode_count=sum(
            episode.status == "failed" for episode in controller_episodes
        ),
        excluded_episode_count=len(controller_episodes) - len(included),
        mean_integrated_absolute_error_rad_s=_mean(
            tuple(
                episode.metrics.integrated_absolute_error_rad_s
                for episode in included
            )
        ),
        mean_normalized_maximum_overshoot=_mean(
            tuple(
                episode.metrics.normalized_maximum_overshoot
                for episode in included
            )
        ),
        mean_measured_steady_state_absolute_error_rad=(
            _mean(steady_state_errors) if steady_state_errors else None
        ),
        measured_steady_state_error_count=len(steady_state_errors),
        unavailable_steady_state_error_count=(
            len(included) - len(steady_state_errors)
        ),
        mean_squared_control_effort_nm2_s=_mean(
            tuple(
                episode.metrics.squared_control_effort_nm2_s
                for episode in included
            )
        ),
        mean_saturation_fraction=_mean(
            tuple(episode.metrics.saturation_fraction for episode in included)
        ),
        mean_measured_rise_time_s=_mean(rise_times) if rise_times else None,
        measured_rise_time_count=len(rise_times),
        undefined_rise_time_count=len(included) - len(rise_times),
        mean_measured_settling_time_s=(
            _mean(settling_times) if settling_times else None
        ),
        measured_settling_time_count=len(settling_times),
        unsettled_episode_count=sum(
            episode.metrics.settling_time.outcome is TimingOutcome.NOT_SETTLED
            for episode in included
        ),
    )


def _mean(values: tuple[float, ...]) -> float:
    if not values:
        raise ValueError("cannot aggregate an empty episode set")
    return math.fsum(values) / len(values)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _jsonable(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    return value
