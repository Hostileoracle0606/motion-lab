"""Validated saved-record inputs for Experiment 01 reporting."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Literal

from feedback_control.config import EvaluationConfig, load_evaluation_config
from feedback_control.trajectory import TrajectoryRecord, load_trajectory


ControllerName = Literal["p", "pd"]


@dataclass(frozen=True)
class EpisodeEvidence:
    episode_id: str
    controller: ControllerName
    scenario_id: str
    seed: int
    status: str
    failure_reasons: tuple[str, ...]
    included_in_aggregate: bool
    exclusion_reason: str | None
    trajectory_file: str
    trajectory_sha256: str
    config_digest: str
    metrics: dict[str, Any]
    trajectory: TrajectoryRecord


@dataclass(frozen=True)
class EvaluationEvidence:
    input_dir: Path
    evaluation_sha256: str
    provenance: dict[str, Any]
    config: EvaluationConfig
    episodes: tuple[EpisodeEvidence, ...]


def load_evaluation_evidence(
    input_dir: Path,
    config_path: Path,
) -> EvaluationEvidence:
    """Load and cross-check a summary, configuration, and source trajectories."""
    resolved_input = input_dir.resolve()
    evaluation_path = resolved_input / "evaluation.json"
    payload = _load_json_object(evaluation_path)
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported evaluation schema version")
    provenance = _require_object(payload, "provenance")
    _validate_provenance(provenance)

    resolved_config = config_path.resolve()
    actual_config_sha256 = hashlib.sha256(resolved_config.read_bytes()).hexdigest()
    if actual_config_sha256 != provenance["config_sha256"]:
        raise ValueError("evaluation configuration checksum mismatch")
    config = load_evaluation_config(resolved_config)

    episodes_payload = payload.get("episodes")
    if not isinstance(episodes_payload, list) or not episodes_payload:
        raise ValueError("episodes must be a non-empty array")
    episodes = tuple(
        _load_episode(resolved_input, episode_payload, index)
        for index, episode_payload in enumerate(episodes_payload)
    )
    _validate_shared_plan(episodes, provenance)
    return EvaluationEvidence(
        input_dir=resolved_input,
        evaluation_sha256=hashlib.sha256(evaluation_path.read_bytes()).hexdigest(),
        provenance=provenance,
        config=config,
        episodes=episodes,
    )


def _validate_shared_plan(
    episodes: tuple[EpisodeEvidence, ...],
    provenance: dict[str, Any],
) -> None:
    if len({episode.episode_id for episode in episodes}) != len(episodes):
        raise ValueError("episode identifiers must be unique")
    if any(
        episode.trajectory.code_version != provenance["code_version"]
        for episode in episodes
    ):
        raise ValueError("trajectory code versions do not match evaluation provenance")
    plan_by_controller = {
        controller: tuple(
            (episode.scenario_id, episode.seed)
            for episode in episodes
            if episode.controller == controller
        )
        for controller in ("p", "pd")
    }
    if not plan_by_controller["p"] or (
        plan_by_controller["p"] != plan_by_controller["pd"]
    ):
        raise ValueError("P and PD evidence must contain the same ordered cases")
    p_episodes = tuple(
        episode for episode in episodes if episode.controller == "p"
    )
    pd_episodes = tuple(
        episode for episode in episodes if episode.controller == "pd"
    )
    if any(
        p_episode.config_digest != pd_episode.config_digest
        for p_episode, pd_episode in zip(p_episodes, pd_episodes, strict=True)
    ):
        raise ValueError("paired P and PD cases must share one configuration")
    for controller in ("p", "pd"):
        gains = {
            (episode.trajectory.controller.kp, episode.trajectory.controller.kd)
            for episode in episodes
            if episode.controller == controller
        }
        if len(gains) != 1:
            raise ValueError(f"{controller.upper()} controller gains must be shared")


def _load_episode(
    input_dir: Path,
    payload: Any,
    index: int,
) -> EpisodeEvidence:
    if not isinstance(payload, dict):
        raise ValueError(f"episodes[{index}] must be an object")
    episode = payload
    episode_id = _require_string(episode, "episode_id", index)
    controller = _require_string(episode, "controller", index)
    if controller not in ("p", "pd"):
        raise ValueError(f"episodes[{index}].controller is invalid")
    scenario_id = _require_string(episode, "scenario_id", index)
    seed = episode.get("seed")
    if type(seed) is not int or seed < 0:
        raise ValueError(f"episodes[{index}].seed must be non-negative")
    trajectory_file = _require_string(episode, "trajectory_file", index)
    relative_trajectory = Path(trajectory_file)
    if relative_trajectory.is_absolute() or ".." in relative_trajectory.parts:
        raise ValueError(f"episodes[{index}].trajectory_file must stay in input_dir")
    trajectory_path = (input_dir / relative_trajectory).resolve()
    if not trajectory_path.is_relative_to(input_dir):
        raise ValueError(f"episodes[{index}].trajectory_file must stay in input_dir")
    expected_sha256 = _require_sha256(episode, "trajectory_sha256", index)
    actual_sha256 = hashlib.sha256(trajectory_path.read_bytes()).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(f"episodes[{index}] trajectory checksum mismatch")
    trajectory = load_trajectory(trajectory_path)
    config_digest = _require_sha256(episode, "config_digest", index)
    if (
        trajectory.episode_id != episode_id
        or trajectory.controller.name != controller
        or trajectory.seed != seed
        or trajectory.config_digest != config_digest
    ):
        raise ValueError(f"episodes[{index}] does not match its trajectory")

    status = _require_string(episode, "status", index)
    if status not in ("passed", "failed"):
        raise ValueError(f"episodes[{index}].status is invalid")
    failure_reasons_value = episode.get("failure_reasons")
    if not isinstance(failure_reasons_value, list) or any(
        not isinstance(reason, str) or not reason
        for reason in failure_reasons_value
    ):
        raise ValueError(f"episodes[{index}].failure_reasons is invalid")
    if len(set(failure_reasons_value)) != len(failure_reasons_value):
        raise ValueError(f"episodes[{index}].failure_reasons must be unique")
    included = episode.get("included_in_aggregate")
    if type(included) is not bool:
        raise ValueError(f"episodes[{index}].included_in_aggregate must be boolean")
    exclusion_reason = episode.get("exclusion_reason")
    if exclusion_reason is not None and (
        not isinstance(exclusion_reason, str) or not exclusion_reason
    ):
        raise ValueError(f"episodes[{index}].exclusion_reason is invalid")
    if (status == "failed") != bool(failure_reasons_value):
        raise ValueError(f"episodes[{index}] status does not match failure reasons")
    if included == (exclusion_reason is not None):
        raise ValueError(f"episodes[{index}] aggregate inclusion is inconsistent")
    metrics = _require_object(episode, "metrics")
    _validate_metrics(metrics, index)
    return EpisodeEvidence(
        episode_id=episode_id,
        controller=controller,
        scenario_id=scenario_id,
        seed=seed,
        status=status,
        failure_reasons=tuple(failure_reasons_value),
        included_in_aggregate=included,
        exclusion_reason=exclusion_reason,
        trajectory_file=relative_trajectory.as_posix(),
        trajectory_sha256=actual_sha256,
        config_digest=config_digest,
        metrics=metrics,
        trajectory=trajectory,
    )


def _validate_metrics(metrics: dict[str, Any], index: int) -> None:
    for name in (
        "integrated_absolute_error_rad_s",
        "normalized_maximum_overshoot",
        "squared_control_effort_nm2_s",
        "saturation_fraction",
    ):
        _require_finite(metrics.get(name), f"episodes[{index}].metrics.{name}")
        if metrics[name] < 0:
            raise ValueError(f"episodes[{index}].metrics.{name} must be non-negative")
    if metrics["saturation_fraction"] > 1:
        raise ValueError(
            f"episodes[{index}].metrics.saturation_fraction must not exceed one"
        )
    steady_state_error = metrics.get("steady_state_absolute_error_rad")
    if steady_state_error is not None:
        _require_finite(
            steady_state_error,
            f"episodes[{index}].metrics.steady_state_absolute_error_rad",
        )
    availability = metrics.get("steady_state_error_availability")
    if availability not in ("measured", "insufficient_duration"):
        raise ValueError(
            f"episodes[{index}].metrics.steady_state_error_availability is invalid"
        )
    if (availability == "measured") != (steady_state_error is not None):
        raise ValueError(
            f"episodes[{index}] steady-state availability is inconsistent"
        )
    for name in ("rise_time", "settling_time"):
        timing = _require_object(metrics, name)
        outcome = timing.get("outcome")
        if not isinstance(outcome, str) or not outcome:
            raise ValueError(f"episodes[{index}].metrics.{name}.outcome is invalid")
        time_s = timing.get("time_s")
        if time_s is not None:
            _require_finite(time_s, f"episodes[{index}].metrics.{name}.time_s")
            if time_s < 0:
                raise ValueError(
                    f"episodes[{index}].metrics.{name}.time_s must be non-negative"
                )
        if (outcome == "measured") != (time_s is not None):
            raise ValueError(
                f"episodes[{index}].metrics.{name} outcome is inconsistent"
            )


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as input_file:
            value = json.load(input_file)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load evaluation summary: {path}") from error
    if not isinstance(value, dict):
        raise ValueError("evaluation summary must be an object")
    return value


def _require_object(payload: dict[str, Any], field_name: str) -> dict[str, Any]:
    value = payload.get(field_name)
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object")
    return value


def _validate_provenance(provenance: dict[str, Any]) -> None:
    code_version = provenance.get("code_version")
    if (
        not isinstance(code_version, str)
        or len(code_version) not in (40, 64)
        or any(character not in "0123456789abcdef" for character in code_version)
    ):
        raise ValueError("provenance.code_version must be a Git object ID")
    if type(provenance.get("working_tree_dirty")) is not bool:
        raise ValueError("provenance.working_tree_dirty must be boolean")
    config_sha256 = provenance.get("config_sha256")
    if (
        not isinstance(config_sha256, str)
        or len(config_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in config_sha256
        )
    ):
        raise ValueError("provenance.config_sha256 must be a SHA-256 digest")
    for field_name in (
        "config_file",
        "command",
        "python_version",
        "platform",
        "mujoco_version",
    ):
        value = provenance.get(field_name)
        if not isinstance(value, str) or not value:
            raise ValueError(f"provenance.{field_name} must not be empty")


def _require_string(payload: dict[str, Any], field_name: str, index: int) -> str:
    value = payload.get(field_name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"episodes[{index}].{field_name} must not be empty")
    return value


def _require_sha256(payload: dict[str, Any], field_name: str, index: int) -> str:
    value = _require_string(payload, field_name, index)
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"episodes[{index}].{field_name} must be a SHA-256 digest")
    return value


def _require_finite(value: Any, field_name: str) -> None:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be finite")
    try:
        finite = math.isfinite(value)
    except TypeError as error:
        raise ValueError(f"{field_name} must be finite") from error
    if not finite:
        raise ValueError(f"{field_name} must be finite")
