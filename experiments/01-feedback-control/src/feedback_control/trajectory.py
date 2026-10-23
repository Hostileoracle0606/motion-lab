"""Durable trajectory records for Experiment 01 control episodes."""

from dataclasses import asdict, dataclass
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
from typing import Any

import mujoco

from feedback_control.config import ExperimentConfig
from feedback_control.simulation import EpisodeResult


TRAJECTORY_SCHEMA_VERSION = 4
ENVIRONMENT_ID = "motion-lab/experiment-01/one-link"


@dataclass(frozen=True)
class StateRecord:
    """One scalar joint observation in radians and radians per second."""

    position_rad: float
    velocity_rad_s: float


@dataclass(frozen=True)
class ControllerMetadata:
    """Controller policy and gains used to collect an episode."""

    name: str
    kp: float
    kd: float


@dataclass(frozen=True)
class ConfigurationProvenance:
    """Experiment-defining values needed to identify a collected episode."""

    model_file: str
    model_sha256: str
    model_keyframe: str
    simulation_timestep_s: float
    episode_duration_s: float
    control_interval_s: float
    position_lower_limit_rad: float
    position_upper_limit_rad: float
    start_position_rad: float
    initial_velocity_rad_s: float
    target_position_rad: float
    target_velocity_rad_s: float
    torque_limit_nm: float
    p_kp: float
    pd_kp: float
    pd_kd: float


@dataclass(frozen=True)
class TransitionRecord:
    """One observation-before-action transition."""

    step_index: int
    time_s: float
    state: StateRecord
    target: StateRecord
    raw_torque_nm: float
    applied_torque_nm: float
    saturated: bool
    next_state: StateRecord | None
    non_finite_state_fields: tuple[str, ...]
    terminated: bool
    truncated: bool
    termination_reason: str | None


@dataclass(frozen=True)
class TrajectoryRecord:
    """A complete, independently loadable Experiment 01 episode."""

    schema_version: int
    episode_id: str
    seed: int
    code_version: str
    environment_id: str
    environment_version: str
    mujoco_version: str
    config_digest: str
    configuration: ConfigurationProvenance
    controller: ControllerMetadata
    transitions: tuple[TransitionRecord, ...]


def save_trajectory(
    path: Path,
    *,
    episode_id: str,
    seed: int,
    code_version: str,
    result: EpisodeResult,
    config: ExperimentConfig,
) -> TrajectoryRecord:
    """Save one fixed-duration episode as versioned, machine-readable JSON."""
    record = _build_trajectory_record(
        episode_id=episode_id,
        seed=seed,
        code_version=code_version,
        result=result,
        config=config,
    )
    validate_trajectory(record)
    resolved_path = path.resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    with resolved_path.open("w", encoding="utf-8") as output_file:
        json.dump(asdict(record), output_file, indent=2, allow_nan=False)
        output_file.write("\n")
    return record


def load_trajectory(path: Path) -> TrajectoryRecord:
    """Load one Experiment 01 trajectory JSON file."""
    with path.resolve().open(encoding="utf-8") as input_file:
        payload = json.load(input_file)
    record = _record_from_mapping(payload)
    validate_trajectory(record)
    return record


def validate_trajectory(record: TrajectoryRecord) -> None:
    """Reject trajectory values that cannot be trusted as experiment evidence."""
    if (
        type(record.schema_version) is not int
        or record.schema_version != TRAJECTORY_SCHEMA_VERSION
    ):
        raise ValueError(
            f"unsupported trajectory schema version: {record.schema_version}"
        )
    if not isinstance(record.episode_id, str) or not record.episode_id.strip():
        raise ValueError("episode_id must not be empty")
    if type(record.seed) is not int or record.seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if not _is_git_object_id(record.code_version):
        raise ValueError(
            "code_version must be a full lowercase Git commit object ID"
        )
    if record.environment_id != ENVIRONMENT_ID:
        raise ValueError(f"unsupported environment_id: {record.environment_id!r}")
    if not isinstance(record.mujoco_version, str) or not record.mujoco_version:
        raise ValueError("mujoco_version must not be empty")
    _validate_configuration(record.configuration)
    if not _is_sha256(record.config_digest):
        raise ValueError("config_digest must be a lowercase SHA-256 digest")
    if record.config_digest != _configuration_digest(record.configuration):
        raise ValueError("config_digest does not match configuration")
    if record.environment_version != record.configuration.model_sha256:
        raise ValueError("environment_version does not match model_sha256")
    _require_finite_number("controller.kp", record.controller.kp)
    _require_finite_number("controller.kd", record.controller.kd)
    if record.controller.name == "p":
        expected_gains = (record.configuration.p_kp, 0.0)
    elif record.controller.name == "pd":
        expected_gains = (
            record.configuration.pd_kp,
            record.configuration.pd_kd,
        )
    else:
        raise ValueError("controller metadata does not match configuration")
    if not (
        math.isclose(
            record.controller.kp,
            expected_gains[0],
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and math.isclose(
            record.controller.kd,
            expected_gains[1],
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("controller metadata does not match configuration")
    expected_transition_count = round(
        record.configuration.episode_duration_s
        / record.configuration.control_interval_s
    )
    final_transition = record.transitions[-1] if record.transitions else None
    transition_count_is_invalid = (
        len(record.transitions) > expected_transition_count
        or (
            len(record.transitions) < expected_transition_count
            and (
                final_transition is None
                or not final_transition.terminated
            )
        )
    )
    if transition_count_is_invalid:
        raise ValueError(
            "transition count does not match configured episode duration"
        )
    boundary_count = sum(
        int(transition.terminated) + int(transition.truncated)
        for transition in record.transitions
    )
    if (
        boundary_count != 1
        or not record.transitions
        or not (
            record.transitions[-1].terminated or record.transitions[-1].truncated
        )
    ):
        raise ValueError("trajectory must have exactly one final terminal boundary")
    expected_target = StateRecord(
        position_rad=record.configuration.target_position_rad,
        velocity_rad_s=record.configuration.target_velocity_rad_s,
    )
    for index, transition in enumerate(record.transitions):
        if type(transition.step_index) is not int or transition.step_index != index:
            raise ValueError("step indices must be contiguous from zero")
        for field_name, value in (
            ("saturated", transition.saturated),
            ("terminated", transition.terminated),
            ("truncated", transition.truncated),
        ):
            if type(value) is not bool:
                raise ValueError(f"transitions[{index}].{field_name} must be boolean")
        if index > 0 and not _states_match(
            record.transitions[index - 1].next_state,
            transition.state,
        ):
            raise ValueError(
                f"next_state at transition {index - 1} does not match "
                f"state at transition {index}"
            )
        if not _states_match(transition.target, expected_target):
            raise ValueError("transition target does not match configuration")
        expected_time = index * record.configuration.control_interval_s
        if not math.isclose(
            transition.time_s,
            expected_time,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "times must follow the configured control schedule from zero"
            )
        is_final = index == len(record.transitions) - 1
        is_boundary = transition.terminated or transition.truncated
        if is_boundary:
            if (
                not isinstance(transition.termination_reason, str)
                or not transition.termination_reason
            ):
                raise ValueError(
                    f"transitions[{index}].termination_reason must describe "
                    "the terminal boundary"
                )
        elif transition.termination_reason is not None:
            raise ValueError(
                f"transitions[{index}].termination_reason must be null before "
                "the terminal boundary"
            )
        if not isinstance(transition.non_finite_state_fields, tuple) or any(
            field_name not in {"position_rad", "velocity_rad_s"}
            for field_name in transition.non_finite_state_fields
        ):
            raise ValueError(
                f"transitions[{index}].non_finite_state_fields is invalid"
            )
        if len(set(transition.non_finite_state_fields)) != len(
            transition.non_finite_state_fields
        ):
            raise ValueError(
                f"transitions[{index}].non_finite_state_fields must be unique"
            )
        is_non_finite_boundary = (
            is_final
            and transition.terminated
            and not transition.truncated
            and transition.termination_reason == "non_finite_state"
        )
        if is_non_finite_boundary:
            if transition.next_state is not None:
                raise ValueError(
                    "non-finite terminal transition must not contain next_state"
                )
            if not transition.non_finite_state_fields:
                raise ValueError(
                    "non-finite terminal transition must identify invalid fields"
                )
        elif transition.next_state is None:
            raise ValueError(
                f"transitions[{index}].next_state must be an object"
            )
        elif transition.non_finite_state_fields:
            raise ValueError(
                "non_finite_state_fields requires a non-finite terminal boundary"
            )

        numeric_values = {
            "time_s": transition.time_s,
            "state.position_rad": transition.state.position_rad,
            "state.velocity_rad_s": transition.state.velocity_rad_s,
            "target.position_rad": transition.target.position_rad,
            "target.velocity_rad_s": transition.target.velocity_rad_s,
            "raw_torque_nm": transition.raw_torque_nm,
            "applied_torque_nm": transition.applied_torque_nm,
        }
        if transition.next_state is not None:
            numeric_values.update(
                {
                    "next_state.position_rad": transition.next_state.position_rad,
                    "next_state.velocity_rad_s": transition.next_state.velocity_rad_s,
                }
            )
        for field_name, value in numeric_values.items():
            _require_finite_number(f"transitions[{index}].{field_name}", value)
        if not (
            record.configuration.position_lower_limit_rad
            <= transition.state.position_rad
            <= record.configuration.position_upper_limit_rad
        ):
            raise ValueError(
                f"transitions[{index}].state.position_rad is outside "
                "the model position limits"
            )
        if transition.next_state is not None:
            next_state_is_in_range = (
                record.configuration.position_lower_limit_rad
                <= transition.next_state.position_rad
                <= record.configuration.position_upper_limit_rad
            )
            is_joint_limit_boundary = (
                is_final
                and transition.terminated
                and not transition.truncated
                and transition.termination_reason == "joint_limit_violation"
            )
            if not next_state_is_in_range and not is_joint_limit_boundary:
                raise ValueError(
                    f"transitions[{index}].next_state.position_rad is outside "
                    "the model position limits"
                )
            if is_joint_limit_boundary and next_state_is_in_range:
                raise ValueError(
                    "joint-limit terminal transition must preserve the violating "
                    "next_state"
                )
        if (
            abs(transition.applied_torque_nm)
            > record.configuration.torque_limit_nm
        ):
            raise ValueError(
                f"transitions[{index}].applied_torque_nm exceeds torque limit"
            )
        expected_applied_torque = max(
            -record.configuration.torque_limit_nm,
            min(
                record.configuration.torque_limit_nm,
                transition.raw_torque_nm,
            ),
        )
        if transition.applied_torque_nm != expected_applied_torque:
            raise ValueError(
                f"transitions[{index}].applied_torque_nm does not match "
                "clipped raw torque"
            )
        expected_saturation = transition.raw_torque_nm != expected_applied_torque
        if transition.saturated is not expected_saturation:
            raise ValueError(
                f"transitions[{index}].saturated does not match torque clipping"
            )

    first = record.transitions[0]
    if not math.isclose(
        first.state.position_rad,
        record.configuration.start_position_rad,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("initial state does not match configuration")
    if not math.isclose(
        first.state.velocity_rad_s,
        record.configuration.initial_velocity_rad_s,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("initial state does not match configuration")


def _validate_configuration(configuration: ConfigurationProvenance) -> None:
    for field_name, value in (
        ("model_file", configuration.model_file),
        ("model_keyframe", configuration.model_keyframe),
    ):
        if not isinstance(value, str) or not value:
            raise ValueError(f"configuration.{field_name} must not be empty")
    if not _is_sha256(configuration.model_sha256):
        raise ValueError(
            "configuration.model_sha256 must be a lowercase SHA-256 digest"
        )

    numeric_values = {
        "simulation_timestep_s": configuration.simulation_timestep_s,
        "episode_duration_s": configuration.episode_duration_s,
        "control_interval_s": configuration.control_interval_s,
        "position_lower_limit_rad": configuration.position_lower_limit_rad,
        "position_upper_limit_rad": configuration.position_upper_limit_rad,
        "start_position_rad": configuration.start_position_rad,
        "initial_velocity_rad_s": configuration.initial_velocity_rad_s,
        "target_position_rad": configuration.target_position_rad,
        "target_velocity_rad_s": configuration.target_velocity_rad_s,
        "torque_limit_nm": configuration.torque_limit_nm,
        "p_kp": configuration.p_kp,
        "pd_kp": configuration.pd_kp,
        "pd_kd": configuration.pd_kd,
    }
    for field_name, value in numeric_values.items():
        _require_finite_number(f"configuration.{field_name}", value)

    for field_name in (
        "simulation_timestep_s",
        "episode_duration_s",
        "control_interval_s",
        "torque_limit_nm",
    ):
        if numeric_values[field_name] <= 0:
            raise ValueError(f"configuration.{field_name} must be greater than zero")
    if (
        configuration.position_lower_limit_rad
        >= configuration.position_upper_limit_rad
    ):
        raise ValueError(
            "configuration position lower limit must be less than upper limit"
        )
    for field_name in ("start_position_rad", "target_position_rad"):
        value = numeric_values[field_name]
        if not (
            configuration.position_lower_limit_rad
            <= value
            <= configuration.position_upper_limit_rad
        ):
            raise ValueError(
                f"configuration.{field_name} is outside the model position limits"
            )
    for field_name in ("p_kp", "pd_kp", "pd_kd"):
        if numeric_values[field_name] < 0:
            raise ValueError(f"configuration.{field_name} must be non-negative")

    for ratio, description in (
        (
            configuration.control_interval_s
            / configuration.simulation_timestep_s,
            "control interval / simulation timestep",
        ),
        (
            configuration.episode_duration_s / configuration.control_interval_s,
            "duration / control interval",
        ),
    ):
        if not math.isclose(ratio, round(ratio), rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(f"configuration {description} must be an integer")


def _require_finite_number(field_name: str, value: Any) -> None:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be finite")
    try:
        finite = math.isfinite(value)
    except TypeError as error:
        raise ValueError(f"{field_name} must be finite") from error
    if not finite:
        raise ValueError(f"{field_name} must be finite")


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_git_object_id(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) in (40, 64)
        and all(character in "0123456789abcdef" for character in value)
    )


def _states_match(left: StateRecord, right: StateRecord) -> bool:
    return left == right


def _build_trajectory_record(
    *,
    episode_id: str,
    seed: int,
    code_version: str,
    result: EpisodeResult,
    config: ExperimentConfig,
) -> TrajectoryRecord:
    configuration = _configuration_provenance(config)
    controller_gains = config.p if result.controller == "p" else config.pd
    transitions: list[TransitionRecord] = []
    for index, sample in enumerate(result.samples):
        if index + 1 < len(result.samples):
            following = result.samples[index + 1]
            next_state = StateRecord(
                position_rad=following.position_rad,
                velocity_rad_s=following.velocity_rad_s,
            )
        elif result.final_position_rad is None or result.final_velocity_rad_s is None:
            next_state = None
        else:
            next_state = StateRecord(
                position_rad=result.final_position_rad,
                velocity_rad_s=result.final_velocity_rad_s,
            )
        is_last = index == len(result.samples) - 1
        transitions.append(
            TransitionRecord(
                step_index=sample.step_index,
                time_s=sample.time_s,
                state=StateRecord(
                    position_rad=sample.position_rad,
                    velocity_rad_s=sample.velocity_rad_s,
                ),
                target=StateRecord(
                    position_rad=sample.target_position_rad,
                    velocity_rad_s=sample.target_velocity_rad_s,
                ),
                raw_torque_nm=sample.raw_torque_nm,
                applied_torque_nm=sample.applied_torque_nm,
                saturated=sample.saturated,
                next_state=next_state,
                non_finite_state_fields=(
                    result.non_finite_state_fields if is_last else ()
                ),
                terminated=is_last and result.terminated,
                truncated=is_last and result.truncated,
                termination_reason=result.termination_reason if is_last else None,
            )
        )
    return TrajectoryRecord(
        schema_version=TRAJECTORY_SCHEMA_VERSION,
        episode_id=episode_id,
        seed=seed,
        code_version=code_version,
        environment_id=ENVIRONMENT_ID,
        environment_version=configuration.model_sha256,
        mujoco_version=version("mujoco"),
        config_digest=_configuration_digest(configuration),
        configuration=configuration,
        controller=ControllerMetadata(
            name=result.controller,
            kp=controller_gains.kp,
            kd=controller_gains.kd,
        ),
        transitions=tuple(transitions),
    )


def _configuration_provenance(config: ExperimentConfig) -> ConfigurationProvenance:
    model = mujoco.MjModel.from_xml_path(str(config.model.path))
    position_lower_limit_rad, position_upper_limit_rad = model.jnt_range[0]
    return ConfigurationProvenance(
        model_file=config.model.path.name,
        model_sha256=hashlib.sha256(config.model.path.read_bytes()).hexdigest(),
        model_keyframe=config.model.keyframe,
        simulation_timestep_s=config.model.simulation_timestep_s,
        episode_duration_s=config.episode.duration_s,
        control_interval_s=config.episode.control_interval_s,
        position_lower_limit_rad=float(position_lower_limit_rad),
        position_upper_limit_rad=float(position_upper_limit_rad),
        start_position_rad=config.episode.start_position_rad,
        initial_velocity_rad_s=config.episode.initial_velocity_rad_s,
        target_position_rad=config.episode.target_position_rad,
        target_velocity_rad_s=config.episode.target_velocity_rad_s,
        torque_limit_nm=config.episode.torque_limit_nm,
        p_kp=config.p.kp,
        pd_kp=config.pd.kp,
        pd_kd=config.pd.kd,
    )


def _configuration_digest(configuration: ConfigurationProvenance) -> str:
    canonical = json.dumps(
        asdict(configuration),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _record_from_mapping(payload: Any) -> TrajectoryRecord:
    payload = _require_object(payload, "trajectory")
    _require_fields(
        payload,
        (
            "schema_version",
            "episode_id",
            "seed",
            "code_version",
            "environment_id",
            "environment_version",
            "mujoco_version",
            "config_digest",
            "configuration",
            "controller",
            "transitions",
        ),
    )
    configuration_payload = _require_object(
        payload["configuration"],
        "configuration",
    )
    _require_fields(
        configuration_payload,
        (
            "model_file",
            "model_sha256",
            "model_keyframe",
            "simulation_timestep_s",
            "episode_duration_s",
            "control_interval_s",
            "position_lower_limit_rad",
            "position_upper_limit_rad",
            "start_position_rad",
            "initial_velocity_rad_s",
            "target_position_rad",
            "target_velocity_rad_s",
            "torque_limit_nm",
            "p_kp",
            "pd_kp",
            "pd_kd",
        ),
        prefix="configuration.",
    )
    configuration = ConfigurationProvenance(
        model_file=configuration_payload["model_file"],
        model_sha256=configuration_payload["model_sha256"],
        model_keyframe=configuration_payload["model_keyframe"],
        simulation_timestep_s=configuration_payload["simulation_timestep_s"],
        episode_duration_s=configuration_payload["episode_duration_s"],
        control_interval_s=configuration_payload["control_interval_s"],
        position_lower_limit_rad=configuration_payload["position_lower_limit_rad"],
        position_upper_limit_rad=configuration_payload["position_upper_limit_rad"],
        start_position_rad=configuration_payload["start_position_rad"],
        initial_velocity_rad_s=configuration_payload["initial_velocity_rad_s"],
        target_position_rad=configuration_payload["target_position_rad"],
        target_velocity_rad_s=configuration_payload["target_velocity_rad_s"],
        torque_limit_nm=configuration_payload["torque_limit_nm"],
        p_kp=configuration_payload["p_kp"],
        pd_kp=configuration_payload["pd_kp"],
        pd_kd=configuration_payload["pd_kd"],
    )
    controller_payload = _require_object(payload["controller"], "controller")
    _require_fields(
        controller_payload,
        ("name", "kp", "kd"),
        prefix="controller.",
    )
    controller = ControllerMetadata(
        name=controller_payload["name"],
        kp=controller_payload["kp"],
        kd=controller_payload["kd"],
    )
    transitions_payload = payload["transitions"]
    if not isinstance(transitions_payload, list):
        raise ValueError("transitions must be an array")
    transitions: list[TransitionRecord] = []
    for index, transition_value in enumerate(transitions_payload):
        transition = _require_object(
            transition_value,
            f"transitions[{index}]",
        )
        _require_fields(
            transition,
            (
                "step_index",
                "time_s",
                "state",
                "target",
                "raw_torque_nm",
                "applied_torque_nm",
                "saturated",
                "next_state",
                "non_finite_state_fields",
                "terminated",
                "truncated",
                "termination_reason",
            ),
            prefix=f"transitions[{index}].",
        )
        transitions.append(
            TransitionRecord(
                step_index=transition["step_index"],
                time_s=transition["time_s"],
                state=_state_from_mapping(
                    transition["state"],
                    f"transitions[{index}].state",
                ),
                target=_state_from_mapping(
                    transition["target"],
                    f"transitions[{index}].target",
                ),
                raw_torque_nm=transition["raw_torque_nm"],
                applied_torque_nm=transition["applied_torque_nm"],
                saturated=transition["saturated"],
                next_state=(
                    None
                    if transition["next_state"] is None
                    else _state_from_mapping(
                        transition["next_state"],
                        f"transitions[{index}].next_state",
                    )
                ),
                non_finite_state_fields=_string_tuple_from_value(
                    transition["non_finite_state_fields"],
                    f"transitions[{index}].non_finite_state_fields",
                ),
                terminated=transition["terminated"],
                truncated=transition["truncated"],
                termination_reason=transition["termination_reason"],
            )
        )
    return TrajectoryRecord(
        schema_version=payload["schema_version"],
        episode_id=payload["episode_id"],
        seed=payload["seed"],
        code_version=payload["code_version"],
        environment_id=payload["environment_id"],
        environment_version=payload["environment_version"],
        mujoco_version=payload["mujoco_version"],
        config_digest=payload["config_digest"],
        configuration=configuration,
        controller=controller,
        transitions=tuple(transitions),
    )


def _state_from_mapping(payload: Any, field_path: str) -> StateRecord:
    payload = _require_object(payload, field_path)
    _require_fields(
        payload,
        ("position_rad", "velocity_rad_s"),
        prefix=f"{field_path}.",
    )
    return StateRecord(
        position_rad=payload["position_rad"],
        velocity_rad_s=payload["velocity_rad_s"],
    )


def _string_tuple_from_value(payload: Any, field_path: str) -> tuple[str, ...]:
    if not isinstance(payload, list) or not all(
        isinstance(value, str) for value in payload
    ):
        raise ValueError(f"{field_path} must be an array of strings")
    return tuple(payload)


def _require_object(payload: Any, field_path: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(f"{field_path} must be an object")
    return payload


def _require_fields(
    payload: dict[str, Any],
    field_names: tuple[str, ...],
    *,
    prefix: str = "",
) -> None:
    for field_name in field_names:
        if field_name not in payload:
            raise ValueError(f"missing required field: {prefix}{field_name}")
