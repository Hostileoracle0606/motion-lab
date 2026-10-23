"""Versioned configuration for the feedback-control experiment."""

from dataclasses import dataclass
import math
from pathlib import Path
import tomllib
from typing import Literal

import mujoco


@dataclass(frozen=True)
class ModelConfig:
    path: Path
    keyframe: str
    simulation_timestep_s: float


@dataclass(frozen=True)
class EpisodeConfig:
    duration_s: float
    control_interval_s: float
    start_position_rad: float
    target_position_rad: float
    target_velocity_rad_s: float
    torque_limit_nm: float


@dataclass(frozen=True)
class ControllerGains:
    kp: float
    kd: float = 0.0


@dataclass(frozen=True)
class ExperimentConfig:
    model: ModelConfig
    episode: EpisodeConfig
    p: ControllerGains
    pd: ControllerGains


@dataclass(frozen=True)
class EvaluationEpisodeConfig:
    duration_s: float
    control_interval_s: float
    target_velocity_rad_s: float
    torque_limit_nm: float


@dataclass(frozen=True)
class EvaluationScenario:
    id: str
    start_position_rad: float
    target_position_rad: float
    initial_velocity_rad_s: float


@dataclass(frozen=True)
class MetricThresholds:
    rise_lower_fraction: float
    rise_upper_fraction: float
    settling_position_tolerance_rad: float
    settling_velocity_tolerance_rad_s: float
    steady_state_window_s: float
    minimum_command_magnitude_rad: float


@dataclass(frozen=True)
class NoiseConfig:
    position_std_rad: float
    velocity_std_rad_s: float
    action_std_nm: float


@dataclass(frozen=True)
class FailureRules:
    fail_on_non_finite_state: bool
    fail_on_joint_limit_violation: bool
    fail_if_not_settled: bool


@dataclass(frozen=True)
class EvaluationConfig:
    schema_version: int
    seeds: tuple[int, ...]
    model: ModelConfig
    episode: EvaluationEpisodeConfig
    p: ControllerGains
    pd: ControllerGains
    metrics: MetricThresholds
    noise: NoiseConfig
    failure_rules: FailureRules
    scenarios: tuple[EvaluationScenario, ...]


EvaluationControllerName = Literal["p", "pd"]


@dataclass(frozen=True)
class EvaluationPlanCase:
    controller: EvaluationControllerName
    scenario: EvaluationScenario
    seed: int


def load_experiment_config(path: Path) -> ExperimentConfig:
    """Load a TOML configuration, resolving the model relative to that file."""
    config_path = path.resolve()
    with config_path.open("rb") as config_file:
        raw = tomllib.load(config_file)

    model_values = raw["model"]
    episode_values = raw["episode"]
    controller_values = raw["controllers"]

    config = ExperimentConfig(
        model=ModelConfig(
            path=(config_path.parent / model_values["path"]).resolve(),
            keyframe=str(model_values["keyframe"]),
            simulation_timestep_s=float(model_values["simulation_timestep_s"]),
        ),
        episode=EpisodeConfig(
            duration_s=float(episode_values["duration_s"]),
            control_interval_s=float(episode_values["control_interval_s"]),
            start_position_rad=float(episode_values["start_position_rad"]),
            target_position_rad=float(episode_values["target_position_rad"]),
            target_velocity_rad_s=float(episode_values["target_velocity_rad_s"]),
            torque_limit_nm=float(episode_values["torque_limit_nm"]),
        ),
        p=ControllerGains(kp=float(controller_values["p"]["kp"])),
        pd=ControllerGains(
            kp=float(controller_values["pd"]["kp"]),
            kd=float(controller_values["pd"]["kd"]),
        ),
    )
    validate_experiment_config(config)
    return config


def load_evaluation_config(path: Path) -> EvaluationConfig:
    """Load the versioned multi-scenario evaluation contract from TOML."""
    config_path = path.resolve()
    with config_path.open("rb") as config_file:
        raw = tomllib.load(config_file)

    model_values = raw["model"]
    episode_values = raw["episode"]
    controller_values = raw["controllers"]
    metric_values = raw["metrics"]
    noise_values = raw["noise"]
    failure_values = raw["failure_rules"]

    config = EvaluationConfig(
        schema_version=_require_integer("schema_version", raw["schema_version"]),
        seeds=tuple(_require_integer("seed", seed) for seed in raw["seeds"]),
        model=ModelConfig(
            path=(config_path.parent / model_values["path"]).resolve(),
            keyframe=str(model_values["keyframe"]),
            simulation_timestep_s=float(model_values["simulation_timestep_s"]),
        ),
        episode=EvaluationEpisodeConfig(
            duration_s=float(episode_values["duration_s"]),
            control_interval_s=float(episode_values["control_interval_s"]),
            target_velocity_rad_s=float(episode_values["target_velocity_rad_s"]),
            torque_limit_nm=float(episode_values["torque_limit_nm"]),
        ),
        p=ControllerGains(kp=float(controller_values["p"]["kp"])),
        pd=ControllerGains(
            kp=float(controller_values["pd"]["kp"]),
            kd=float(controller_values["pd"]["kd"]),
        ),
        metrics=MetricThresholds(
            rise_lower_fraction=float(metric_values["rise_lower_fraction"]),
            rise_upper_fraction=float(metric_values["rise_upper_fraction"]),
            settling_position_tolerance_rad=float(
                metric_values["settling_position_tolerance_rad"]
            ),
            settling_velocity_tolerance_rad_s=float(
                metric_values["settling_velocity_tolerance_rad_s"]
            ),
            steady_state_window_s=float(metric_values["steady_state_window_s"]),
            minimum_command_magnitude_rad=float(
                metric_values["minimum_command_magnitude_rad"]
            ),
        ),
        noise=NoiseConfig(
            position_std_rad=float(noise_values["position_std_rad"]),
            velocity_std_rad_s=float(noise_values["velocity_std_rad_s"]),
            action_std_nm=float(noise_values["action_std_nm"]),
        ),
        failure_rules=FailureRules(
            fail_on_non_finite_state=_require_boolean(
                "fail_on_non_finite_state",
                failure_values["fail_on_non_finite_state"],
            ),
            fail_on_joint_limit_violation=_require_boolean(
                "fail_on_joint_limit_violation",
                failure_values["fail_on_joint_limit_violation"],
            ),
            fail_if_not_settled=_require_boolean(
                "fail_if_not_settled",
                failure_values["fail_if_not_settled"],
            ),
        ),
        scenarios=tuple(
            EvaluationScenario(
                id=str(scenario["id"]),
                start_position_rad=float(scenario["start_position_rad"]),
                target_position_rad=float(scenario["target_position_rad"]),
                initial_velocity_rad_s=float(scenario["initial_velocity_rad_s"]),
            )
            for scenario in raw["scenarios"]
        ),
    )
    validate_evaluation_config(config)
    return config


def build_evaluation_plan(
    config: EvaluationConfig,
) -> tuple[EvaluationPlanCase, ...]:
    """Pair both controllers with one validated ordered scenario/seed suite."""
    validate_evaluation_config(config)
    controllers: tuple[EvaluationControllerName, ...] = ("p", "pd")
    return tuple(
        EvaluationPlanCase(
            controller=controller,
            scenario=scenario,
            seed=seed,
        )
        for controller in controllers
        for scenario in config.scenarios
        for seed in config.seeds
    )


def validate_evaluation_config(config: EvaluationConfig) -> None:
    """Reject evaluation suites that cannot support the declared comparison."""
    if config.schema_version != 1:
        raise ValueError(
            f"unsupported evaluation schema version: {config.schema_version}"
        )
    if not config.seeds:
        raise ValueError("evaluation requires at least one seed")
    if len(set(config.seeds)) != len(config.seeds):
        raise ValueError("evaluation seeds must be unique")
    if any(seed < 0 for seed in config.seeds):
        raise ValueError("evaluation seeds must be non-negative")

    positive_values = {
        "simulation_timestep_s": config.model.simulation_timestep_s,
        "duration_s": config.episode.duration_s,
        "control_interval_s": config.episode.control_interval_s,
        "torque_limit_nm": config.episode.torque_limit_nm,
    }
    for name, value in positive_values.items():
        _require_finite(name, value)
        if value <= 0:
            raise ValueError(f"{name} must be greater than zero")

    _require_integer_ratio(
        numerator=config.episode.control_interval_s,
        denominator=config.model.simulation_timestep_s,
        description="control interval / simulation timestep",
    )
    _require_integer_ratio(
        numerator=config.episode.duration_s,
        denominator=config.episode.control_interval_s,
        description="duration / control interval",
    )
    _require_finite(
        "target_velocity_rad_s", config.episode.target_velocity_rad_s
    )

    gain_values = {
        "p.kp": config.p.kp,
        "pd.kp": config.pd.kp,
        "pd.kd": config.pd.kd,
    }
    for name, value in gain_values.items():
        _require_finite(name, value)
        if value < 0:
            raise ValueError(f"{name} must be non-negative")

    _require_finite("rise_lower_fraction", config.metrics.rise_lower_fraction)
    _require_finite("rise_upper_fraction", config.metrics.rise_upper_fraction)
    if not (
        0.0
        <= config.metrics.rise_lower_fraction
        < config.metrics.rise_upper_fraction
        <= 1.0
    ):
        raise ValueError(
            "rise thresholds must satisfy 0 <= lower < upper <= 1"
        )
    positive_metric_values = {
        "settling_position_tolerance_rad": (
            config.metrics.settling_position_tolerance_rad
        ),
        "settling_velocity_tolerance_rad_s": (
            config.metrics.settling_velocity_tolerance_rad_s
        ),
        "steady_state_window_s": config.metrics.steady_state_window_s,
        "minimum_command_magnitude_rad": (
            config.metrics.minimum_command_magnitude_rad
        ),
    }
    for name, value in positive_metric_values.items():
        _require_finite(name, value)
        if value <= 0:
            raise ValueError(f"{name} must be greater than zero")
    if config.metrics.steady_state_window_s > config.episode.duration_s:
        raise ValueError("steady_state_window_s must not exceed duration_s")
    _require_integer_ratio(
        numerator=config.metrics.steady_state_window_s,
        denominator=config.episode.control_interval_s,
        description="steady-state window / control interval",
    )

    noise_values = {
        "position_std_rad": config.noise.position_std_rad,
        "velocity_std_rad_s": config.noise.velocity_std_rad_s,
        "action_std_nm": config.noise.action_std_nm,
    }
    for name, value in noise_values.items():
        _require_finite(name, value)
        if value < 0:
            raise ValueError(f"{name} must be non-negative")

    if not config.model.path.is_file():
        raise ValueError(f"model does not exist: {config.model.path}")
    model = mujoco.MjModel.from_xml_path(str(config.model.path))
    if (model.nq, model.nv, model.nu) != (1, 1, 1):
        raise ValueError(
            "evaluation requires exactly one position, velocity, and actuator"
        )
    if not math.isclose(
        model.opt.timestep,
        config.model.simulation_timestep_s,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "MuJoCo timestep does not match the evaluation configuration"
        )
    if (
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, config.model.keyframe)
        < 0
    ):
        raise ValueError(f"unknown model keyframe: {config.model.keyframe!r}")

    control_min, control_max = model.actuator_ctrlrange[0]
    expected_limit = config.episode.torque_limit_nm
    if not (
        math.isclose(control_min, -expected_limit, rel_tol=0.0, abs_tol=1e-12)
        and math.isclose(control_max, expected_limit, rel_tol=0.0, abs_tol=1e-12)
    ):
        raise ValueError(
            "MuJoCo actuator limits do not match the evaluation configuration"
        )

    scenario_ids = [scenario.id for scenario in config.scenarios]
    if any(not scenario_id.strip() for scenario_id in scenario_ids):
        raise ValueError("scenario id must not be empty")
    if len(set(scenario_ids)) != len(scenario_ids):
        raise ValueError("scenario ids must be unique")
    start_target_pairs = [
        (scenario.start_position_rad, scenario.target_position_rad)
        for scenario in config.scenarios
    ]
    if len(set(start_target_pairs)) != len(start_target_pairs):
        raise ValueError("scenario start-target pairs must be unique")

    joint_min, joint_max = model.jnt_range[0]
    command_directions: set[str] = set()
    for scenario in config.scenarios:
        for name, value in (
            ("start_position_rad", scenario.start_position_rad),
            ("target_position_rad", scenario.target_position_rad),
            ("initial_velocity_rad_s", scenario.initial_velocity_rad_s),
        ):
            _require_finite(name, value)
        command = scenario.target_position_rad - scenario.start_position_rad
        command_magnitude = abs(command)
        if command_magnitude < config.metrics.minimum_command_magnitude_rad:
            raise ValueError(
                f"scenario {scenario.id!r} command magnitude is below the configured "
                "minimum"
            )
        for name, position in (
            ("start", scenario.start_position_rad),
            ("target", scenario.target_position_rad),
        ):
            if not joint_min <= position <= joint_max:
                raise ValueError(
                    f"scenario {scenario.id!r} {name} position is outside the model "
                    "joint range"
                )
        command_directions.add("positive" if command > 0 else "negative")

    if command_directions != {"positive", "negative"}:
        raise ValueError(
            "evaluation scenarios must span positive and negative command directions"
        )


def validate_experiment_config(config: ExperimentConfig) -> None:
    """Reject inconsistent timing, unsafe limits, and invalid gains."""
    if not config.model.path.is_file():
        raise ValueError(f"model does not exist: {config.model.path}")
    if not config.model.keyframe:
        raise ValueError("model keyframe must not be empty")

    positive_values = {
        "simulation_timestep_s": config.model.simulation_timestep_s,
        "duration_s": config.episode.duration_s,
        "control_interval_s": config.episode.control_interval_s,
        "torque_limit_nm": config.episode.torque_limit_nm,
    }
    for name, value in positive_values.items():
        _require_finite(name, value)
        if value <= 0:
            raise ValueError(f"{name} must be greater than zero")

    state_values = {
        "start_position_rad": config.episode.start_position_rad,
        "target_position_rad": config.episode.target_position_rad,
        "target_velocity_rad_s": config.episode.target_velocity_rad_s,
    }
    for name, value in state_values.items():
        _require_finite(name, value)

    gain_values = {
        "p.kp": config.p.kp,
        "pd.kp": config.pd.kp,
        "pd.kd": config.pd.kd,
    }
    for name, value in gain_values.items():
        _require_finite(name, value)
        if value < 0:
            raise ValueError(f"{name} must be non-negative")

    _require_integer_ratio(
        numerator=config.episode.control_interval_s,
        denominator=config.model.simulation_timestep_s,
        description="control interval / simulation timestep",
    )
    _require_integer_ratio(
        numerator=config.episode.duration_s,
        denominator=config.episode.control_interval_s,
        description="duration / control interval",
    )


def _require_integer_ratio(
    *, numerator: float, denominator: float, description: str
) -> int:
    ratio = numerator / denominator
    rounded_ratio = round(ratio)
    if not math.isclose(ratio, rounded_ratio, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"{description} must be an integer, got {ratio}")
    return rounded_ratio


def _require_finite(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


def _require_boolean(name: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _require_integer(name: str, value: object) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer")
    return value
