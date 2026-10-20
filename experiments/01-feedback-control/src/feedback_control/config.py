"""Versioned configuration for the feedback-control experiment."""

from dataclasses import dataclass
import math
from pathlib import Path
import tomllib


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
