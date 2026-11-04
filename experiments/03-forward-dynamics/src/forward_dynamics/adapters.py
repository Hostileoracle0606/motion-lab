"""Adapter construction and reproducible settings for the declared protocol."""

from forward_dynamics.baselines import ConstantVelocityPredictor, NoChangePredictor
from forward_dynamics.config import EvaluationConfig
from forward_dynamics.types import Predictor


def make_predictor(predictor_id: str, config: EvaluationConfig) -> Predictor:
    if predictor_id == "no_change":
        return NoChangePredictor()
    if predictor_id == "constant_velocity":
        return ConstantVelocityPredictor()
    if predictor_id == "mujoco":
        from forward_dynamics.mujoco_predictor import MujocoPredictor
        return MujocoPredictor(config.model_path)
    raise ValueError(f"unsupported predictor_id: {predictor_id}")


def adapter_descriptions(config: EvaluationConfig) -> list[dict]:
    """Describe the supported reference without invoking simulator dynamics."""
    return [
        {"predictor_id": name, "parameters": (
            {"simulation_timestep_s": 0.002, "model_sha256": config.environment_version,
             "reset": "fresh_data"} if name == "mujoco" else {}
        )}
        for name in config.predictor_ids
    ]
