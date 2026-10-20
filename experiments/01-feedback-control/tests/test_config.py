from pathlib import Path

from feedback_control.config import load_experiment_config


CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "demo.toml"


def test_demo_config_declares_integer_clock_ratios() -> None:
    config = load_experiment_config(CONFIG_PATH)

    assert config.episode.control_interval_s / config.model.simulation_timestep_s == 5
    assert config.episode.duration_s / config.episode.control_interval_s == 400
    assert config.model.path.is_file()
