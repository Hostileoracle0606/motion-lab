from pathlib import Path

import pytest

from feedback_control.config import load_experiment_config
from feedback_control.simulation import run_episode
from feedback_control.tuning import (
    estimate_critical_derivative_gain,
    load_gain_sweep_config,
    nearest_candidate,
    summarize_response,
)


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = EXPERIMENT_DIR / "configs" / "demo.toml"
SWEEP_PATH = EXPERIMENT_DIR / "configs" / "gain_sweep.toml"


def test_declared_pd_gain_is_nearest_candidate_to_critical_estimate() -> None:
    config = load_experiment_config(CONFIG_PATH)
    sweep = load_gain_sweep_config(SWEEP_PATH)
    critical_kd = estimate_critical_derivative_gain(config, kp=sweep.pd_kp)

    assert critical_kd == pytest.approx(1.3136, abs=1e-4)
    assert nearest_candidate(sweep.pd_kd_values, critical_kd) == pytest.approx(1.3)
    assert config.p.kp == config.pd.kp == sweep.pd_kp
    assert config.pd.kd == pytest.approx(1.3)


def test_pd_demo_has_less_overshoot_and_no_repeated_crossing() -> None:
    config = load_experiment_config(CONFIG_PATH)
    p_result = run_episode(config, "p")
    pd_result = run_episode(config, "pd")

    p_diagnostics = summarize_response(p_result, kp=config.p.kp, kd=0.0)
    pd_diagnostics = summarize_response(
        pd_result,
        kp=config.pd.kp,
        kd=config.pd.kd,
    )

    assert p_diagnostics.target_crossings >= 2
    assert pd_diagnostics.target_crossings == 0
    assert pd_diagnostics.overshoot_rad < p_diagnostics.overshoot_rad
    assert pd_diagnostics.final_abs_error_rad < p_diagnostics.final_abs_error_rad


def test_nearest_candidate_breaks_ties_toward_smaller_gain() -> None:
    assert nearest_candidate((1.0, 2.0), 1.5) == pytest.approx(1.0)
