from pathlib import Path
import tomllib

import pytest

from model_predictive_control.predictor_comparison import (
    CONTROLLER_IDS, build_roster, verify_frozen_manifest,
)


EXPERIMENT = Path(__file__).resolve().parents[1]
CONFIG = EXPERIMENT / "configs/predictor-comparison-v1.toml"


def test_frozen_roster_pairs_truth_conditions_and_controllers():
    config = tomllib.loads(CONFIG.read_text())
    cases, attempts = build_roster(config)
    assert (len(cases), len(attempts)) == (80, 320)
    assert (cases, attempts) == build_roster(config)
    assert len({case.scenario.scenario_id for case in cases}) == 80
    for seed in config["roster"]["seeds"]:
        paired = [case.scenario for case in cases if case.seed == seed]
        assert len({case.initial_state for case in paired}) == 1
        assert len({case.target_position_rad for case in paired}) == 1
    assert [attempt["controller_id"] for attempt in attempts[:4]] == list(CONTROLLER_IDS)


def test_frozen_manifest_refuses_mismatched_inputs_before_execution():
    declared = {"protocol": {"config_sha256": "original"}, "source_commit": "fixture"}
    verify_frozen_manifest({"config_sha256": "original"}, declared)
    with pytest.raises(ValueError, match="frozen manifest"):
        verify_frozen_manifest({"config_sha256": "changed"}, declared)


def test_roster_refuses_duplicate_seeds():
    config = tomllib.loads(CONFIG.read_text())
    config["roster"]["seeds"].append(config["roster"]["seeds"][0])
    with pytest.raises(ValueError, match="unique"):
        build_roster(config)
