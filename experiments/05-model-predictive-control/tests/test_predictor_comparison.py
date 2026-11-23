from pathlib import Path
import hashlib
import json
import tomllib

import pytest

from model_predictive_control import predictor_comparison as comparison
from model_predictive_control.predictor_comparison import (
    CONTROLLER_IDS, build_roster, run_identity, verify_frozen_manifest,
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
    declared = {"schema_version": 1, "protocol": {"config_sha256": "original"},
                "source_commit": "fixture"}
    verify_frozen_manifest({"config_sha256": "original"}, declared)
    with pytest.raises(ValueError, match="frozen manifest"):
        verify_frozen_manifest({"config_sha256": "changed"}, declared)


def test_roster_refuses_duplicate_seeds():
    config = tomllib.loads(CONFIG.read_text())
    config["roster"]["seeds"].append(config["roster"]["seeds"][0])
    with pytest.raises(ValueError, match="unique"):
        build_roster(config)


def test_run_identity_is_bounded_by_frozen_repetition_counts():
    config = tomllib.loads(CONFIG.read_text())
    assert run_identity(config, "fixed-work", 2) == "fixed-work-02"
    assert run_identity(config, "deadline-discard", 1) == "deadline-discard-01"
    with pytest.raises(ValueError, match="repetition"):
        run_identity(config, "deadline-discard", 2)


def test_frozen_manifest_pins_transitive_sources_and_attempt_policy():
    protocol = {"implementation_objects": {"dataset/source": "original", "policy.md": "original"}}
    frozen = {"schema_version": 1, "protocol": protocol}
    for key in protocol["implementation_objects"]:
        changed = {"implementation_objects": {**protocol["implementation_objects"], key: "changed"}}
        with pytest.raises(ValueError, match="frozen manifest"):
            verify_frozen_manifest(changed, frozen)


def test_study_audit_refuses_missing_or_duplicate_repetitions(tmp_path, monkeypatch):
    config = tomllib.loads(CONFIG.read_text())
    runs = comparison.planned_runs(config)
    attempts = [{"scenario_id": "fixture", "controller_id": "pd", "seed": 0}]
    protocol = {"planned_runs": runs, "planned_attempts": attempts}
    frozen = tmp_path / "frozen.json"
    frozen.write_text(json.dumps({"schema_version": 1, "protocol": protocol}))
    digest = hashlib.sha256(frozen.read_bytes()).hexdigest()
    monkeypatch.setattr(comparison, "audit_run", lambda _: {"planned": attempts})
    outputs = []
    for run in runs:
        output = tmp_path / run["run_id"]
        output.mkdir()
        (output / "manifest.json").write_text(json.dumps({
            **run, "frozen_manifest_sha256": digest, "protocol": protocol,
        }))
        outputs.append(output)
    assert comparison.audit_comparison_runs(frozen, outputs) == [row["run_id"] for row in runs]
    with pytest.raises(ValueError, match="missing planned runs"):
        comparison.audit_comparison_runs(frozen, outputs[:-1])
    with pytest.raises(ValueError, match="duplicate run ID"):
        comparison.audit_comparison_runs(frozen, [*outputs, outputs[0]])
