import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from learned_dynamics.config import load_training_config
from learned_dynamics.studies import (
    freeze_validation_roster, load_study_config, require_frozen_validation,
    run_studies, study_matrix,
)


ROOT = Path(__file__).resolve().parents[3]
TRAINING = ROOT / "experiments/04-learned-dynamics/configs/training-v1.toml"
STUDIES = ROOT / "experiments/04-learned-dynamics/configs/studies-v1.toml"


def test_frozen_matrix_has_six_variants_45_runs_and_30_final_candidates():
    matrix = study_matrix(load_training_config(TRAINING))
    assert matrix["variant_membership"] == {
        "canonical32": "all", "capacity16": "all",
        "history0": "history_common", "history1": "history_common",
        "loss1": "loss_common", "loss5": "loss_common",
    }
    assert matrix["planned_run_count"] == 45
    assert matrix["final_test_run_count"] == 30
    assert matrix["final_test_seeds"] == [0, 1, 2, 3, 4]
    assert matrix["diagnostic_run_count"] == 15
    assert matrix["matched_pairs"] == [
        {"variants": ["history0", "history1"], "membership": "history_common"},
        {"variants": ["loss1", "loss5"], "membership": "loss_common"},
    ]


def test_committed_study_config_binds_training_evaluation_and_fresh_outputs():
    config = load_study_config(STUDIES)
    assert config.training_config == TRAINING
    assert config.evaluation_config.name == "evaluation-v1.toml"
    assert config.output_dir.name == "study-v1"
    assert config.artifact_budget_bytes == 524288000


def test_final_test_is_blocked_until_validation_roster_is_frozen(tmp_path):
    with pytest.raises(ValueError, match="frozen validation"):
        require_frozen_validation(tmp_path)


def test_freeze_binds_every_selection_and_rejects_later_roster_tampering(tmp_path):
    roster_dir = tmp_path / "training"
    roster_dir.mkdir()
    roster = {
        "schema_version": 1, "training_config_sha256": "a" * 64,
        "publication_status": "clean_source", "config": {},
        "provenance": {"source_sha": "b" * 40, "working_tree_dirty": False},
        "planned_runs": [{"run_id": "run-a", "run_spec": {"seed": 0}}],
        "results": [{"run_id": "run-a", "status": "success",
                     "checkpoint_dir": "run-a/checkpoint", "checkpoint_sha256": "c" * 64,
                     "selected_epoch": 7, "selected_validation_mse": 0.25,
                     "failure_reason": None}],
    }
    (roster_dir / "run-manifest.json").write_text(json.dumps(roster))
    freeze_validation_roster(tmp_path / "study", roster_dir, roster, "d" * 64)
    loaded = require_frozen_validation(tmp_path / "study")
    assert loaded["selections"][0]["selected_epoch"] == 7

    roster["results"][0]["selected_epoch"] = 8
    with pytest.raises(ValueError, match="frozen validation"):
        require_frozen_validation(tmp_path / "study", roster=roster)

    roster["results"][0]["selected_epoch"] = 7
    (roster_dir / "run-manifest.json").write_text(json.dumps(roster) + "\n")
    with pytest.raises(ValueError, match="frozen validation"):
        require_frozen_validation(tmp_path / "study", roster=roster)


def test_validation_phase_trains_then_freezes_before_benchmark(tmp_path, monkeypatch):
    from learned_dynamics import studies

    roster_dir, study_dir = tmp_path / "training", tmp_path / "study"
    roster = {
        "schema_version": 1, "training_config_sha256": "a" * 64,
        "publication_status": "clean_source", "config": {},
        "provenance": {"source_sha": "b" * 40, "working_tree_dirty": False},
        "planned_runs": [{"run_id": "canonical32-mps-seed0-repeat0",
                          "run_spec": {"variant_id": "canonical32", "seed": 0,
                                       "device": "mps", "repeat": 0}}],
        "results": [{"run_id": "canonical32-mps-seed0-repeat0", "status": "success",
                     "checkpoint_dir": "canonical/checkpoint", "checkpoint_sha256": "c" * 64,
                     "selected_epoch": 7, "selected_validation_mse": 0.25,
                     "failure_reason": None}],
    }
    config = studies.StudyConfig(1, tmp_path / "studies.toml", "d" * 64,
                                 tmp_path / "training.toml", tmp_path / "evaluation.toml",
                                 study_dir, tmp_path / "benchmark", 524288000)
    events = []

    def fake_train(config_path, output, allow_full_study):
        events.append("train")
        roster_dir.mkdir()
        (roster_dir / "run-manifest.json").write_text(json.dumps(roster))
        return roster_dir

    monkeypatch.setattr(studies, "load_study_config", lambda path: config)
    monkeypatch.setattr(studies, "pin_cpu_threads", lambda: events.append("pin"))
    monkeypatch.setattr(studies, "load_training_config",
                        lambda path: type("Training", (), {"output_dir": roster_dir,
                            "dataset_dir": tmp_path / "dataset",
                            "dataset_version": "experiment-02-transition-v1"})())
    monkeypatch.setattr(studies, "train", fake_train)
    monkeypatch.setattr(studies, "load_run_manifest", lambda path: roster)
    monkeypatch.setattr(studies, "load_views",
                        lambda path, version: type("Views", (), {"validation": object()})())
    monkeypatch.setattr(studies, "benchmark_inference",
                        lambda checkpoint, validation, output: events.append("benchmark") or output)
    path = run_studies(tmp_path / "studies.toml", study_dir, "validation")
    assert path.name == "validation-freeze.json"
    assert events == ["pin", "train", "benchmark"]
    assert path.is_file()


def test_final_phase_requires_exact_frozen_roster_before_evaluation(tmp_path, monkeypatch):
    from learned_dynamics import studies

    roster_dir, study_dir = tmp_path / "training", tmp_path / "study"
    roster_dir.mkdir()
    roster = {
        "schema_version": 1, "training_config_sha256": "a" * 64,
        "publication_status": "clean_source", "config": {},
        "provenance": {"source_sha": "b" * 40, "working_tree_dirty": False},
        "planned_runs": [{"run_id": "run-a", "run_spec": {"seed": 0}}],
        "results": [{"run_id": "run-a", "status": "failed", "checkpoint_dir": None,
                     "checkpoint_sha256": None, "selected_epoch": None,
                     "selected_validation_mse": None, "failure_reason": "synthetic"}],
    }
    (roster_dir / "run-manifest.json").write_text(json.dumps(roster))
    config = studies.StudyConfig(1, tmp_path / "studies.toml", "d" * 64,
                                 tmp_path / "training.toml", tmp_path / "evaluation.toml",
                                 study_dir, tmp_path / "benchmark", 524288000)
    freeze_validation_roster(study_dir, roster_dir, roster, config.config_sha256)
    calls = []
    monkeypatch.setattr(studies, "load_study_config", lambda path: config)
    monkeypatch.setattr(studies, "load_training_config",
                        lambda path: type("Training", (), {"output_dir": roster_dir})())
    monkeypatch.setattr(studies, "load_run_manifest", lambda path: roster)
    evaluation_output = tmp_path / "evaluation"
    config.benchmark_output_dir.mkdir()
    (config.benchmark_output_dir / "manifest.json").write_text("benchmark\n")
    monkeypatch.setattr(studies, "load_evaluation_config",
                        lambda path: type("Evaluation", (), {"output_dir": evaluation_output})())

    def fake_evaluation(evaluation, roster_path, output):
        calls.append((roster_path, output))
        output.mkdir()
        (output / "manifest.json").write_text("evaluation\n")
        return output

    monkeypatch.setattr(studies, "run_evaluation", fake_evaluation)
    path = run_studies(tmp_path / "studies.toml", study_dir, "final_test")
    assert path == study_dir / "final-test.json"
    assert len(calls) == 1
    payload = json.loads(path.read_text())
    assert payload["evaluation_dir"] == "../evaluation"
    assert payload["benchmark_dir"] == "../benchmark"


def test_run_studies_cli_requires_explicit_phase_and_output():
    script = ROOT / "experiments/04-learned-dynamics/scripts/run_studies.py"
    result = subprocess.run([sys.executable, str(script), "--help"], cwd=ROOT,
                            capture_output=True, text=True)
    assert result.returncode == 0
    for option in ("--config", "--output", "--phase", "--power-state", "--competing-load"):
        assert option in result.stdout
