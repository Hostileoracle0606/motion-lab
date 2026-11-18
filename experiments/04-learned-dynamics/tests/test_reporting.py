import csv
import json
import math
from pathlib import Path
import subprocess
import sys
import hashlib

import pytest

from learned_dynamics.report_plots import horizon_series
from learned_dynamics.report_tables import aggregate_seed_statistics
from learned_dynamics.reporting import generate_report


def seed_row(seed, value, *, success=10, failed=0):
    return {
        "variant_id": "canonical32", "membership": "all", "horizon_steps": 1,
        "training_seed": seed,
        "counts": {"attempted": 10, "eligible": 10, "success": success,
                   "failed": failed, "unavailable": 0},
        "success_metrics": None if success == 0 else {
            "sample_count": success,
            "position_mae_rad": value, "position_rmse_rad": value,
            "position_max_abs_rad": value,
            "velocity_mae_rad_s": value * 2, "velocity_rmse_rad_s": value * 2,
            "velocity_max_abs_rad_s": value * 2,
        },
    }


def test_seed_statistics_use_sample_std_and_never_hide_missing_or_failed_seeds():
    complete = aggregate_seed_statistics(tuple(seed_row(seed, float(seed * 2))
                                               for seed in range(5)))
    row = complete[0]
    assert row["seed_count"] == 5
    assert row["complete_seed_matrix"] is True
    assert row["position_mae_rad_mean"] == 4.0
    assert math.isclose(row["position_mae_rad_std"], math.sqrt(10.0))
    assert row["failed_windows"] == 0

    partial = aggregate_seed_statistics(tuple(seed_row(seed, float(seed), failed=1)
                                              for seed in range(4)))
    assert partial[0]["complete_seed_matrix"] is False
    assert partial[0]["missing_seeds"] == [4]
    assert partial[0]["failed_windows"] == 4


def test_zero_error_horizon_series_remains_numeric_and_matches_table_rows():
    rows = (
        {"variant_id": "canonical32", "horizon_steps": 1,
         "position_mae_rad_mean": 0.0},
        {"variant_id": "canonical32", "horizon_steps": 5,
         "position_mae_rad_mean": 0.25},
    )
    assert horizon_series(rows, "position_mae_rad_mean") == {
        "canonical32": {"horizons": [1, 5], "values": [0.0, 0.25]}}


def test_generate_report_uses_only_audited_saved_records(tmp_path, monkeypatch):
    from learned_dynamics import reporting

    summaries = []
    records = []
    for seed in range(5):
        predictor = f"canonical32-mps-seed{seed}-repeat0"
        summaries.append({
            "predictor_id": predictor, "membership": "all", "horizon_steps": 1,
            "group_by": None, "group_value": "all", "training_seed": seed,
            "units": {"position": "rad", "velocity": "rad/s", "horizon": "s"},
            "counts": {"attempted": 10, "eligible": 10, "success": 10,
                       "failed": 0, "unavailable": 0},
            "success_metrics": {"sample_count": 10, "position_mae_rad": seed * 0.1,
                "position_rmse_rad": seed * 0.1, "position_max_abs_rad": seed * 0.1,
                "velocity_mae_rad_s": seed * 0.2, "velocity_rmse_rad_s": seed * 0.2,
                "velocity_max_abs_rad_s": seed * 0.2},
        })
        records.append({"predictor_id": predictor, "membership": "all",
                        "horizon_steps": 1, "status": "success",
                        "episode_id": "test-episode", "scenario_id": "test-scenario",
                        "controller_id": "p", "split": "test",
                        "candidate": {"run_spec": {"variant_id": "canonical32",
                                                     "seed": seed}},
                        "residual": {"position_error_rad": 0.0,
                                     "velocity_error_rad_s": 0.0}})
    evidence = {
        "schema_version": 1, "summaries": summaries,
        "grouped_summaries": {name: [] for name in
                              ("episode_id", "scenario_id", "controller_id", "split")},
        "episode_outcomes": [], "records": tuple(records),
        "roster": {"results": []}, "training_roster_dir": "training",
        "evaluation_provenance": {"source_sha": "a" * 40},
    }
    summaries.append({
        "predictor_id": "no_change", "membership": "all", "horizon_steps": 1,
        "group_by": None, "group_value": "all", "training_seed": None,
        "units": {"position": "rad", "velocity": "rad/s", "horizon": "s"},
        "counts": {"attempted": 10, "eligible": 10, "success": 10,
                   "failed": 0, "unavailable": 0},
        "success_metrics": {"sample_count": 10, "position_mae_rad": 0.0,
            "position_rmse_rad": 0.0, "position_max_abs_rad": 0.0,
            "velocity_mae_rad_s": 0.0, "velocity_rmse_rad_s": 0.0,
            "velocity_max_abs_rad_s": 0.0},
    })
    benchmark = {"schema_version": 1, "results": [], "environment": {}, "protocol": {}}
    monkeypatch.setattr(reporting, "load_evidence", lambda path: evidence)
    monkeypatch.setattr(reporting, "load_benchmark_evidence", lambda path: benchmark)
    monkeypatch.setattr(reporting, "load_run_manifest", lambda path, validate_checkpoints=False:
                        {"results": []})
    monkeypatch.setattr("learned_dynamics.training.train",
                        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("training")))
    monkeypatch.setattr("learned_dynamics.evaluation.run_evaluation",
                        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("inference")))
    results_dir = tmp_path / "results"
    input_dir = results_dir / "study-v1"
    evaluation_dir = results_dir / "evaluation-v1"
    benchmark_dir = results_dir / "benchmark-v1"
    input_dir.mkdir(parents=True)
    evaluation_dir.mkdir()
    benchmark_dir.mkdir()
    (evaluation_dir / "manifest.json").write_text("evaluation\n")
    (benchmark_dir / "manifest.json").write_text("benchmark\n")
    evaluation_hash = hashlib.sha256(
        (evaluation_dir / "manifest.json").read_bytes()).hexdigest()
    benchmark_hash = hashlib.sha256(
        (benchmark_dir / "manifest.json").read_bytes()).hexdigest()
    (input_dir / "final-test.json").write_text(json.dumps({
        "schema_version": 1, "study_config_sha256": "b" * 64,
        "validation_freeze_sha256": "c" * 64,
        "evaluation_dir": "../evaluation-v1", "evaluation_manifest_sha256": evaluation_hash,
        "benchmark_dir": "../benchmark-v1", "benchmark_manifest_sha256": benchmark_hash}))
    report = generate_report(input_dir, tmp_path / "report")
    assert report.is_file()
    for name in ("one_step.csv", "rollouts.csv", "coverage.csv", "per_episode.csv",
                 "seed_summary.csv", "learning_curves.csv", "timing.csv",
                 "plot_data.json", "manifest.json", "position_error_by_horizon.png",
                 "velocity_error_by_horizon.png"):
        assert (report.parent / name).is_file()
    seed_rows = list(csv.DictReader((report.parent / "seed_summary.csv").open()))
    assert seed_rows[0]["position_mae_rad_mean"] == "0.2"
    plot_data = json.loads((report.parent / "plot_data.json").read_text())
    assert plot_data["position"]["no_change"]["values"] == [0.0]
    text = report.read_text()
    assert "overlapping windows" in text
    assert "five training seeds" in text
    assert "known-physics" in text

    final = json.loads((input_dir / "final-test.json").read_text())
    final["evaluation_manifest_sha256"] = "f" * 64
    (input_dir / "final-test.json").write_text(json.dumps(final))
    with pytest.raises(ValueError, match="final-test evaluation manifest"):
        generate_report(input_dir, tmp_path / "report-tampered")


def test_generate_report_cli_has_only_saved_input_and_fresh_output():
    root = Path(__file__).resolve().parents[3]
    script = root / "experiments/04-learned-dynamics/scripts/generate_report.py"
    result = subprocess.run([sys.executable, str(script), "--help"], cwd=root,
                            capture_output=True, text=True)
    assert result.returncode == 0
    assert "--input" in result.stdout
    assert "--output" in result.stdout
