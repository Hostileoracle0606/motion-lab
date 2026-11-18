"""Inference-free CSV, plot, and Markdown generation from audited records."""

import csv
import json
from pathlib import Path

from forward_dynamics.serialization import (
    decode_object, json_bytes, require_equal, require_fields, sha256_file,
)
from learned_dynamics.benchmarking import load_benchmark_evidence
from learned_dynamics.evidence import load_evidence
from learned_dynamics.report_markdown import render_report
from learned_dynamics.report_plots import write_horizon_plot
from learned_dynamics.report_tables import aggregate_seed_statistics, report_summary_rows
from learned_dynamics.training_evidence import load_run_manifest


def _csv_value(value):
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or ["status"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fields})


def _flatten_summary(row: dict) -> dict:
    return {key: value for key, value in row.items()
            if key not in ("counts", "success_metrics", "units")} | {
        **{f"count_{key}": value for key, value in row["counts"].items()},
        **(row["success_metrics"] or {}),
    }


def _learning_curves(evaluation_dir: Path, evidence: dict) -> list[dict]:
    roster_dir = (evaluation_dir / evidence["training_roster_dir"]).resolve()
    live = load_run_manifest(roster_dir, validate_checkpoints=False)
    rows = []
    for result in live["results"]:
        path = roster_dir / result["run_id"] / "learning-curve.json"
        if not path.is_file():
            continue
        curve = decode_object(path.read_text())
        for epoch in curve["epochs"]:
            rows.append({"run_id": result["run_id"], **epoch})
    return rows


def _timing_rows(benchmark: dict) -> list[dict]:
    rows = []
    for result in benchmark["results"]:
        if result["status"] == "failed":
            rows.append(dict(result))
            continue
        for index, sample in enumerate(result["samples_s"]):
            rows.append({
                "status": "success", "device": result["device"], "round": result["round"],
                "mode": result["mode"], "batch_size": result["batch_size"],
                "sample_index": index, "seconds": sample,
            })
    return rows


def _resolve_study_input(study: Path, value: object) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError("report inputs must be local relative directories")
    results_root = study.resolve().parent
    resolved = (study / value).resolve()
    try:
        resolved.relative_to(results_root)
    except ValueError as error:
        raise ValueError("report inputs must stay within the study results root") from error
    if resolved == study.resolve():
        raise ValueError("report inputs must be distinct from the study directory")
    return resolved


def generate_report(input_dir: Path, output_dir: Path) -> Path:
    study = Path(input_dir)
    output = Path(output_dir)
    if output.exists() or output.is_symlink():
        raise ValueError("report output must be a fresh directory")
    final = decode_object((study / "final-test.json").read_text())
    require_fields(final, {
        "schema_version", "study_config_sha256", "validation_freeze_sha256",
        "evaluation_dir", "evaluation_manifest_sha256",
        "benchmark_dir", "benchmark_manifest_sha256",
    }, "final-test study record")
    if type(final["schema_version"]) is not int or final["schema_version"] != 1:
        raise ValueError("unsupported final-test study schema")
    evaluation_dir = _resolve_study_input(study, final["evaluation_dir"])
    benchmark_dir = _resolve_study_input(study, final["benchmark_dir"])
    require_equal(final["evaluation_manifest_sha256"],
                  sha256_file(evaluation_dir / "manifest.json"),
                  "final-test evaluation manifest")
    require_equal(final["benchmark_manifest_sha256"],
                  sha256_file(benchmark_dir / "manifest.json"),
                  "final-test benchmark manifest")
    evidence = load_evidence(evaluation_dir)
    benchmark = load_benchmark_evidence(benchmark_dir)
    learned = report_summary_rows(evidence)
    seed_rows = aggregate_seed_statistics(learned)
    summaries = [_flatten_summary(row) for row in evidence["summaries"]]
    grouped = [_flatten_summary(row) for row in evidence["grouped_summaries"]["episode_id"]]
    coverage = [{key: value for key, value in row.items() if key != "success_metrics"}
                for row in summaries]
    output.mkdir(parents=True)
    _write_csv(output / "one_step.csv", [row for row in summaries if row["horizon_steps"] == 1])
    _write_csv(output / "rollouts.csv", [row for row in summaries if row["horizon_steps"] > 1])
    _write_csv(output / "coverage.csv", coverage)
    _write_csv(output / "per_episode.csv", grouped)
    _write_csv(output / "seed_summary.csv", list(seed_rows))
    _write_csv(output / "learning_curves.csv", _learning_curves(evaluation_dir, evidence))
    _write_csv(output / "timing.csv", _timing_rows(benchmark))
    plot_rows = list(seed_rows)
    for summary in evidence["summaries"]:
        if summary["training_seed"] is None and summary["success_metrics"] is not None:
            plot_rows.append({
                "variant_id": summary["predictor_id"],
                "horizon_steps": summary["horizon_steps"],
                "position_mae_rad_mean": summary["success_metrics"]["position_mae_rad"],
                "velocity_mae_rad_s_mean": summary["success_metrics"]["velocity_mae_rad_s"],
            })
    position_series = write_horizon_plot(
        plot_rows, "position_mae_rad_mean", "Position MAE (rad)",
        output / "position_error_by_horizon.png")
    velocity_series = write_horizon_plot(
        plot_rows, "velocity_mae_rad_s_mean", "Velocity MAE (rad/s)",
        output / "velocity_error_by_horizon.png")
    (output / "plot_data.json").write_text(json.dumps(
        {"position": position_series, "velocity": velocity_series},
        sort_keys=True, indent=2) + "\n")
    source = evidence["evaluation_provenance"]["source_sha"]
    report = output / "report.md"
    report.write_text(render_report(seed_rows, benchmark, source))
    files = {path.name: sha256_file(path) for path in sorted(output.iterdir()) if path.is_file()}
    (output / "manifest.json").write_bytes(json_bytes({
        "schema_version": 1, "files": files,
    }, indent=2))
    return report
