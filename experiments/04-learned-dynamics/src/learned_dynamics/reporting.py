"""Inference-free CSV, plot, and Markdown generation from audited records."""

import csv
import json
from pathlib import Path
import subprocess

from forward_dynamics.serialization import (
    decode_object, json_bytes, require_equal, require_fields, sha256_file,
)
from learned_dynamics.benchmarking import load_benchmark_evidence
from learned_dynamics.evidence import load_evidence
from learned_dynamics.report_markdown import render_report
from learned_dynamics.report_plots import write_horizon_plot
from learned_dynamics.report_tables import (
    aggregate_seed_statistics, report_summary_rows, study_summary_rows,
    timing_summary_rows,
)
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


def _learning_curves(roster_dir: Path, live: dict) -> list[dict]:
    rows = []
    for result in live["results"]:
        path = roster_dir / result["run_id"] / "learning-curve.json"
        if not path.is_file():
            continue
        curve = decode_object(path.read_text())
        for epoch in curve["epochs"]:
            rows.append({"run_id": result["run_id"], **epoch})
    return rows


def _load_final_record(study: Path) -> dict:
    final = decode_object((study / "final-test.json").read_text())
    legacy = {
        "schema_version", "study_config_sha256", "validation_freeze_sha256",
        "evaluation_dir", "evaluation_manifest_sha256",
        "benchmark_dir", "benchmark_manifest_sha256",
    }
    current = legacy | {"benchmark_status", "benchmark_unavailable_sha256"}
    if set(final) == legacy:
        final = final | {"benchmark_status": "available",
                         "benchmark_unavailable_sha256": None}
    else:
        require_fields(final, current, "final-test study record")
    if type(final["schema_version"]) is not int or final["schema_version"] != 1:
        raise ValueError("unsupported final-test study schema")
    return final


def _load_benchmark(study: Path, final: dict) -> dict:
    status = final["benchmark_status"]
    if status == "available":
        benchmark_dir = _resolve_study_input(study, final["benchmark_dir"])
        require_equal(final["benchmark_manifest_sha256"],
                      sha256_file(benchmark_dir / "manifest.json"),
                      "final-test benchmark manifest")
        return load_benchmark_evidence(benchmark_dir)
    if status != "unavailable" or final["benchmark_dir"] is not None or (
            final["benchmark_manifest_sha256"] is not None):
        raise ValueError("final-test benchmark status is invalid")
    unavailable_path = study / "benchmark-unavailable.json"
    require_equal(final["benchmark_unavailable_sha256"], sha256_file(unavailable_path),
                  "final-test benchmark-unavailable record")
    unavailable = decode_object(unavailable_path.read_text())
    return {
        "schema_version": 1,
        "environment": {"hardware_model": "unavailable",
                        "power_state": "not recorded", "competing_load": "not recorded"},
        "protocol": {},
        "results": [{"status": "failed", "device": "unavailable", "round": None,
                     "failure_reason": unavailable["reason"]}],
    }


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


def _command_path(path: Path) -> str:
    resolved = path.resolve()
    repository = Path(__file__).resolve().parents[4]
    return str(resolved.relative_to(repository)) if resolved.is_relative_to(repository) else str(resolved)


def _report_generator_provenance() -> dict:
    repository = Path(__file__).resolve().parents[4]
    source = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository,
        capture_output=True, text=True, check=True).stdout.strip()
    tracked_status = subprocess.run(
        ["git", "status", "--short", "--untracked-files=no"], cwd=repository,
        capture_output=True, text=True, check=True).stdout.strip()
    return {"source_sha": source, "working_tree_dirty": bool(tracked_status)}


def generate_report(input_dir: Path, output_dir: Path) -> Path:
    study = Path(input_dir)
    output = Path(output_dir)
    if output.exists() or output.is_symlink():
        raise ValueError("report output must be a fresh directory")
    final = _load_final_record(study)
    evaluation_dir = _resolve_study_input(study, final["evaluation_dir"])
    require_equal(final["evaluation_manifest_sha256"],
                  sha256_file(evaluation_dir / "manifest.json"),
                  "final-test evaluation manifest")
    evidence = load_evidence(evaluation_dir)
    benchmark = _load_benchmark(study, final)
    roster_dir = (evaluation_dir / evidence["training_roster_dir"]).resolve()
    roster = load_run_manifest(roster_dir, validate_checkpoints=False)
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
    _write_csv(output / "learning_curves.csv", _learning_curves(roster_dir, roster))
    _write_csv(output / "study_summary.csv", study_summary_rows(roster_dir, roster))
    _write_csv(output / "timing.csv", _timing_rows(benchmark))
    _write_csv(output / "timing_summary.csv", timing_summary_rows(benchmark))
    plot_rows = list(seed_rows)
    for summary in evidence["summaries"]:
        if summary["training_seed"] is None and summary["success_metrics"] is not None:
            plot_rows.append({
                "variant_id": summary["predictor_id"],
                "membership": summary["membership"],
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
    benchmark_environment = dict(benchmark["environment"])
    if "hardware_model" not in benchmark_environment:
        benchmark_environment["hardware_model"] = "unavailable in benchmark schema v1"
    provenance = {
        "schema_version": 1, "source_sha": source,
        "study_config_sha256": final["study_config_sha256"],
        "validation_freeze_sha256": final["validation_freeze_sha256"],
        "evaluation_manifest_sha256": final["evaluation_manifest_sha256"],
        "benchmark_status": final["benchmark_status"],
        "benchmark_manifest_sha256": final["benchmark_manifest_sha256"],
        "benchmark_unavailable_sha256": final["benchmark_unavailable_sha256"],
        "evaluation_provenance": evidence["evaluation_provenance"],
        "report_generator_provenance": _report_generator_provenance(),
        "training_provenance": roster["provenance"],
        "dataset_identity": {
            "dataset_version": evidence["dataset_version"],
            "files": evidence["dataset_files"],
            "evaluation_config_sha256": evidence["config_sha256"],
            "training_roster_sha256": evidence["training_roster_sha256"],
        },
        "benchmark_environment": benchmark_environment,
        "benchmark_protocol": benchmark["protocol"],
        "regeneration_command": [
            "uv", "run", "--frozen", "python",
            "experiments/04-learned-dynamics/scripts/generate_report.py",
            "--input", _command_path(study), "--output", _command_path(output),
        ],
    }
    (output / "provenance.json").write_bytes(json_bytes(provenance, indent=2))
    report = output / "report.md"
    report.write_text(render_report(
        seed_rows, benchmark, source, evidence["episode_outcomes"], provenance))
    files = {path.name: sha256_file(path) for path in sorted(output.iterdir()) if path.is_file()}
    (output / "manifest.json").write_bytes(json_bytes({
        "schema_version": 1, "files": files,
    }, indent=2))
    return report
