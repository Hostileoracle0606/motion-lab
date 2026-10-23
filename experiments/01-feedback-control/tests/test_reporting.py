import csv
import hashlib
from pathlib import Path
import subprocess
import sys

import pytest

from feedback_control.reporting import generate_report, load_evaluation_evidence
from feedback_control.evaluator import run_evaluation


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
BASE_CONFIG_PATH = EXPERIMENT_DIR / "configs" / "evaluation-v1.toml"
SCRIPT_PATH = EXPERIMENT_DIR / "scripts" / "generate_report.py"
SOURCE_COMMIT = "1" * 40
REPORT_COMMAND = (
    "uv run python "
    "experiments/01-feedback-control/scripts/generate_report.py"
)


def _evaluation_input(tmp_path: Path) -> Path:
    config_contents = BASE_CONFIG_PATH.read_text(encoding="utf-8")
    config_contents = config_contents.replace(
        'path = "../models/one_link.xml"',
        f'path = "{EXPERIMENT_DIR / "models" / "one_link.xml"}"',
    )
    config_contents = config_contents.replace("duration_s = 4.0", "duration_s = 0.02")
    config_contents = config_contents.replace(
        "steady_state_window_s = 0.50",
        "steady_state_window_s = 0.01",
    )
    config_path = tmp_path / "evaluation.toml"
    config_path.write_text(config_contents, encoding="utf-8")
    input_dir = tmp_path / "evaluation"
    run_evaluation(
        config_path=config_path,
        output_dir=input_dir,
        code_version=SOURCE_COMMIT,
        command="test evaluation command",
    )
    return input_dir


def test_report_is_generated_only_from_validated_saved_records(
    tmp_path: Path,
) -> None:
    input_dir = _evaluation_input(tmp_path)
    output_dir = tmp_path / "report"

    artifacts = generate_report(
        input_dir=input_dir,
        config_path=tmp_path / "evaluation.toml",
        output_dir=output_dir,
        command=REPORT_COMMAND,
    )

    assert len(artifacts) == 9
    assert all(path.is_file() for path in artifacts)
    assert all(
        (output_dir / filename).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        for filename in (
            "position-rad.png",
            "velocity-rad-s.png",
            "position-error-rad.png",
            "controller-output-nm.png",
            "applied-action-nm.png",
        )
    )

    with (output_dir / "episode-metrics.csv").open(
        encoding="utf-8",
        newline="",
    ) as metrics_file:
        metric_rows = list(csv.DictReader(metrics_file))
    assert len(metric_rows) == 8
    assert {(row["controller"], row["scenario_id"]) for row in metric_rows} == {
        (controller, scenario)
        for controller in ("p", "pd")
        for scenario in (
            "negative-to-positive-wide",
            "positive-to-negative-wide",
            "negative-to-positive-narrow",
            "positive-to-negative-narrow",
        )
    }

    with (output_dir / "controller-summary.csv").open(
        encoding="utf-8",
        newline="",
    ) as summary_file:
        summary_rows = list(csv.DictReader(summary_file))
    assert len(summary_rows) == 14
    assert all("population_standard_deviation" in row for row in summary_rows)
    assert all("failed_episode_count" in row for row in summary_rows)

    report = (output_dir / "report.md").read_text(encoding="utf-8")
    assert "## Observed" in report
    assert "## Protocol" in report
    assert "### Gain selection" in report
    assert "### Metric definitions and tolerances" in report
    assert "0.1 to 0.9" in report
    assert "0.05 rad" in report
    assert "0.1 rad/s" in report
    assert "final 0.01 s" in report
    assert "sum of squared applied torque" in report
    assert "## Interpretation" in report
    assert "## Unknowns and limitations" in report
    assert "## Source records" in report
    assert "| P | 2 | 0 |" in report
    assert "| PD | 2 | 1.3 |" in report
    assert "Declared seeds present in the saved records: `[0]`" in report
    assert "trajectories/0000.json" in report
    evaluation_sha256 = hashlib.sha256(
        (input_dir / "evaluation.json").read_bytes()
    ).hexdigest()
    assert evaluation_sha256 in report


def test_report_generation_is_byte_reproducible(tmp_path: Path) -> None:
    input_dir = _evaluation_input(tmp_path)
    output_dir = tmp_path / "report"
    first_artifacts = generate_report(
        input_dir=input_dir,
        config_path=tmp_path / "evaluation.toml",
        output_dir=output_dir,
        command=REPORT_COMMAND,
    )
    first_bytes = {path.name: path.read_bytes() for path in first_artifacts}

    second_artifacts = generate_report(
        input_dir=input_dir,
        config_path=tmp_path / "evaluation.toml",
        output_dir=output_dir,
        command=REPORT_COMMAND,
    )

    assert {path.name: path.read_bytes() for path in second_artifacts} == first_bytes


def test_report_rejects_a_trajectory_checksum_mismatch(tmp_path: Path) -> None:
    input_dir = _evaluation_input(tmp_path)
    trajectory_path = input_dir / "trajectories" / "0000.json"
    trajectory_path.write_text(
        trajectory_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="trajectory checksum mismatch"):
        load_evaluation_evidence(input_dir, tmp_path / "evaluation.toml")


def test_report_rejects_a_configuration_checksum_mismatch(tmp_path: Path) -> None:
    input_dir = _evaluation_input(tmp_path)
    config_path = tmp_path / "evaluation.toml"
    config_path.write_text(
        config_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="configuration checksum mismatch"):
        load_evaluation_evidence(input_dir, config_path)


def test_report_command_consumes_existing_records(tmp_path: Path) -> None:
    input_dir = _evaluation_input(tmp_path)
    output_dir = tmp_path / "cli-report"

    completed = subprocess.run(
        (
            sys.executable,
            str(SCRIPT_PATH),
            "--input",
            str(input_dir),
            "--config",
            str(tmp_path / "evaluation.toml"),
            "--output",
            str(output_dir),
        ),
        check=True,
        capture_output=True,
        text=True,
    )

    assert "generated 9 report artifacts" in completed.stdout
    assert (output_dir / "report.md").is_file()
