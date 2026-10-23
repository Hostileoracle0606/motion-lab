"""Public orchestration boundary for Experiment 01 report generation."""

from pathlib import Path

from feedback_control.report_evidence import (
    EvaluationEvidence,
    load_evaluation_evidence,
)
from feedback_control.report_markdown import write_markdown_report
from feedback_control.report_plots import write_report_plots
from feedback_control.report_tables import write_report_tables


def generate_report(
    *,
    input_dir: Path,
    config_path: Path,
    output_dir: Path,
    command: str,
) -> tuple[Path, ...]:
    """Generate all report artifacts without running the simulator."""
    if not command.strip():
        raise ValueError("command must not be empty")
    evidence = load_evaluation_evidence(input_dir, config_path)
    resolved_output = output_dir.resolve()
    resolved_output.mkdir(parents=True, exist_ok=True)

    metrics_path, summary_path, outcomes_path, summary_rows = write_report_tables(
        resolved_output,
        evidence.episodes,
    )
    plot_paths = write_report_plots(resolved_output, evidence.episodes)
    report_path = resolved_output / "report.md"
    write_markdown_report(
        report_path,
        evidence=evidence,
        command=command,
        summary_rows=summary_rows,
        plot_paths=plot_paths,
    )
    return (report_path, metrics_path, summary_path, outcomes_path, *plot_paths)


__all__ = ["EvaluationEvidence", "generate_report", "load_evaluation_evidence"]
