"""Audit first, stage all derived views, then publish a fresh report directory."""

from pathlib import Path
import tempfile

from forward_dynamics.config import load_evaluation_config
from forward_dynamics.evidence import load_evaluation_evidence
from forward_dynamics.generation import _validate_output
from forward_dynamics.provenance import input_path
from forward_dynamics.report_markdown import report_markdown
from forward_dynamics.report_plots import write_plots
from forward_dynamics.report_tables import write_tables


def generate_report(input_dir: Path, output_dir: Path) -> Path:
    """Return report.md; saved evidence and all participating inputs are protected."""
    source, output = input_dir.resolve(), output_dir.resolve()
    if source == output or source.is_relative_to(output):
        raise ValueError("report output must not replace or contain evidence")
    if output.exists():
        raise ValueError("report output already exists; choose a fresh path")
    evidence = load_evaluation_evidence(source)
    if evidence.evaluation["scope"] != "full":
        raise ValueError("report requires full six-adapter comparison evidence")
    config_path = input_path(evidence.evaluation["provenance"]["config_file"])
    _validate_output(output, config_path, load_evaluation_config(config_path))
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".report-staging-", dir=output.parent) as temporary:
        stage = Path(temporary)
        write_tables(evidence, stage)
        write_plots(evidence, stage)
        (stage / "report.md").write_text(report_markdown(evidence), encoding="utf-8")
        if output.exists():
            raise ValueError("report output was created during generation")
        stage.rename(output)
    return output / "report.md"
