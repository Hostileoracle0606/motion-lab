"""Generate Experiment 01 tables and plots from saved evaluation records."""

import argparse
from pathlib import Path
import shlex
import sys

from feedback_control.reporting import generate_report


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = (
    REPOSITORY_ROOT
    / "experiments"
    / "01-feedback-control"
    / "results"
    / "evaluation-v1"
)
DEFAULT_OUTPUT = DEFAULT_INPUT / "report"
DEFAULT_CONFIG = (
    REPOSITORY_ROOT
    / "experiments"
    / "01-feedback-control"
    / "configs"
    / "evaluation-v1.toml"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    command = shlex.join(
        (
            "uv",
            "run",
            "python",
            "experiments/01-feedback-control/scripts/generate_report.py",
            *sys.argv[1:],
        )
    )
    artifacts = generate_report(
        input_dir=arguments.input,
        config_path=arguments.config,
        output_dir=arguments.output,
        command=command,
    )
    print(
        f"generated {len(artifacts)} report artifacts -> "
        f"{arguments.output.resolve()}"
    )


if __name__ == "__main__":
    main()
