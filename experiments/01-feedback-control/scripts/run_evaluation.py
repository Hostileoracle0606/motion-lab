"""Run the complete versioned Experiment 01 P-versus-PD evaluation."""

import argparse
from pathlib import Path
import shlex
import subprocess
import sys

from feedback_control.evaluator import run_evaluation


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = (
    REPOSITORY_ROOT
    / "experiments"
    / "01-feedback-control"
    / "configs"
    / "evaluation-v1.toml"
)
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT
    / "experiments"
    / "01-feedback-control"
    / "results"
    / "evaluation-v1"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate P and PD across the same versioned scenarios and seeds."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="run with uncommitted tracked changes and record that state",
    )
    arguments = parser.parse_args()

    code_version = _git_output("rev-parse", "HEAD")
    working_tree_dirty = bool(_git_output("status", "--porcelain"))
    if working_tree_dirty and not arguments.allow_dirty:
        parser.error(
            "tracked changes are uncommitted; commit them or pass --allow-dirty"
        )
    recorded_command = shlex.join(
        (
            "uv",
            "run",
            "python",
            "experiments/01-feedback-control/scripts/run_evaluation.py",
            *sys.argv[1:],
        )
    )
    result = run_evaluation(
        config_path=arguments.config,
        output_dir=arguments.output,
        code_version=code_version,
        command=recorded_command,
        working_tree_dirty=working_tree_dirty,
    )
    print(
        f"evaluated {len(result.episodes)} episodes -> "
        f"{arguments.output.resolve() / 'evaluation.json'}"
    )


def _git_output(*arguments: str) -> str:
    completed = subprocess.run(
        ("git", "-C", str(REPOSITORY_ROOT), *arguments),
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


if __name__ == "__main__":
    main()
