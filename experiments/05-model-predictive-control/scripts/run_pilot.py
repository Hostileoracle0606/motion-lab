"""Run the nominal paired pilot and save its evidence in a fresh directory."""

import argparse
from dataclasses import asdict
import hashlib
import platform
from pathlib import Path
import subprocess
import tomllib

import mujoco
import numpy as np

from model_predictive_control.contracts import NOMINAL_CONTEXT, PHYSICS_INTERVAL_S
from model_predictive_control.pilot import run_recorded_pilot
from model_predictive_control.provenance import cpu_model


EXPERIMENT = Path(__file__).resolve().parents[1]
REPOSITORY = EXPERIMENT.parent.parent
MODEL = EXPERIMENT.parent / "01-feedback-control/models/one_link.xml"


def git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPOSITORY), *arguments], text=True,
    ).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=EXPERIMENT / "configs/pilot-v1.toml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deadline-ns", type=int)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output must name a fresh directory")
    if args.deadline_ns is not None and args.deadline_ns < 0:
        parser.error("--deadline-ns must be nonnegative")
    if git("status", "--porcelain"):
        parser.error("commit source changes before recording pilot evidence")
    config_bytes = args.config.read_bytes()
    config = tomllib.loads(config_bytes.decode("utf-8"))
    manifest = {
        "source_commit": git("rev-parse", "HEAD"),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "config": config,
        "scope": ("nominal_optimizer_development" if "cem" in config["mpc"]
                  else "nominal_one_link_development"),
        "truth_model_sha256": hashlib.sha256(MODEL.read_bytes()).hexdigest(),
        "truth_auto_reset": False,
        "predictor_context": asdict(NOMINAL_CONTEXT),
        "physics_interval_s": PHYSICS_INTERVAL_S,
        "predictor_backend": "CPU",
        "mode": "fixed_work" if args.deadline_ns is None else "deadline_discard",
        "deadline_ns": args.deadline_ns,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_model": cpu_model(),
        "python": platform.python_version(),
        "mujoco": mujoco.__version__,
        "numpy": np.__version__,
    }
    count = run_recorded_pilot(
        args.output, config, MODEL, manifest, deadline_ns=args.deadline_ns,
    )
    print(f"Saved {count} development episodes to {args.output}")


if __name__ == "__main__":
    main()
