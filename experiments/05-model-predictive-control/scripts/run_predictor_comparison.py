"""Validate a frozen protocol or run the matched one-link predictor comparison."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import tomllib

import mujoco
import numpy as np
import torch

from forward_dynamics.provenance import git_output
from model_predictive_control.predictor_comparison import (
    execute_comparison, prepare_comparison, verify_frozen_manifest,
)
from model_predictive_control.provenance import cpu_model


EXPERIMENT = Path(__file__).resolve().parents[1]
MODEL = EXPERIMENT.parent / "01-feedback-control/models/one_link.xml"
IMPLEMENTATION_PATHS = (
    "pyproject.toml", "uv.lock",
    "experiments/01-feedback-control/src", "experiments/03-forward-dynamics/src",
    "experiments/02-state-transition-dataset/data/dataset-v1",
    "experiments/04-learned-dynamics/src",
    "experiments/05-model-predictive-control/src",
    "experiments/05-model-predictive-control/scripts/run_predictor_comparison.py",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true", help="write the manifest without executing cases")
    parser.add_argument("--frozen-manifest", type=Path, help="required to run held-out cases")
    parser.add_argument("--mode", choices=("fixed-work", "deadline-discard"), default="fixed-work")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output must be fresh")
    if git_output("status", "--porcelain"):
        parser.error("commit source changes before running the comparison")
    config_bytes = args.config.read_bytes()
    config = tomllib.loads(config_bytes.decode("utf-8"))
    if config["purpose"] == "held_out" and not args.preflight and args.frozen_manifest is None:
        parser.error("running held-out cases requires --frozen-manifest")
    cases, plants, comparators, protocol = prepare_comparison(config, MODEL, args.checkpoint)
    source_commit = git_output("rev-parse", "HEAD")
    protocol.update({
        "config": config,
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "implementation_objects": {
            path: git_output("rev-parse", f"{source_commit}:{path}") for path in IMPLEMENTATION_PATHS
        },
    })
    manifest = {"schema_version": 1, "source_commit": source_commit, "protocol": protocol}
    if args.preflight:
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print(f"Validated {len(cases)} cases without running any episodes")
        return

    deadline = config["deadline_ns"] if args.mode == "deadline-discard" else None
    if args.frozen_manifest is not None:
        locked = args.frozen_manifest.read_bytes()
        verify_frozen_manifest(protocol, json.loads(locked))
        manifest["frozen_manifest_sha256"] = hashlib.sha256(locked).hexdigest()
    manifest.update({
        "mode": args.mode, "deadline_ns": deadline,
        "platform": platform.platform(), "cpu_model": cpu_model(),
        "python": platform.python_version(), "mujoco": mujoco.__version__,
        "numpy": np.__version__, "torch": torch.__version__,
    })
    count = execute_comparison(
        args.output, config, cases, plants, comparators, manifest, deadline,
    )
    print(f"Saved {count} predictor comparison episodes to {args.output}")


if __name__ == "__main__":
    main()
