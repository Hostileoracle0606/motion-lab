#!/usr/bin/env python3
"""Execute a declared training roster; full study execution remains gated to Task 9."""

import argparse
import json
from pathlib import Path

from learned_dynamics.training import train
from learned_dynamics.training_evidence import load_run_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Train declared Experiment 04 dynamics runs from scratch.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-full-study", action="store_true",
                        help="Task 9 gate for the frozen 45-run study; do not use for smoke checks.")
    arguments = parser.parse_args()
    output = train(arguments.config, arguments.output,
                   allow_full_study=arguments.allow_full_study)
    evidence = load_run_manifest(output)
    successful = sum(row["status"] == "success" for row in evidence["results"])
    print(json.dumps({"manifest_dir": str(output),
                      "planned_runs": len(evidence["planned_runs"]),
                      "successful_runs": successful,
                      "failed_runs": len(evidence["results"]) - successful}, sort_keys=True))


if __name__ == "__main__":
    main()
