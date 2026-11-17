#!/usr/bin/env python3
"""Run the frozen validation or one-time final-test phase."""

import argparse
import json
import os
from pathlib import Path

from learned_dynamics.studies import run_studies


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen Experiment 04 study matrix.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("validation", "final_test"), required=True)
    parser.add_argument("--power-state",
                        help="Observed power/battery state; required for validation benchmark.")
    parser.add_argument("--competing-load",
                        help="Observed competing system load; required for validation benchmark.")
    args = parser.parse_args()
    if args.phase == "validation" and (not args.power_state or not args.competing_load):
        parser.error("validation requires --power-state and --competing-load")
    if args.power_state:
        os.environ["MOTION_LAB_POWER_STATE"] = args.power_state
    if args.competing_load:
        os.environ["MOTION_LAB_COMPETING_LOAD"] = args.competing_load
    path = run_studies(args.config, args.output, args.phase)
    print(json.dumps({"phase": args.phase, "manifest": str(path)}, sort_keys=True))


if __name__ == "__main__":
    main()
