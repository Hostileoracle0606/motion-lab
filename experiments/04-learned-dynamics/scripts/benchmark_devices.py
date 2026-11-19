#!/usr/bin/env python3
"""Benchmark the frozen canonical checkpoint on CPU and MPS."""

import argparse
import json
import os
from pathlib import Path

from learned_dynamics.benchmarking import benchmark_inference
from learned_dynamics.data import load_views
from learned_dynamics.types import DATASET_VERSION


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record synchronized CPU/MPS inference timing distributions.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--power-state", required=True,
                        help="Observed power/battery state during measurement.")
    parser.add_argument("--competing-load", required=True,
                        help="Observed competing system load during measurement.")
    args = parser.parse_args()
    os.environ["MOTION_LAB_POWER_STATE"] = args.power_state
    os.environ["MOTION_LAB_COMPETING_LOAD"] = args.competing_load
    validation = load_views(args.dataset, DATASET_VERSION).validation
    path = benchmark_inference(args.checkpoint, validation, args.output)
    payload = json.loads(path.read_text())
    print(json.dumps({
        "benchmark": str(path),
        "successful_workloads": sum(row["status"] == "success" for row in payload["results"]),
        "failed_device_rounds": sum(row["status"] == "failed" for row in payload["results"]),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
