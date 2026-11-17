#!/usr/bin/env python3
"""Generate report artifacts from audited saved study evidence only."""

import argparse
import json
from pathlib import Path

from learned_dynamics.reporting import generate_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate inference-free Experiment 04 tables, plots, and Markdown.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = generate_report(args.input, args.output)
    print(json.dumps({"report": str(report)}, sort_keys=True))


if __name__ == "__main__":
    main()
