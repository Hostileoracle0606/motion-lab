"""Verify that every declared frozen comparison run completed exactly once."""

import argparse
from pathlib import Path

from model_predictive_control.predictor_comparison import audit_comparison_runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-manifest", type=Path, required=True)
    parser.add_argument("outputs", type=Path, nargs="+")
    args = parser.parse_args()
    runs = audit_comparison_runs(args.frozen_manifest, args.outputs)
    print("Audited frozen runs:", ", ".join(runs))


if __name__ == "__main__":
    main()
