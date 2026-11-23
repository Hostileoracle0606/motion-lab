"""Audit episode attempt evidence; use --forensic for incomplete runs."""

import argparse
import json
from pathlib import Path
import sys

from model_predictive_control.attempts import audit_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--forensic", action="store_true")
    args = parser.parse_args()
    result = audit_run(args.output, require_complete=False)
    print(json.dumps(result, indent=2))
    if not args.forensic:
        try:
            audit_run(args.output)
        except ValueError:
            sys.exit(1)


if __name__ == "__main__":
    main()
