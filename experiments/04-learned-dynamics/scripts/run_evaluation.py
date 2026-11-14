#!/usr/bin/env python3
"""Generate Stage 4 evaluation evidence from a fully validated training roster."""

import argparse
import json
from pathlib import Path

from learned_dynamics.evaluation import run_evaluation
from learned_dynamics.evidence import load_evidence


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen Stage 4 learned-dynamics evaluation.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-final-test", action="store_true",
                        help="Issue #29 gate: permit the full learned final-test evaluation.")
    args = parser.parse_args()
    if not args.allow_final_test:
        parser.error("full final-test evaluation is deferred to Issue #29; pass --allow-final-test there")
    output = run_evaluation(args.config, args.roster, args.output)
    evidence = load_evidence(output)
    print(json.dumps({"manifest_dir": str(output), "records": evidence["record_count"]},
                     sort_keys=True))


if __name__ == "__main__":
    main()
