"""Evaluate the frozen prediction protocol and independently audit evidence."""

import argparse
from pathlib import Path

from forward_dynamics.config import DEFAULT_CONFIG, DEFAULT_OUTPUT
from forward_dynamics.generation import run_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--allow-dirty", action="store_true",
                        help="allow uncommitted inputs and label their provenance")
    args = parser.parse_args()
    try:
        manifest = run_evaluation(args.config, args.output, allow_dirty=args.allow_dirty)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(f"prediction evaluation saved and validated -> {manifest}")


if __name__ == "__main__":
    main()
