"""Generate tables and plots from independently audited saved predictions."""

import argparse
from pathlib import Path

from forward_dynamics.config import DEFAULT_OUTPUT
from forward_dynamics.reporting import generate_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        path = generate_report(args.input, args.output or args.input / "report")
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(f"audited report saved -> {path}")


if __name__ == "__main__":
    main()
