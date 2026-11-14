#!/usr/bin/env python3
"""Replay learned predictions from their declared immutable checkpoints."""

import argparse
import json
from pathlib import Path

from learned_dynamics.replay import replay_checkpoints


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify learned records by checkpoint replay.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    args = parser.parse_args()
    print(json.dumps(replay_checkpoints(args.input, args.device), sort_keys=True))


if __name__ == "__main__":
    main()
