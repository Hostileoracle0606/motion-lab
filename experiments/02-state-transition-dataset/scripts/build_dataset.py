"""Build the versioned Experiment 02 state-transition dataset."""

import argparse
from pathlib import Path

from state_transition_dataset.generation import (
    DEFAULT_CONFIG,
    DEFAULT_OUTPUT,
    build_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build validated scenario-level train, validation, and test partitions."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="build from uncommitted changes and record that state",
    )
    arguments = parser.parse_args()
    try:
        dataset = build_dataset(
            arguments.config,
            arguments.output,
            allow_dirty=arguments.allow_dirty,
        )
    except ValueError as error:
        parser.error(str(error))
    row_count = sum(
        partition.row_count for partition in dataset.manifest.partitions
    )
    print(
        f"built {row_count} transitions across "
        f"{len(dataset.manifest.metadata.episodes)} episodes -> "
        f"{arguments.output.resolve() / 'manifest.json'}"
    )


if __name__ == "__main__":
    main()
