"""Publication orchestration for the versioned state-transition dataset."""

import hashlib
from pathlib import Path
import shlex
import subprocess

from state_transition_dataset.builder import (
    build_episode_manifests,
    build_rows,
)
from state_transition_dataset.config import load_dataset_config
from state_transition_dataset.schema import (
    DATASET_SCHEMA_VERSION,
    DatasetMetadata,
    ValidatedDataset,
)
from state_transition_dataset.validation import (
    load_and_validate_dataset,
    write_dataset,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG = (
    REPOSITORY_ROOT
    / "experiments/02-state-transition-dataset/configs/dataset-v1.toml"
)
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT
    / "experiments/02-state-transition-dataset/data/dataset-v1"
)
BUILD_SCRIPT = (
    "experiments/02-state-transition-dataset/scripts/build_dataset.py"
)


def build_dataset(
    config_path: Path,
    output_dir: Path,
    *,
    allow_dirty: bool = False,
) -> ValidatedDataset:
    """Build, write, then independently reload one dataset version."""
    builder_code_version = _git_output("rev-parse", "HEAD")
    working_tree_dirty = bool(_git_output("status", "--porcelain"))
    if working_tree_dirty and not allow_dirty:
        raise ValueError(
            "working-tree changes are uncommitted; commit them or pass --allow-dirty"
        )

    resolved_config = config_path.resolve()
    config = load_dataset_config(resolved_config)
    build = build_rows(config)
    episodes = build_episode_manifests(
        build,
        config.replay_absolute_tolerance,
    )
    metadata = DatasetMetadata(
        schema_version=DATASET_SCHEMA_VERSION,
        dataset_version=config.dataset_version,
        dataset_config_file=_repository_relative_or_name(resolved_config),
        dataset_config_sha256=hashlib.sha256(
            resolved_config.read_bytes()
        ).hexdigest(),
        builder_code_version=builder_code_version,
        builder_working_tree_dirty=working_tree_dirty,
        build_command=_recorded_command(resolved_config, allow_dirty),
        source_evaluation_sha256=build.source_evaluation_sha256,
        source_code_version=build.source_code_version,
        replay_absolute_tolerance=config.replay_absolute_tolerance,
        split_policy=(
            "complete scenario assignment; paired controllers and adjacent "
            "transitions remain in one partition"
        ),
        episodes=episodes,
    )
    write_dataset(output_dir, metadata, build.rows_by_split)
    return load_and_validate_dataset(output_dir, config.dataset_version)


def _recorded_command(config_path: Path, allow_dirty: bool) -> str:
    arguments = ["uv", "run", "python", BUILD_SCRIPT]
    if config_path != DEFAULT_CONFIG.resolve():
        arguments.extend(("--config", _repository_relative_or_name(config_path)))
    if allow_dirty:
        arguments.append("--allow-dirty")
    return shlex.join(arguments)


def _repository_relative_or_name(path: Path) -> str:
    try:
        return path.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return path.name


def _git_output(*arguments: str) -> str:
    completed = subprocess.run(
        ("git", "-C", str(REPOSITORY_ROOT), *arguments),
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()
