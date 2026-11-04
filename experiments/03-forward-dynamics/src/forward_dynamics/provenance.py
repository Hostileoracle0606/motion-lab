"""Checkout-relative input identity and auditable evaluator provenance."""

import hashlib
from importlib.metadata import version
import platform
from pathlib import Path
import re
import shlex
import subprocess

from state_transition_dataset.validation import load_and_validate_dataset

from forward_dynamics.config import EvaluationConfig, REPOSITORY_ROOT
from forward_dynamics.serialization import require_equal, require_fields, sha256_file


SCRIPT = "experiments/03-forward-dynamics/scripts/run_evaluation.py"


def git_bytes(*arguments: str) -> bytes:
    try:
        result = subprocess.run(
            ("git", "-C", str(REPOSITORY_ROOT), *arguments),
            check=True, capture_output=True,
        )
    except subprocess.CalledProcessError as error:
        raise ValueError("evaluator commit/input is not available in this checkout") from error
    return result.stdout


def git_output(*arguments: str) -> str:
    return git_bytes(*arguments).decode("utf-8").strip()


def relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError as error:
        raise ValueError("evidence inputs must be inside the evaluator checkout") from error


def input_path(value: str) -> Path:
    if (not isinstance(value, str) or not value or Path(value).is_absolute() or
            ".." in Path(value).parts):
        raise ValueError("invalid repository-relative input path")
    path = (REPOSITORY_ROOT / value).resolve()
    if relative_path(path) != value:
        raise ValueError("input path must be normalized and stay inside the checkout")
    return path


def input_entries(config_path: Path, config: EvaluationConfig) -> list[dict]:
    paths = {config_path.resolve(), config.model_path, REPOSITORY_ROOT / "uv.lock",
             REPOSITORY_ROOT / "pyproject.toml", config.dataset_dir / "manifest.json"}
    paths.update(config.dataset_dir / f"{split}.jsonl" for split in ("train", "validation", "test"))
    # Capture every experiment source module that may participate in input
    # validation or prediction, not only the top-level script's Git label.
    for experiment in ("01-feedback-control", "02-state-transition-dataset", "03-forward-dynamics"):
        paths.update((REPOSITORY_ROOT / "experiments" / experiment / "src").rglob("*.py"))
    paths.add(REPOSITORY_ROOT / SCRIPT)
    return [{"path": relative_path(path), "sha256": sha256_file(path)}
            for path in sorted(paths, key=relative_path)]


def recorded_command(config_file: str, allow_dirty: bool) -> str:
    command = ["uv", "run", "python", SCRIPT, "--config", config_file, "--output", "<output-dir>"]
    if allow_dirty:
        command.append("--allow-dirty")
    return shlex.join(command)


def capture_provenance(config_path: Path, config: EvaluationConfig, allow_dirty: bool) -> dict:
    if type(allow_dirty) is not bool:
        raise ValueError("allow_dirty must be a boolean")
    commit = git_output("rev-parse", "HEAD")
    dirty = bool(git_output("status", "--porcelain", "--untracked-files=all"))
    if dirty and not allow_dirty:
        raise ValueError("working-tree changes are uncommitted; commit them or pass --allow-dirty")
    dataset = load_and_validate_dataset(config.dataset_dir, config.dataset_version)
    config_file = relative_path(config_path)
    return {
        "evaluator_commit": commit, "working_tree_dirty": dirty, "allow_dirty": allow_dirty,
        "config_file": config_file, "command": recorded_command(config_file, allow_dirty),
        "dataset_source_commit": dataset.manifest.metadata.source_code_version,
        "dataset_builder_commit": dataset.manifest.metadata.builder_code_version,
        "inputs": input_entries(config_path, config),
        "runtime": {"python": platform.python_version(), "mujoco": version("mujoco"),
                    "os": platform.system(), "architecture": platform.machine()},
    }


def validate_provenance(payload: dict, config_path: Path, config: EvaluationConfig) -> None:
    require_fields(payload, {
        "evaluator_commit", "working_tree_dirty", "allow_dirty", "config_file", "command",
        "dataset_source_commit", "dataset_builder_commit", "inputs", "runtime",
    }, "provenance")
    for field in ("working_tree_dirty", "allow_dirty"):
        if type(payload[field]) is not bool:
            raise ValueError(f"{field} must be a boolean")
    if payload["working_tree_dirty"] and not payload["allow_dirty"]:
        raise ValueError("dirty provenance requires explicit allow_dirty")
    commit = payload["evaluator_commit"]
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("invalid evaluator commit")
    git_output("cat-file", "-e", commit + "^{commit}")
    dataset = load_and_validate_dataset(config.dataset_dir, config.dataset_version)
    require_equal(payload["dataset_source_commit"], dataset.manifest.metadata.source_code_version, "dataset source provenance")
    require_equal(payload["dataset_builder_commit"], dataset.manifest.metadata.builder_code_version, "dataset builder provenance")
    require_equal(payload["config_file"], relative_path(config_path), "config file identity")
    require_equal(payload["command"], recorded_command(payload["config_file"], payload["allow_dirty"]), "recorded command")
    require_equal(payload["inputs"], input_entries(config_path, config), "input checksums")
    if not payload["working_tree_dirty"]:
        for entry in payload["inputs"]:
            committed = git_bytes("show", f"{commit}:{entry['path']}")
            if hashlib.sha256(committed).hexdigest() != entry["sha256"]:
                raise ValueError("clean evaluator commit contradicts input bytes")
    runtime = payload["runtime"]
    require_fields(runtime, {"python", "mujoco", "os", "architecture"}, "runtime")
    if (any(not isinstance(value, str) or not value for value in runtime.values()) or
            not re.fullmatch(r"3\.12\.\d+", runtime["python"]) or runtime["mujoco"] != "3.16.0"):
        raise ValueError("unsupported recorded runtime")
