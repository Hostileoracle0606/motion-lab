"""Declared evaluator identity, distinct from the source that trained weights."""

from importlib.metadata import version
import platform
import re
import sys

from forward_dynamics.provenance import git_output
from forward_dynamics.serialization import require_fields


def capture_evaluation_provenance() -> dict:
    return {
        "source_sha": git_output("rev-parse", "HEAD"),
        "working_tree_dirty": bool(git_output("status", "--porcelain", "--untracked-files=all")),
        "actual_command": list(sys.argv),
        "runtime": {"python": platform.python_version(), "torch": version("torch"),
                    "numpy": version("numpy"), "mujoco": version("mujoco"),
                    "os": platform.system(), "architecture": platform.machine()},
    }


def validate_evaluation_provenance(value: object) -> None:
    require_fields(value, {"source_sha", "working_tree_dirty", "actual_command", "runtime"},
                   "evaluation provenance")
    source = value["source_sha"]
    if not isinstance(source, str) or re.fullmatch(r"[0-9a-f]{40}", source) is None:
        raise ValueError("evaluation provenance source must be a full commit SHA")
    git_output("cat-file", "-e", source + "^{commit}")
    if type(value["working_tree_dirty"]) is not bool:
        raise ValueError("evaluation provenance dirty state must be boolean")
    command = value["actual_command"]
    if (not isinstance(command, list) or not command or
            any(not isinstance(item, str) or not item for item in command)):
        raise ValueError("evaluation provenance command must be a nonempty string array")
    runtime = value["runtime"]
    require_fields(runtime, {"python", "torch", "numpy", "mujoco", "os", "architecture"},
                   "evaluation runtime")
    if (any(not isinstance(item, str) or not item for item in runtime.values()) or
            re.fullmatch(r"3\.12\.\d+", runtime["python"]) is None or
            runtime["torch"] != "2.15.0" or runtime["mujoco"] != "3.16.0"):
        raise ValueError("evaluation runtime does not match the supported protocol")
