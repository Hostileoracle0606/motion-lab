"""Convert independently validated data into explicit fitting/evaluation views."""

from pathlib import Path

import numpy as np

from forward_dynamics.dataset import load_prediction_cases
from forward_dynamics.serialization import sha256_file
from learned_dynamics.types import DataViews, SplitView, TrainingView


def load_views(dataset_dir: Path, expected_dataset_version: str) -> DataViews:
    """Validate every original partition before exposing checked views."""
    directory = Path(dataset_dir).resolve()
    cases = load_prediction_cases(directory, expected_dataset_version)
    manifest_sha = sha256_file(directory / "manifest.json")
    train_sha = sha256_file(directory / "train.jsonl")
    views = {
        split: SplitView(split, tuple(c for c in cases if c.split == split),
                         expected_dataset_version, manifest_sha, train_sha)
        for split in ("train", "validation", "test")
    }
    return DataViews(TrainingView(views["train"]), views["validation"], views["test"])


def training_arrays(view: TrainingView) -> tuple[np.ndarray, np.ndarray]:
    """Return physical float64 [N,3] inputs and [N,2] deltas; no metadata features."""
    if not isinstance(view, TrainingView):
        raise ValueError("training_arrays requires a TrainingView")
    inputs = np.array([
        (c.state.position_rad, c.state.velocity_rad_s, c.applied_torque_nm)
        for c in view.cases
    ], dtype=np.float64)
    deltas = np.array([
        (float(c.expected_next_state.position_rad) - float(c.state.position_rad),
         float(c.expected_next_state.velocity_rad_s) - float(c.state.velocity_rad_s))
        for c in view.cases
    ], dtype=np.float64)
    if not np.isfinite(inputs).all() or not np.isfinite(deltas).all():
        raise ValueError("training arrays must be finite")
    return inputs, deltas
