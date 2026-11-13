from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import zipfile

import numpy as np
import pytest

from forward_dynamics.types import DynamicsState
from learned_dynamics.data import load_views
from learned_dynamics.normalization import fit_normalizer


ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def checkpoint_normalizer():
    return fit_normalizer(load_views(
        ROOT / "experiments/02-state-transition-dataset/data/dataset-v1",
        "experiment-02-transition-v1").train)


@pytest.fixture
def checkpoint_parts(checkpoint_normalizer):
    import torch
    from learned_dynamics.model import ResidualMLP
    from learned_dynamics.predictor import SUPPORTED_CONTEXT
    from learned_dynamics.types import ModelSpec
    spec = ModelSpec()
    model = ResidualMLP(spec, seed=0)
    norm = checkpoint_normalizer
    metadata = {
        "schema_version": 1, "model_spec": asdict(spec),
        "run_spec": {"variant_id": "canonical32", "model": asdict(spec), "loss": "one_step",
                     "membership": "all", "seed": 0, "device": "cpu", "repeat": 0},
        "context": asdict(SUPPORTED_CONTEXT), "selected_epoch": 1,
        "selection_metric": "validation_normalized_one_step_mse",
        "selected_validation_mse": 0.5, "dataset_version": norm.dataset_version,
        "dataset_manifest_sha256": norm.manifest_sha256, "train_sha256": norm.train_sha256,
        "training_config_sha256": "d" * 64,
        "source_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "working_tree_dirty": True,
        "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                    "numpy": np.__version__, "os": platform.system(),
                    "architecture": platform.machine(), "device": "cpu"},
    }
    return model, norm, metadata


def repair_manifest(path):
    manifest = json.loads((path / "manifest.json").read_text())
    for name in manifest["files"]:
        manifest["files"][name] = hashlib.sha256((path / name).read_bytes()).hexdigest()
    (path / "manifest.json").write_text(json.dumps(manifest))


def test_checkpoint_round_trip_preserves_predictions_and_fit_keys(tmp_path, checkpoint_parts):
    from learned_dynamics.checkpoint import save_checkpoint, load_checkpoint
    from learned_dynamics.predictor import LearnedPredictor, SUPPORTED_CONTEXT
    model, norm, metadata = checkpoint_parts
    dest = tmp_path / "bundle"
    assert save_checkpoint(dest, model, norm, metadata) == dest.resolve()
    restored = load_checkpoint(dest)
    assert restored.normalizer == norm
    assert restored.normalizer.fit_keys == norm.fit_keys
    assert len(norm.fit_keys) == 940
    original = LearnedPredictor(model, norm, SUPPORTED_CONTEXT, "a", "cpu")
    loaded = LearnedPredictor(restored.model, restored.normalizer, SUPPORTED_CONTEXT, "a", "cpu")
    assert original.predict(DynamicsState(.3, -.2), .1, SUPPORTED_CONTEXT) == loaded.predict(
        DynamicsState(.3, -.2), .1, SUPPORTED_CONTEXT)
    assert restored.metadata["training_config_sha256"] == "d" * 64


def test_existing_output_and_source_destinations_are_refused(tmp_path, checkpoint_parts):
    from learned_dynamics.checkpoint import save_checkpoint
    model, norm, meta = checkpoint_parts
    dest = tmp_path / "existing"
    dest.mkdir()
    for path in (dest, ROOT / "experiments/04-learned-dynamics/src/new-checkpoint"):
        with pytest.raises(ValueError):
            save_checkpoint(path, model, norm, meta)
    assert list(dest.iterdir()) == []


@pytest.mark.parametrize("field,value", [
    ("schema_version", True), ("schema_version", 2),
    ("selected_epoch", True), ("selected_epoch", 301),
    ("selected_validation_mse", float("nan")),
    ("train_sha256", "e" * 64), ("source_sha", "f" * 40),
    ("working_tree_dirty", 0), ("training_config_sha256", "wrong"),
    ("tensor_file", "../outside.npz"), ("extra", 1),
])
def test_metadata_corruption_rejected_after_rehash(tmp_path, checkpoint_parts, field, value):
    from learned_dynamics.checkpoint import save_checkpoint, load_checkpoint
    dest = tmp_path / "bundle"
    save_checkpoint(dest, *checkpoint_parts)
    meta = json.loads((dest / "metadata.json").read_text())
    meta[field] = value
    (dest / "metadata.json").write_text(json.dumps(meta))
    repair_manifest(dest)
    with pytest.raises(ValueError):
        load_checkpoint(dest)


@pytest.mark.parametrize("change", ["seed", "architecture", "context", "runtime"])
def test_nested_metadata_must_be_consistent(tmp_path, checkpoint_parts, change):
    from learned_dynamics.checkpoint import save_checkpoint, load_checkpoint
    dest = tmp_path / "bundle"
    save_checkpoint(dest, *checkpoint_parts)
    meta = json.loads((dest / "metadata.json").read_text())
    if change == "seed":
        meta["run_spec"]["seed"] = True
    elif change == "architecture":
        meta["model_spec"]["hidden_width"] = 16
    elif change == "context":
        meta["context"]["control_interval_s"] = .02
    else:
        meta["runtime"]["device"] = "mps"
    (dest / "metadata.json").write_text(json.dumps(meta))
    repair_manifest(dest)
    with pytest.raises(ValueError):
        load_checkpoint(dest)


@pytest.mark.parametrize("change", ["fit_identity", "statistic"])
def test_repaired_outer_hash_does_not_hide_normalizer_mismatch(tmp_path, checkpoint_parts, change):
    from learned_dynamics.checkpoint import save_checkpoint, load_checkpoint
    dest = tmp_path / "bundle"
    save_checkpoint(dest, *checkpoint_parts)
    file = dest / "normalizer.json"
    norm = json.loads(file.read_text())
    if change == "fit_identity":
        norm["fit_keys"][-1][1] = 999
    else:
        norm["input_mean"][0] += .2
    file.write_text(json.dumps(norm))
    meta = json.loads((dest / "metadata.json").read_text())
    meta["normalizer_sha256"] = hashlib.sha256(file.read_bytes()).hexdigest()
    (dest / "metadata.json").write_text(json.dumps(meta))
    repair_manifest(dest)
    with pytest.raises(ValueError, match="normalizer"):
        load_checkpoint(dest)


@pytest.mark.parametrize("change", ["missing", "extra", "shape", "dtype", "nan", "object"])
def test_unsafe_or_incompatible_tensors_rejected(tmp_path, checkpoint_parts, change):
    from learned_dynamics.checkpoint import save_checkpoint, load_checkpoint
    dest = tmp_path / "bundle"
    save_checkpoint(dest, *checkpoint_parts)
    with np.load(dest / "weights.npz", allow_pickle=False) as stored:
        values = {name: stored[name] for name in stored.files}
    key = "network.0.weight"
    if change == "missing":
        del values[key]
    elif change == "extra":
        values["extra"] = np.zeros(1, dtype=np.float32)
    elif change == "shape":
        values[key] = np.zeros((1, 1), dtype=np.float32)
    elif change == "dtype":
        values[key] = values[key].astype(np.float64)
    elif change == "nan":
        values[key].flat[0] = float("nan")
    else:
        values[key] = np.array([object()], dtype=object)
    np.savez(dest / "weights.npz", **values)
    meta = json.loads((dest / "metadata.json").read_text())
    meta["tensor_sha256"] = hashlib.sha256((dest / "weights.npz").read_bytes()).hexdigest()
    (dest / "metadata.json").write_text(json.dumps(meta))
    repair_manifest(dest)
    with pytest.raises(ValueError):
        load_checkpoint(dest)


def test_symlinked_bundle_file_is_rejected(tmp_path, checkpoint_parts):
    from learned_dynamics.checkpoint import save_checkpoint, load_checkpoint
    dest = tmp_path / "bundle"
    save_checkpoint(dest, *checkpoint_parts)
    outside = tmp_path / "outside.json"
    outside.write_bytes((dest / "metadata.json").read_bytes())
    (dest / "metadata.json").unlink()
    (dest / "metadata.json").symlink_to(outside)
    with pytest.raises(ValueError):
        load_checkpoint(dest)


def test_archive_path_and_duplicate_members_are_rejected(tmp_path, checkpoint_parts):
    from learned_dynamics.checkpoint import save_checkpoint, load_checkpoint
    dest = tmp_path / "bundle"
    save_checkpoint(dest, *checkpoint_parts)
    with zipfile.ZipFile(dest / "weights.npz", "a") as archive:
        archive.writestr("../outside.npy", b"not a tensor")
    meta = json.loads((dest / "metadata.json").read_text())
    meta["tensor_sha256"] = hashlib.sha256((dest / "weights.npz").read_bytes()).hexdigest()
    (dest / "metadata.json").write_text(json.dumps(meta))
    repair_manifest(dest)
    with pytest.raises(ValueError):
        load_checkpoint(dest)


def test_oversized_tensor_file_rejected_before_hashing(tmp_path, checkpoint_parts):
    from learned_dynamics.checkpoint import save_checkpoint, load_checkpoint
    dest = tmp_path / "bundle"
    save_checkpoint(dest, *checkpoint_parts)
    with (dest / "weights.npz").open("r+b") as stream:
        stream.truncate(17 * 1024 * 1024)
    with pytest.raises(ValueError, match="size limit"):
        load_checkpoint(dest)


def test_checkpoint_replays_on_actual_mps(tmp_path, checkpoint_parts):
    import torch
    from learned_dynamics.checkpoint import save_checkpoint, load_checkpoint
    from learned_dynamics.predictor import LearnedPredictor, SUPPORTED_CONTEXT
    if not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable; GPU checkpoint replay not established")
    dest = save_checkpoint(tmp_path / "bundle", *checkpoint_parts)
    cpu, gpu = load_checkpoint(dest, "cpu"), load_checkpoint(dest, "mps")
    state, action = DynamicsState(.3, -.2), .1
    a = LearnedPredictor(cpu.model, cpu.normalizer, SUPPORTED_CONTEXT, "cpu", "cpu").predict(state, action, SUPPORTED_CONTEXT)
    b = LearnedPredictor(gpu.model, gpu.normalizer, SUPPORTED_CONTEXT, "mps", "mps").predict(state, action, SUPPORTED_CONTEXT)
    assert abs(a.position_rad - b.position_rad) <= 1e-6
    assert abs(a.velocity_rad_s - b.velocity_rad_s) <= 1e-5
