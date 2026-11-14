from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import platform

import numpy as np
import pytest
import torch

from learned_dynamics.checkpoint import load_checkpoint, save_checkpoint
from learned_dynamics.config import load_training_config
from learned_dynamics.data import load_views
from learned_dynamics.model import ResidualMLP
from learned_dynamics.normalization import fit_normalizer
from learned_dynamics.predictor import SUPPORTED_CONTEXT
from learned_dynamics.types import ModelSpec
from learned_dynamics.training_evidence import (
    RunResult, load_run_manifest, write_run_manifest,
)


def failed(run_id, reason="synthetic failure"):
    return RunResult(run_id=run_id, status="failed", checkpoint_dir=None,
                     checkpoint_sha256=None, selected_epoch=None,
                     selected_validation_mse=None, failure_reason=reason)


def provenance():
    return {
        "source_sha": "a" * 40,
        "working_tree_dirty": True,
        "dataset_manifest_sha256": "b" * 64,
        "train_sha256": "c" * 64,
        "normalizer_sha256": "d" * 64,
        "environment_sha256": "e" * 64,
        "actual_command": ["python", "train.py", "--smoke"],
    }


def repair_outer_hash(output):
    manifest = json.loads((output / "manifest.json").read_text())
    manifest["files"] = {
        str(path.relative_to(output)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
    (output / "manifest.json").write_text(json.dumps(manifest))


def test_complete_failed_roster_round_trips_and_is_marked_exploratory(tmp_path, training_config_writer):
    config = load_training_config(training_config_writer(tmp_path / "training.toml"))
    results = tuple(failed(run.run_id) for run in config.runs)
    path = write_run_manifest(config.output_dir, config, results, provenance())
    loaded = load_run_manifest(path)
    assert loaded["publication_status"] == "exploratory_dirty_source"
    assert [row["run_id"] for row in loaded["results"]] == [run.run_id for run in config.runs]
    assert all(row["status"] == "failed" for row in loaded["results"])
    assert loaded["provenance"]["actual_command"] == ["python", "train.py", "--smoke"]


def test_missing_duplicate_or_best_seed_only_results_are_rejected(tmp_path, training_config_writer):
    config = load_training_config(training_config_writer(tmp_path / "training.toml"))
    complete = tuple(failed(run.run_id) for run in config.runs)
    with pytest.raises(ValueError, match="complete planned roster"):
        write_run_manifest(config.output_dir, config, complete[:1], provenance())
    with pytest.raises(ValueError, match="duplicate"):
        write_run_manifest(config.output_dir, config, (complete[0], complete[0]), provenance())


def test_manifest_refuses_overwrite_and_budget_excess(tmp_path, training_config_writer):
    config = load_training_config(training_config_writer(tmp_path / "training.toml"))
    results = tuple(failed(run.run_id) for run in config.runs)
    write_run_manifest(config.output_dir, config, results, provenance())
    with pytest.raises(ValueError, match="exists"):
        write_run_manifest(config.output_dir, config, results, provenance())
    config = replace(config, output_dir=(tmp_path / "small"))
    config.output_dir.mkdir()
    with (config.output_dir / "oversized.bin").open("wb") as stream:
        stream.truncate(config.artifact_budget_bytes)
    with pytest.raises(ValueError, match="budget"):
        write_run_manifest(config.output_dir, config, results, provenance())


def test_manifest_rejects_symlinked_existing_artifacts(tmp_path, training_config_writer):
    config = load_training_config(training_config_writer(tmp_path / "training.toml"))
    config.output_dir.mkdir()
    outside = tmp_path / "outside"
    outside.write_text("not owned by this run")
    (config.output_dir / "linked").symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        write_run_manifest(config.output_dir, config,
                           tuple(failed(run.run_id) for run in config.runs), provenance())


def test_manifest_inventory_detects_missing_or_changed_run_artifacts(
        tmp_path, training_config_writer):
    config = load_training_config(training_config_writer(tmp_path / "training.toml"))
    artifact = config.output_dir / config.runs[0].run_id / "diagnostic.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text('{"partial": true}')
    output = write_run_manifest(
        config.output_dir, config,
        tuple(failed(run.run_id) for run in config.runs), provenance())
    artifact.write_text('{"partial": false}')
    with pytest.raises(ValueError, match="checksums"):
        load_run_manifest(output)


def test_loader_rejects_tampered_epoch_duplicate_json_hashes_and_schema(tmp_path, training_config_writer):
    config = load_training_config(training_config_writer(tmp_path / "training.toml"))
    results = tuple(failed(run.run_id) for run in config.runs)
    output = write_run_manifest(config.output_dir, config, results, provenance())
    payload = json.loads((output / "run-manifest.json").read_text())
    payload["results"][0]["selected_epoch"] = config.epochs + 1
    payload["results"][0]["status"] = "success"
    payload["results"][0]["checkpoint_dir"] = "x"
    payload["results"][0]["checkpoint_sha256"] = "f" * 64
    payload["results"][0]["selected_validation_mse"] = 1.0
    payload["results"][0]["failure_reason"] = None
    (output / "run-manifest.json").write_text(json.dumps(payload))
    repair_outer_hash(output)
    with pytest.raises(ValueError, match="selected epoch"):
        load_run_manifest(output)

    encoded = json.dumps(payload)
    duplicate = encoded[:-1] + ', "schema_version": 1}'
    (output / "run-manifest.json").write_text(duplicate)
    repair_outer_hash(output)
    with pytest.raises(ValueError, match="invalid evidence JSON"):
        load_run_manifest(output)

    payload["schema_version"] = 2
    (output / "run-manifest.json").write_text(json.dumps(payload))
    repair_outer_hash(output)
    with pytest.raises(ValueError, match="schema"):
        load_run_manifest(output)


def test_loader_rejects_invalid_provenance_hash_even_with_repaired_outer_hash(tmp_path, training_config_writer):
    config = load_training_config(training_config_writer(tmp_path / "training.toml"))
    output = write_run_manifest(config.output_dir, config,
                                tuple(failed(run.run_id) for run in config.runs), provenance())
    payload = json.loads((output / "run-manifest.json").read_text())
    payload["provenance"]["normalizer_sha256"] = "not-a-hash"
    (output / "run-manifest.json").write_text(json.dumps(payload))
    repair_outer_hash(output)
    with pytest.raises(ValueError, match="sha256"):
        load_run_manifest(output)


def test_original_config_snapshot_prevents_concealing_missing_roster_rows(
        tmp_path, training_config_writer):
    config = load_training_config(training_config_writer(tmp_path / "training.toml"))
    output = write_run_manifest(config.output_dir, config,
                                tuple(failed(run.run_id) for run in config.runs), provenance())
    payload = json.loads((output / "run-manifest.json").read_text())
    payload["planned_runs"] = payload["planned_runs"][:1]
    payload["results"] = payload["results"][:1]
    (output / "run-manifest.json").write_text(json.dumps(payload))
    repair_outer_hash(output)
    with pytest.raises(ValueError, match="original training config roster"):
        load_run_manifest(output)


def test_clean_source_manifest_requires_a_resolvable_commit(tmp_path, training_config_writer):
    config = load_training_config(training_config_writer(tmp_path / "training.toml"))
    clean = provenance() | {"working_tree_dirty": False}
    with pytest.raises(ValueError, match="resolve"):
        write_run_manifest(config.output_dir, config,
                           tuple(failed(run.run_id) for run in config.runs), clean)


def test_success_checkpoint_must_match_the_planned_run_identity(tmp_path, training_config_writer):
    root = Path(__file__).resolve().parents[3]
    data = root / "experiments/02-state-transition-dataset/data/dataset-v1"
    path = training_config_writer(tmp_path / "training.toml")
    path.write_text(path.read_text().replace(
        'dataset_dir = "dataset"', f'dataset_dir = "{data}"'))
    # Absolute paths are deliberately rejected, so make the source relative.
    relative = Path(os.path.relpath(data, tmp_path))
    path.write_text(path.read_text().replace(str(data), str(relative)))
    config = load_training_config(path)
    normalizer = fit_normalizer(load_views(data, "experiment-02-transition-v1").train)
    model = ResidualMLP(ModelSpec(), seed=1)
    run = config.runs[0]
    checkpoint = config.output_dir / run.run_id / "checkpoint"
    metadata = {
        "schema_version": 1, "model_spec": asdict(model.spec),
        "run_spec": run.mapping() | {"seed": 1},
        "context": asdict(SUPPORTED_CONTEXT), "selected_epoch": 1,
        "selection_metric": "validation_normalized_one_step_mse",
        "selected_validation_mse": 1.0,
        "dataset_version": normalizer.dataset_version,
        "dataset_manifest_sha256": normalizer.manifest_sha256,
        "train_sha256": normalizer.train_sha256,
        "training_config_sha256": config.config_sha256,
        "source_sha": "f99d383b8aa1e2ecccc439208180094b57d394e0",
        "working_tree_dirty": True,
        "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                    "numpy": np.__version__, "os": platform.system(),
                    "architecture": platform.machine(), "device": "cpu"},
    }
    save_checkpoint(checkpoint, model, normalizer, metadata)
    bundle = load_checkpoint(checkpoint)
    success = RunResult(run.run_id, "success", checkpoint, bundle.bundle_sha256, 1, 1.0, None)
    results = (success, failed(config.runs[1].run_id))
    linked = provenance() | {
        "source_sha": metadata["source_sha"],
        "dataset_manifest_sha256": normalizer.manifest_sha256,
        "train_sha256": normalizer.train_sha256,
        "normalizer_sha256": bundle.metadata["normalizer_sha256"],
        "environment_sha256": SUPPORTED_CONTEXT.environment_version,
    }
    with pytest.raises(ValueError, match="run specification"):
        write_run_manifest(config.output_dir, config, results, linked)
