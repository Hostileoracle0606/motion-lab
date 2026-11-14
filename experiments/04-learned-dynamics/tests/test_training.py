from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch

from forward_dynamics.types import DynamicsState
from learned_dynamics.config import load_training_config
from learned_dynamics.data import load_views
from learned_dynamics.training import (
    fit_model, select_earliest_epoch, train, train_run, training_membership,
)


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "experiments/02-state-transition-dataset/data/dataset-v1"


def test_synthetic_training_loss_falls_and_earliest_tie_wins(tmp_path, training_config_writer,
                                                              synthetic_views):
    config = load_training_config(training_config_writer(tmp_path / "training.toml", epochs=20, seeds=(0,)))
    train, validation, _ = synthetic_views
    outcome = fit_model(config, config.runs[0], train, validation)
    assert outcome.curve[-1]["training_loss"] < outcome.curve[0]["training_loss"]
    assert select_earliest_epoch((0.5, 0.25, 0.25, 0.4)) == (2, 0.25)


def test_fixed_seed_and_cpu_permutations_repeat_without_state_reuse(tmp_path, training_config_writer,
                                                                    synthetic_views):
    config = load_training_config(training_config_writer(tmp_path / "training.toml", seeds=(0,), epochs=3))
    train, validation, _ = synthetic_views
    previous_determinism = torch.are_deterministic_algorithms_enabled()
    first = fit_model(config, config.runs[0], train, validation)
    second = fit_model(config, config.runs[0], train, validation)
    assert first.deterministic_algorithms_requested is True
    assert torch.are_deterministic_algorithms_enabled() is previous_determinism
    assert [{k: v for k, v in row.items() if k != "training_seconds"} for row in first.curve] == [
        {k: v for k, v in row.items() if k != "training_seconds"} for row in second.curve]
    assert all(torch.equal(first.model.state_dict()[key], second.model.state_dict()[key])
               for key in first.model.state_dict())
    assert first.model is not second.model


def test_training_api_cannot_access_test_view(tmp_path, training_config_writer, synthetic_views):
    config = load_training_config(training_config_writer(tmp_path / "training.toml", seeds=(0,)))
    train, _, test = synthetic_views
    with pytest.raises(ValueError, match="validation"):
        train_run(config, config.runs[0], train, test, config.output_dir)


def test_nonfinite_training_is_retained_as_failed_run(tmp_path, training_config_writer, synthetic_views):
    config = load_training_config(training_config_writer(tmp_path / "training.toml", seeds=(0,)))
    train, validation, _ = synthetic_views
    broken_case = replace(train.cases[0], state=DynamicsState(1e308, 0.0),
                          expected_next_state=DynamicsState(-1e308, 0.0),
                          truncated=True, termination_reason="numerical_fixture")
    broken = replace(train, view=replace(train.view, cases=(broken_case,)))
    result = train_run(config, config.runs[0], broken, validation, config.output_dir)
    assert result.status == "failed"
    assert "finite" in result.failure_reason or "range" in result.failure_reason
    evidence = json.loads((config.output_dir / result.run_id / "failure.json").read_text())
    assert evidence["status"] == "failed"


def test_later_epoch_failure_retains_completed_curve(tmp_path, training_config_writer,
                                                      synthetic_views, monkeypatch):
    import learned_dynamics.training as training_module

    config = load_training_config(training_config_writer(
        tmp_path / "training.toml", seeds=(0,), epochs=3))
    train, validation, _ = synthetic_views
    original = training_module._validation_loss
    calls = 0

    def fail_on_second_epoch(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("synthetic later-epoch failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(training_module, "_validation_loss", fail_on_second_epoch)
    result = train_run(config, config.runs[0], train, validation, config.output_dir)
    assert result.status == "failed"
    curve = json.loads((config.output_dir / result.run_id / "learning-curve.json").read_text())
    assert [row["epoch"] for row in curve["epochs"]] == [1]
    assert curve["selected_epoch"] is None
    failure = json.loads((config.output_dir / result.run_id / "failure.json").read_text())
    assert failure["completed_epochs"] == 1


def test_checkpoint_publication_failure_retains_completed_curve(
        tmp_path, training_config_writer, synthetic_views, monkeypatch):
    import learned_dynamics.training as training_module

    config = load_training_config(training_config_writer(
        tmp_path / "training.toml", seeds=(0,), epochs=1))
    train, validation, _ = synthetic_views

    def fail_checkpoint(*args, **kwargs):
        raise ValueError("synthetic checkpoint publication failure")

    monkeypatch.setattr(training_module, "save_checkpoint", fail_checkpoint)
    result = train_run(config, config.runs[0], train, validation, config.output_dir)
    assert result.status == "failed"
    curve = json.loads((config.output_dir / result.run_id / "learning-curve.json").read_text())
    assert [row["epoch"] for row in curve["epochs"]] == [1]
    failure = json.loads((config.output_dir / result.run_id / "failure.json").read_text())
    assert failure["completed_epochs"] == 1


def test_diagnostic_pairs_share_starts_and_update_counts(tmp_path, committed_config_writer):
    config = load_training_config(committed_config_writer(tmp_path / "training.toml", epochs=1,
                                                          seeds=(0,)))
    views = load_views(DATA, config.dataset_version)
    canonical = config.runs[0]
    history0 = replace(canonical, variant_id="history0", membership="history_common")
    history1 = replace(history0, variant_id="history1", model=replace(history0.model, history_steps=1))
    loss1 = replace(canonical, variant_id="loss1", membership="loss_common")
    loss5 = replace(loss1, variant_id="loss5", loss="rollout_5")
    h0 = training_membership(views.train, history0, config.batch_size)
    h1 = training_membership(views.train, history1, config.batch_size)
    l1 = training_membership(views.train, loss1, config.batch_size)
    l5 = training_membership(views.train, loss5, config.batch_size)
    key = lambda membership: tuple((w.cases[0].episode_id, w.cases[0].step_index) for w in membership.windows)
    assert key(h0) == key(h1)
    assert key(l1) == key(l5)
    assert h0.updates_per_epoch == h1.updates_per_epoch
    assert l1.updates_per_epoch == l5.updates_per_epoch
    assert len(h0.windows) == 936
    assert len(l1.windows) == 924


def test_repeat_zero_and_one_are_independent_declared_runs_not_best_repeat(tmp_path, committed_config_writer):
    config = load_training_config(committed_config_writer(tmp_path / "training.toml", epochs=1,
                                                          seeds=(0,)))
    repeat0 = config.runs[0]
    repeat1 = replace(repeat0, repeat=1)
    assert repeat0.run_id != repeat1.run_id
    assert repeat0.repeat == 0


def test_full_roster_execution_requires_task9_gate():
    config = ROOT / "experiments/04-learned-dynamics/configs/training-v1.toml"
    loaded = load_training_config(config)
    with pytest.raises(ValueError, match="45-run study"):
        train(config, loaded.output_dir)


def test_prepublication_source_state_is_reused_after_outputs_exist(tmp_path, committed_config_writer,
                                                                   monkeypatch):
    import learned_dynamics.training as training_module

    config_path = committed_config_writer(tmp_path / "training.toml", epochs=1, seeds=(0,))
    source = "f99d383b8aa1e2ecccc439208180094b57d394e0"
    observations = iter(((source, False), (source, True)))
    monkeypatch.setattr(training_module, "_source_state", lambda: next(observations))
    output = train(config_path, tmp_path / "results")
    manifest = json.loads((output / "run-manifest.json").read_text())
    assert manifest["publication_status"] == "clean_source"
    checkpoint_metadata = json.loads(next(output.glob("*/checkpoint/metadata.json")).read_text())
    assert checkpoint_metadata["working_tree_dirty"] is False
