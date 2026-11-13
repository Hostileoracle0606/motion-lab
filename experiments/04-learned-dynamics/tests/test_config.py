from pathlib import Path

import pytest

from learned_dynamics.config import load_training_config


ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "experiments/04-learned-dynamics/configs/training-v1.toml"


def test_committed_config_freezes_global_protocol_and_45_run_roster():
    config = load_training_config(CONFIG)
    assert config.schema_version == 1
    assert config.dataset_version == "experiment-02-transition-v1"
    assert config.epochs == 300
    assert config.batch_size == 64
    assert config.optimizer.learning_rate == 0.001
    assert config.optimizer.betas == (0.9, 0.999)
    assert config.optimizer.epsilon == 1e-8
    assert config.optimizer.weight_decay == 0.0
    assert config.sampler_seed_offset == 100000
    assert config.selection_metric == "validation_normalized_one_step_mse"
    assert config.tie_break == "earliest_epoch"
    assert config.artifact_budget_bytes == 500 * 1024 * 1024
    assert config.resume == "reject"
    assert len(config.runs) == 45
    assert len({run.run_id for run in config.runs}) == 45
    assert {run.seed for run in config.runs} == {0, 1, 2, 3, 4}
    assert sum(run.variant_id == "canonical32" for run in config.runs) == 20


def test_paths_resolve_from_config_not_current_working_directory(tmp_path, monkeypatch,
                                                                 training_config_writer):
    path = training_config_writer(tmp_path / "training.toml")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    config = load_training_config(path)
    assert config.dataset_dir == (tmp_path / "dataset").resolve()
    assert config.output_dir == (tmp_path / "results").resolve()


@pytest.mark.parametrize("field,value", [("epochs", True), ("seeds", (False, 1))])
def test_boolean_counts_and_seeds_are_rejected(tmp_path, field, value, training_config_writer):
    kwargs = {field: value}
    path = training_config_writer(tmp_path / "training.toml", **kwargs)
    with pytest.raises(ValueError, match="integer"):
        load_training_config(path)


@pytest.mark.parametrize("output_dir", ["dataset", ".", "src"])
def test_output_cannot_overlap_inputs_or_source(tmp_path, output_dir, training_config_writer):
    (tmp_path / "src").mkdir()
    path = training_config_writer(tmp_path / "training.toml", output_dir=output_dir)
    with pytest.raises(ValueError, match="overlap"):
        load_training_config(path)


def test_unknown_fields_and_duplicate_run_ids_are_rejected(tmp_path, training_config_writer):
    path = training_config_writer(tmp_path / "training.toml", seeds=(0, 0))
    with pytest.raises(ValueError, match="duplicate"):
        load_training_config(path)
    text = training_config_writer(path, seeds=(0, 1)).read_text()
    path.write_text(text.replace("schema_version = 1", "schema_version = 1\nunknown = 2"))
    with pytest.raises(ValueError, match="fields"):
        load_training_config(path)


def test_paths_do_not_expand_home_or_environment(tmp_path, training_config_writer):
    path = training_config_writer(tmp_path / "training.toml")
    text = path.read_text().replace('dataset_dir = "dataset"', 'dataset_dir = "~/dataset"')
    path.write_text(text)
    with pytest.raises(ValueError, match="relative path"):
        load_training_config(path)
