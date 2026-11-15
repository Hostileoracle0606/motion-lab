from copy import deepcopy
from types import SimpleNamespace

import pytest

from learned_dynamics.config import RunSpec
from learned_dynamics.evaluation import EvaluationCandidate, evaluate
from learned_dynamics.replay import replay_checkpoints
from learned_dynamics.types import ModelSpec


class NoChange:
    def predict(self, state, action, context):
        return state


@pytest.fixture
def replay_fixture(
        tmp_path, synthetic_views, monkeypatch):
    cases = synthetic_views[2].cases[:6]
    run = RunSpec("canonical32", ModelSpec(32, 0), "one_step", "all", 0, "mps", 0)
    training_hash, checkpoint_hash = "a" * 64, "b" * 64
    result = {"run_id": run.run_id, "status": "success", "checkpoint_dir": "checkpoint",
              "checkpoint_sha256": checkpoint_hash, "selected_epoch": 2,
              "selected_validation_mse": 0.25, "failure_reason": None}
    roster = {"schema_version": 1, "training_config_sha256": training_hash,
              "planned_runs": [{"run_id": run.run_id, "run_spec": run.mapping()}],
              "results": [result]}
    candidate = EvaluationCandidate.learned(
        predictor_id=run.run_id, predictor=NoChange(), run_spec=run.mapping(),
        checkpoint_dir="checkpoint", checkpoint_sha256=checkpoint_hash,
        selected_epoch=2, selected_validation_mse=0.25,
        training_config_sha256=training_hash)
    records = evaluate(cases, (candidate,), (1, 5, 10, 25, 50, 100), "all")
    evidence = {"training_roster_dir": "training", "dataset_dir": "dataset",
                "dataset_version": "experiment-02-transition-v1",
                "case_ids": [{"split": c.split, "episode_id": c.episode_id,
                              "step_index": c.step_index} for c in cases],
                "roster": roster, "records": records}
    monkeypatch.setattr("learned_dynamics.replay.load_evidence", lambda path: evidence)
    monkeypatch.setattr("learned_dynamics.replay.load_run_manifest", lambda path: roster)
    monkeypatch.setattr("learned_dynamics.replay.load_prediction_cases", lambda path, version: cases)
    bundle = SimpleNamespace(model=object(), normalizer=object(), bundle_sha256=checkpoint_hash)
    monkeypatch.setattr("learned_dynamics.replay.load_checkpoint", lambda path, device: bundle)
    monkeypatch.setattr("learned_dynamics.replay.LearnedPredictor",
                        lambda model, normalizer, context, predictor_id, device: NoChange())
    return tmp_path / "evidence", evidence, bundle


def test_replay_proves_saved_predictions_came_from_declared_checkpoint(replay_fixture):
    path, evidence, _ = replay_fixture
    records = evidence["records"]
    result = replay_checkpoints(path, "cpu")
    assert result["replayed_candidates"] == 1
    assert result["replayed_records"] == len(records)
    assert result["status"] == "verified"

    changed = list(deepcopy(records))
    row = next(item for item in changed if item["status"] == "success")
    row["predicted_endpoint_state"]["position_rad"] += 0.01
    evidence["records"] = tuple(changed)
    with pytest.raises(ValueError, match="checkpoint replay"):
        replay_checkpoints(path, "cpu")


def test_replay_rejects_different_actual_bundle_even_with_identical_predictions(replay_fixture):
    path, _, bundle = replay_fixture
    bundle.bundle_sha256 = "c" * 64
    with pytest.raises(ValueError, match="checkpoint.*hash"):
        replay_checkpoints(path, "cpu")


def test_replay_accepts_only_declared_component_tolerances(replay_fixture, monkeypatch):
    path, _, _ = replay_fixture

    class RoundedPredictor:
        def predict(self, state, action, context):
            from forward_dynamics.types import DynamicsState
            return DynamicsState(state.position_rad + 1e-8, state.velocity_rad_s + 1e-7)

    monkeypatch.setattr("learned_dynamics.replay.LearnedPredictor", lambda *args: RoundedPredictor())
    result = replay_checkpoints(path, "cpu")
    assert result["status"] == "verified"
    assert result["comparison_scope"] == "cross_device_diagnostic"


def test_replay_ignores_diagnostics_but_keeps_final_failed_runs_in_evidence(replay_fixture):
    path, evidence, _ = replay_fixture
    roster = evidence["roster"]
    diagnostic = RunSpec("canonical32", ModelSpec(), "one_step", "all", 0, "cpu", 1)
    roster["planned_runs"].append({"run_id": diagnostic.run_id, "run_spec": diagnostic.mapping()})
    result = dict(roster["results"][0], run_id=diagnostic.run_id)
    roster["results"].append(result)
    assert replay_checkpoints(path, "cpu")["replayed_candidates"] == 1


def test_default_replay_uses_recorded_device(replay_fixture, monkeypatch):
    path, _, bundle = replay_fixture

    def recorded_loader(path, device):
        if device != "mps":
            raise ValueError("unexpected cross-device replay")
        return bundle

    monkeypatch.setattr("learned_dynamics.replay.load_checkpoint", recorded_loader)
    assert replay_checkpoints(path)["comparison_scope"] == "same_device"


@pytest.mark.parametrize("component,drift", [("position_rad", 2e-6), ("velocity_rad_s", 2e-5)])
def test_replay_rejects_either_component_beyond_frozen_tolerance(
        replay_fixture, monkeypatch, component, drift):
    path, _, _ = replay_fixture

    class DriftingPredictor:
        def predict(self, state, action, context):
            from forward_dynamics.types import DynamicsState
            values = {"position_rad": state.position_rad, "velocity_rad_s": state.velocity_rad_s}
            values[component] += drift
            return DynamicsState(**values)

    monkeypatch.setattr("learned_dynamics.replay.LearnedPredictor", lambda *args: DriftingPredictor())
    with pytest.raises(ValueError, match="exceeds declared tolerance"):
        replay_checkpoints(path)
