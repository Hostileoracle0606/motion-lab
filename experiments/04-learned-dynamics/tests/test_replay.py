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


def test_replay_proves_saved_predictions_came_from_declared_checkpoint(
        tmp_path, synthetic_views, monkeypatch):
    cases = synthetic_views[2].cases[:6]
    run = RunSpec("canonical32", ModelSpec(32, 0), "one_step", "all", 0, "cpu", 0)
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
    monkeypatch.setattr("learned_dynamics.replay.load_checkpoint",
                        lambda path, device: SimpleNamespace(model=object(), normalizer=object()))
    monkeypatch.setattr("learned_dynamics.replay.LearnedPredictor",
                        lambda model, normalizer, context, predictor_id, device: NoChange())
    result = replay_checkpoints(tmp_path / "evidence", "cpu")
    assert result == {"schema_version": 1, "replayed_candidates": 1,
                      "replayed_records": len(records), "status": "verified"}

    changed = list(deepcopy(records))
    row = next(item for item in changed if item["status"] == "success")
    row["predicted_endpoint_state"]["position_rad"] += 0.01
    evidence["records"] = tuple(changed)
    with pytest.raises(ValueError, match="checkpoint replay"):
        replay_checkpoints(tmp_path / "evidence", "cpu")
