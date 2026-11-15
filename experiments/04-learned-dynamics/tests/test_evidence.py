from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from learned_dynamics.evaluation import EvaluationCandidate, evaluate, grouped_summaries, summarize_records
from learned_dynamics.evaluation_config import EvaluationConfig
from learned_dynamics.evidence import load_evidence, write_evidence
from learned_dynamics.types import ModelSpec
from learned_dynamics.config import RunSpec


class NoChange:
    def predict(self, state, action, context):
        return state


def repair(output):
    files = {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(output.iterdir()) if p.is_file() and p.name != "manifest.json"}
    (output / "manifest.json").write_text(json.dumps({"schema_version": 1, "files": files}))


@pytest.fixture
def bundle(tmp_path, synthetic_views, monkeypatch):
    cases = synthetic_views[2].cases[:6]
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    for name in ("manifest.json", "train.jsonl", "validation.jsonl", "test.jsonl"):
        (dataset / name).write_text(name + "\n")
    config_file = tmp_path / "evaluation.toml"
    config_file.write_text('''schema_version = 1
dataset_dir = "dataset"
dataset_version = "experiment-02-transition-v1"
training_roster_dir = "training"
output_dir = "evidence"
artifact_budget_bytes = 524288000
horizons = [1, 5, 10, 25, 50, 100]
memberships = ["all", "history_common", "loss_common"]
baseline_ids = ["mujoco", "no_change", "constant_velocity", "analytical_rk4_0.002", "analytical_rk4_0.001", "analytical_rk4_0.0005"]
prediction_tolerance = 1e-12
''')
    output = tmp_path / "evidence"
    config = EvaluationConfig(1, config_file,
        hashlib.sha256(config_file.read_bytes()).hexdigest(), dataset,
        "experiment-02-transition-v1", tmp_path / "training", output,
        524288000, (1, 5, 10, 25, 50, 100),
        ("all", "history_common", "loss_common"),
        ("mujoco", "no_change", "constant_velocity", "analytical_rk4_0.002",
         "analytical_rk4_0.001", "analytical_rk4_0.0005"), 1e-12)
    run_specs = tuple(RunSpec(name, ModelSpec(32, 0), loss, membership, 0, "mps", 0)
                      for name, loss, membership in (
                          ("canonical32", "one_step", "all"),
                          ("history0", "one_step", "history_common"),
                          ("loss1", "one_step", "loss_common")))
    training_hash = "a" * 64
    roster = {
        "schema_version": 1, "training_config_sha256": training_hash,
        "planned_runs": [{"run_id": run.run_id, "run_spec": run.mapping()} for run in run_specs],
        "results": [{"run_id": run.run_id, "status": "failed", "checkpoint_dir": None,
                     "checkpoint_sha256": None, "selected_epoch": None,
                     "selected_validation_mse": None, "failure_reason": "synthetic"}
                    for run in run_specs],
    }
    records = []
    for membership, run in zip(config.memberships, run_specs, strict=True):
        candidates = tuple(EvaluationCandidate.baseline(name, NoChange())
                           for name in config.baseline_ids) + (
            EvaluationCandidate.failed(predictor_id=run.run_id, predictor=object(),
                run_spec=run.mapping(), training_config_sha256=training_hash,
                failure_reason="synthetic"),)
        records.extend(evaluate(cases, candidates, config.horizons, membership))
    write_evidence(output, config, roster, tuple(records), cases)
    monkeypatch.setattr("learned_dynamics.evidence.load_prediction_cases",
                        lambda directory, version: cases)
    monkeypatch.setattr("learned_dynamics.evidence.load_run_manifest",
        lambda directory, validate_checkpoints=False: roster | {"provenance": {
            "dataset_manifest_sha256": hashlib.sha256((dataset / "manifest.json").read_bytes()).hexdigest(),
            "train_sha256": hashlib.sha256((dataset / "train.jsonl").read_bytes()).hexdigest()}})
    return output


def test_saved_evidence_independently_reconstructs_records_and_aggregates(bundle):
    loaded = load_evidence(bundle)
    assert loaded["record_count"] > 0
    assert loaded["summaries"]


@pytest.mark.parametrize("target", [
    "action", "expected", "residual", "summary", "seed", "checkpoint", "selection",
])
def test_semantic_tampering_is_rejected_even_after_outer_hash_repair(bundle, target):
    if target == "summary":
        path = bundle / "evaluation.json"
        payload = json.loads(path.read_text())
        payload["summaries"][0]["counts"]["attempted"] += 1
        path.write_text(json.dumps(payload))
    else:
        path = bundle / "records.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        row = next(item for item in rows if item["status"] == "success")
        if target == "action": row["applied_actions_nm"][0] += 0.1
        if target == "expected": row["expected_endpoint_state"]["position_rad"] += 0.1
        if target == "residual": row["residual"]["position_error_rad"] += 0.1
        if target == "seed": row["data_seed"] += 1
        if target == "checkpoint": row["candidate"]["checkpoint_sha256"] = "f" * 64
        if target == "selection": row["candidate"]["selected_epoch"] = 99
        path.write_text("".join(json.dumps(item) + "\n" for item in rows))
    repair(bundle)
    with pytest.raises(ValueError):
        load_evidence(bundle)


def test_audit_does_not_invoke_evaluation_or_checkpoint_inference(bundle, monkeypatch):
    monkeypatch.setattr("learned_dynamics.evaluation.evaluate",
                        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("inference")))
    monkeypatch.setattr("learned_dynamics.checkpoint.load_checkpoint",
                        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("checkpoint")))
    assert load_evidence(bundle)["record_count"]


def test_grouped_summary_tampering_is_rejected_after_outer_hash_repair(bundle):
    path = bundle / "evaluation.json"
    payload = json.loads(path.read_text())
    assert "grouped_summaries" in payload
    payload["grouped_summaries"]["episode_id"][0]["counts"]["attempted"] += 1
    path.write_text(json.dumps(payload))
    repair(bundle)
    with pytest.raises(ValueError, match="grouped"):
        load_evidence(bundle)


def test_failed_training_cannot_be_rewritten_as_success_even_with_repaired_derivations(bundle):
    path = bundle / "records.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    row = next(item for item in rows if item["candidate"]["kind"] == "learned_failed"
               and item["status"] == "failed")
    row.update(status="success", predicted_endpoint_state=row["expected_endpoint_state"],
               residual={"position_error_rad": 0.0, "velocity_error_rad_s": 0.0},
               failure_step=None, failure_reason=None)
    path.write_text("".join(json.dumps(item) + "\n" for item in rows))
    payload_path = bundle / "evaluation.json"
    payload = json.loads(payload_path.read_text())
    payload["summaries"] = list(summarize_records(tuple(rows)))
    payload["grouped_summaries"] = grouped_summaries(tuple(rows))
    payload_path.write_text(json.dumps(payload))
    repair(bundle)
    with pytest.raises(ValueError, match="failed training"):
        load_evidence(bundle)


def test_evidence_retains_evaluator_source_and_runtime_identity(bundle):
    payload = load_evidence(bundle)
    assert "evaluation_provenance" in payload
    assert len(payload["evaluation_provenance"]["source_sha"]) == 40
    assert payload["evaluation_provenance"]["runtime"]["python"].startswith("3.12.")


def test_episode_outcomes_are_saved_and_independently_audited(bundle):
    path = bundle / "evaluation.json"
    payload = json.loads(path.read_text())
    assert "episode_outcomes" in payload
    payload["episode_outcomes"][0]["termination_reason"] = "concealed outcome"
    path.write_text(json.dumps(payload))
    repair(bundle)
    with pytest.raises(ValueError, match="episode outcomes"):
        load_evidence(bundle)


def test_evaluator_provenance_tampering_is_rejected_after_outer_hash_repair(bundle):
    path = bundle / "evaluation.json"
    payload = json.loads(path.read_text())
    payload["evaluation_provenance"]["working_tree_dirty"] = 1
    path.write_text(json.dumps(payload))
    repair(bundle)
    with pytest.raises(ValueError, match="dirty state"):
        load_evidence(bundle)


@pytest.mark.parametrize("field", ["dataset_dir", "training_roster_dir"])
def test_audit_binds_input_locations_to_original_evaluation_config(bundle, field):
    import shutil

    path = bundle / "evaluation.json"
    payload = json.loads(path.read_text())
    alternate = bundle.parent / "alternate"
    if field == "dataset_dir":
        shutil.copytree(bundle.parent / "dataset", alternate)
    payload[field] = "../alternate"
    path.write_text(json.dumps(payload))
    repair(bundle)
    with pytest.raises(ValueError, match="configured.*input"):
        load_evidence(bundle)


def test_audit_binds_evaluation_data_to_training_provenance(bundle, monkeypatch):
    roster = load_evidence(bundle)["roster"]
    monkeypatch.setattr("learned_dynamics.evidence.load_run_manifest",
        lambda directory, validate_checkpoints=False: roster | {"provenance": {
            "dataset_manifest_sha256": "f" * 64, "train_sha256": "e" * 64}})
    with pytest.raises(ValueError, match="training dataset"):
        load_evidence(bundle)


def test_audit_requires_original_config_bytes_to_match_snapshot(bundle):
    original = bundle.parent / "evaluation.toml"
    original.write_text(original.read_text() + "\n# edited after evaluation\n")
    with pytest.raises(ValueError, match="original evaluation config"):
        load_evidence(bundle)
