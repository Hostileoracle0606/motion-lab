import hashlib
import json
from pathlib import Path
import shutil

import pytest

from forward_dynamics.config import DEFAULT_CONFIG
from forward_dynamics.evidence import load_evaluation_evidence
from forward_dynamics.generation import run_evaluation


@pytest.fixture(scope="module")
def saved_bundle(tmp_path_factory):
    output = tmp_path_factory.mktemp("one-step-evidence") / "bundle"
    run_evaluation(DEFAULT_CONFIG, output, allow_dirty=True)
    return output


def rehash(output: Path) -> None:
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for name in ("evaluation.json", "one_step.jsonl"):
        manifest["files"][name]["sha256"] = hashlib.sha256((output / name).read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")


def test_loads_complete_three_adapter_one_step_evidence(saved_bundle) -> None:
    evidence = load_evaluation_evidence(saved_bundle)
    assert evidence.evaluation["scope"] == "one_step"
    assert evidence.evaluation["case_count"] == 2540
    assert evidence.evaluation["prediction_count"] == 7620
    assert evidence.evaluation["split_counts"] == {"train": 940, "validation": 800, "test": 800}
    assert len(evidence.predictions) == 7620
    assert sum(row.case.terminated for row in evidence.predictions) == 6
    assert sum(row.case.truncated for row in evidence.predictions) == 18
    assert {row.predictor_id for row in evidence.predictions} == {"mujoco", "no_change", "constant_velocity"}
    assert all(row.residual.position_error_rad == 0.0 and row.residual.velocity_error_rad_s == 0.0
               for row in evidence.predictions if row.predictor_id == "mujoco")


@pytest.mark.parametrize("change", ["prediction", "residual", "membership", "duplicate", "consistent_bad_prediction"])
def test_rejects_semantic_row_tampering_even_with_repaired_checksums(saved_bundle, tmp_path, change) -> None:
    output = tmp_path / "changed"
    shutil.copytree(saved_bundle, output)
    records = [json.loads(line) for line in (output / "one_step.jsonl").read_text().splitlines()]
    first = records[0]
    if change in ("prediction", "consistent_bad_prediction"):
        first["predicted_next_state"]["position_rad"] += 0.1
        if change == "consistent_bad_prediction":
            first["residual"]["position_error_rad"] = (
                first["predicted_next_state"]["position_rad"] - first["case"]["expected_next_state"]["position_rad"]
            )
    elif change == "residual":
        first["residual"]["velocity_error_rad_s"] += 0.1
    elif change == "membership":
        first["case"]["split"] = "test"
    else:
        records[1] = first
    (output / "one_step.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
    rehash(output)
    with pytest.raises(ValueError):
        load_evaluation_evidence(output)


@pytest.mark.parametrize("change", ["aggregate", "count", "provenance", "schema", "input_hash", "scope", "dirty_boolean"])
def test_rejects_contradictory_metadata_even_with_repaired_checksums(saved_bundle, tmp_path, change) -> None:
    output = tmp_path / "changed"
    shutil.copytree(saved_bundle, output)
    path = output / "evaluation.json"
    payload = json.loads(path.read_text())
    if change == "aggregate":
        payload["aggregates"][0]["summary"]["position_mae_rad"] += 0.1
    elif change == "count":
        payload["prediction_count"] = 1
    elif change == "provenance":
        payload["provenance"]["dataset_builder_commit"] = "0" * 40
    elif change == "schema":
        payload["schema_version"] = True
    elif change == "input_hash":
        payload["provenance"]["inputs"][0]["sha256"] = "0" * 64
    elif change == "scope":
        payload["scope"] = "full"
    else:
        payload["provenance"]["working_tree_dirty"] = 1
    path.write_text(json.dumps(payload) + "\n")
    rehash(output)
    with pytest.raises(ValueError):
        load_evaluation_evidence(output)


def test_saved_evidence_loader_never_steps_the_simulator(saved_bundle, monkeypatch) -> None:
    import mujoco

    def forbidden(*args, **kwargs):
        raise AssertionError("evidence loading must not query simulator dynamics")

    monkeypatch.setattr(mujoco, "mj_step", forbidden)
    monkeypatch.setattr(mujoco, "mj_forward", forbidden)
    assert len(load_evaluation_evidence(saved_bundle).predictions) == 7620


def test_clean_commit_cannot_claim_input_bytes_missing_from_that_commit(saved_bundle, tmp_path) -> None:
    output = tmp_path / "wrong-clean-commit"
    shutil.copytree(saved_bundle, output)
    path = output / "evaluation.json"
    payload = json.loads(path.read_text())
    payload["provenance"]["evaluator_commit"] = "ef2223862e0ed301de8c3143ac51daafb31646fe"
    payload["provenance"]["working_tree_dirty"] = False
    path.write_text(json.dumps(payload) + "\n")
    rehash(output)
    with pytest.raises(ValueError, match="commit|input"):
        load_evaluation_evidence(output)
