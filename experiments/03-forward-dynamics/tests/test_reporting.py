"""Real full comparison; mutations repair hashes to test independent auditing."""

import csv
from dataclasses import replace
import hashlib
import json
import shutil

import pytest

from forward_dynamics.config import DEFAULT_CONFIG
from forward_dynamics.evidence import load_evaluation_evidence
from forward_dynamics.generation import run_evaluation


@pytest.fixture(scope="module")
def full_bundle(tmp_path_factory):
    output = tmp_path_factory.mktemp("full-comparison") / "bundle"
    run_evaluation(DEFAULT_CONFIG, output, allow_dirty=True)
    return output


@pytest.fixture(scope="module")
def full_evidence(full_bundle):
    return load_evaluation_evidence(full_bundle)


def test_full_protocol_preserves_shared_membership_and_zero_reference(full_evidence):
    evidence = full_evidence
    assert evidence.evaluation["scope"] == "full"
    assert len(evidence.predictions) == 15240
    assert len(evidence.rollouts) == 82908
    assert len(evidence.windows) == 48
    assert evidence.plant_parameters.inertia_kg_m2 == 0.23242512112403102
    assert evidence.plant_parameters.damping_nm_s_per_rad == 0.05
    identities = {}
    for row in evidence.rollouts:
        identities.setdefault(row.predictor_id, []).append((row.episode_id, row.start_step_index, row.horizon_steps))
    assert len(identities) == 6
    assert all(value == identities["mujoco"] for value in identities.values())
    assert sum(row.eligible_count for row in evidence.windows) == 13818
    assert any(row.eligible_count == 0 for row in evidence.windows)
    assert all(row.residual.position_error_rad == row.residual.velocity_error_rad_s == 0
               for row in evidence.rollouts if row.predictor_id == "mujoco")


@pytest.mark.parametrize("change", ["endpoint", "expected_endpoint", "source", "count", "aggregate", "parameters", "adapter_parameters", "consistent_endpoint"])
def test_repaired_full_evidence_is_still_semantically_audited(full_bundle, full_evidence, tmp_path, change):
    output = tmp_path / "corrupt"
    shutil.copytree(full_bundle, output)
    path = output / "evaluation.json"
    payload = json.loads(path.read_text())
    if change in ("endpoint", "expected_endpoint", "source", "consistent_endpoint"):
        path = output / "rollouts.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        row = next(row for row in rows if row["predictor_id"] == "analytical_rk4_0.002")
        if change == "source":
            row["episode_id"] = "invented"
        elif change == "expected_endpoint":
            row["expected_endpoint_state"]["position_rad"] += 0.1
        else:
            row["predicted_endpoint_state"]["position_rad"] += 0.1
            if change == "consistent_endpoint":
                row["residual"]["position_error_rad"] = row["predicted_endpoint_state"]["position_rad"] - row["expected_endpoint_state"]["position_rad"]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        if change == "consistent_endpoint":
            # Repair every affected grouping too, so only independent replay
            # under the original actions can expose this invented endpoint.
            from forward_dynamics.metrics import StateResidual
            from forward_dynamics.rollout_evidence import summarize_rollouts
            from forward_dynamics.types import DynamicsState

            typed = list(full_evidence.rollouts)
            index = next(index for index, value in enumerate(typed)
                         if value.predictor_id == "analytical_rk4_0.002")
            typed[index] = replace(typed[index],
                predicted_endpoint_state=DynamicsState(**row["predicted_endpoint_state"]),
                residual=StateResidual(**row["residual"]))
            payload["rollout_aggregates"] = summarize_rollouts(tuple(typed))
            (output / "evaluation.json").write_text(json.dumps(payload))
    else:
        if change == "count":
            payload["windows"][0]["eligible_count"] += 1
        elif change == "aggregate":
            payload["rollout_aggregates"][0]["summary"]["position_mae_rad"] += 0.1
        elif change == "parameters":
            payload["plant_parameters"]["inertia_kg_m2"] += 0.1
        else:
            payload["adapters"][3]["parameters"]["integration_timestep_s"] = 0.004
        path.write_text(json.dumps(payload))
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for name in manifest["files"]:
        manifest["files"][name]["sha256"] = hashlib.sha256((output / name).read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="rollout baseline" if change == "consistent_endpoint" else None):
        load_evaluation_evidence(output)


def test_report_uses_saved_data_without_simulator_queries(full_bundle, full_evidence, monkeypatch, tmp_path):
    import mujoco
    from forward_dynamics.reporting import generate_report
    from forward_dynamics.report_plots import plot_series

    def forbidden(*args, **kwargs):
        raise AssertionError("report must not compile/query simulator")

    for name in ("mj_step", "mj_forward", "mj_kinematics", "mj_crb", "mj_fullM"):
        monkeypatch.setattr(mujoco, name, forbidden)
    monkeypatch.setattr(mujoco, "MjModel", forbidden)
    report = generate_report(full_bundle, tmp_path / "report")
    assert report.name == "report.md" and report.is_file()
    series = plot_series(full_evidence)
    assert set(series) == {"one_step_position", "one_step_velocity", "rollout_position", "rollout_velocity"}
    for name, values in series.items():
        oracle = next(row for row in values if row["predictor_id"] == "mujoco" and row["split"] == "test")
        assert oracle["sample_counts"] == ([800] if name.startswith("one_step") else [800, 792, 782, 752, 702, 602])
        assert all(value == 0 for value in oracle["rmse"])
        assert (report.parent / (name + ".png")).stat().st_size > 100
    with (report.parent / "rollouts.csv").open() as source:
        rows = list(csv.DictReader(source))
    empty = [row for row in rows if row["group"] == "episode" and row["sample_count"] == "0"]
    assert empty and all(row["position_rmse_rad"] == "" for row in empty)
    for row in rows:
        if row["group"] == "split" and row["predictor_id"] == "mujoco" and row["split"] == "test":
            assert float(row["position_rmse_rad"]) == 0


def test_report_output_protection(full_bundle, tmp_path):
    from forward_dynamics.reporting import generate_report

    for target in (full_bundle, full_bundle.parent, DEFAULT_CONFIG.parent / "bad-report"):
        with pytest.raises(ValueError):
            generate_report(full_bundle, target)
    target = tmp_path / "existing"
    target.mkdir()
    with pytest.raises(ValueError):
        generate_report(full_bundle, target)
    assert generate_report(full_bundle, full_bundle / "report").is_file()
