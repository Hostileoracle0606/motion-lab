"""Real full comparison; mutations repair hashes to test independent auditing."""

import csv
from dataclasses import replace
import hashlib
import json
import math
import shutil

import pytest

from forward_dynamics.config import DEFAULT_CONFIG
from forward_dynamics.evidence import load_evaluation_evidence
from forward_dynamics.generation import run_evaluation


ADAPTERS = ("mujoco", "no_change", "constant_velocity", "analytical_rk4_0.002",
            "analytical_rk4_0.001", "analytical_rk4_0.0005")
SPLITS = ("train", "validation", "test")
HORIZONS = (1, 5, 10, 25, 50, 100)
COMPONENT_METRICS = ("position_mae_rad", "position_rmse_rad", "position_max_abs_rad",
                     "velocity_mae_rad_s", "velocity_rmse_rad_s", "velocity_max_abs_rad_s")


def independently_grouped_errors(evidence, mode):
    """Derive membership, coverage and metrics from states, not saved summaries.

    Deliberately do not share production grouping definitions or metric helpers:
    dropping/mislabeling a group or swapping coordinates must fail these checks.
    """
    episodes = {}
    for row in evidence.predictions:
        if row.predictor_id == "mujoco":
            episodes.setdefault(row.case.episode_id, []).append(row.case)
    errors, coverage = {}, {}
    for name in ADAPTERS:
        for episode in episodes.values():
            first = episode[0]
            for horizon in ((1,) if mode == "one_step" else HORIZONS):
                eligible = len(episode) if mode == "one_step" else max(len(episode) - horizon + 1, 0)
                unavailable = len(episode) - eligible
                for group, value in (("episode", first.episode_id), ("scenario", first.scenario_id),
                                     ("controller", first.controller_id), ("split", first.split)):
                    key = (name, group, first.split, value, horizon)
                    errors.setdefault(key, [])
                    old = coverage.get(key, (0, 0))
                    coverage[key] = (old[0] + eligible, old[1] + unavailable)
    for row in (evidence.predictions if mode == "one_step" else evidence.rollouts):
        if mode == "one_step":
            case, horizon = row.case, 1
            predicted, expected = row.predicted_next_state, case.expected_next_state
        else:
            episode = episodes[row.episode_id]
            case, horizon = episode[0], row.horizon_steps
            predicted = row.predicted_endpoint_state
            expected = episode[row.start_step_index + horizon - 1].expected_next_state
        pair = (predicted.position_rad - expected.position_rad,
                predicted.velocity_rad_s - expected.velocity_rad_s)
        for group, value in (("episode", case.episode_id), ("scenario", case.scenario_id),
                             ("controller", case.controller_id), ("split", case.split)):
            errors[(row.predictor_id, group, case.split, value, horizon)].append(pair)
    result = {}
    for key, pairs in errors.items():
        count = len(pairs)
        eligible, unavailable = coverage[key]
        assert count == eligible
        summary = dict(sample_count=count, eligible_count=eligible, unavailable_count=unavailable)
        for index, names in enumerate((COMPONENT_METRICS[:3], COMPONENT_METRICS[3:])):
            values = [pair[index] for pair in pairs]
            metrics = ((math.fsum(abs(value) for value in values) / count,
                        math.sqrt(math.fsum(value * value for value in values) / count),
                        max(abs(value) for value in values)) if count else (None, None, None))
            summary.update(zip(names, metrics, strict=True))
        result[key] = summary
    return result


def assert_csv_matches_independent_errors(path, expected):
    with path.open() as source:
        rows = list(csv.DictReader(source))
    actual = {}
    for row in rows:
        key = (row["predictor_id"], row["group"], row["split"], row["key"], int(row["horizon_steps"]))
        assert key not in actual, f"duplicate CSV group: {key}"
        actual[key] = row
    assert actual.keys() == expected.keys()
    for key, want in expected.items():
        row = actual[key]
        for count in ("sample_count", "eligible_count", "unavailable_count"):
            assert int(row[count]) == want[count], (path.name, key, count)
        for metric in COMPONENT_METRICS:
            if want[metric] is None:
                assert row[metric] == "", (path.name, key, metric)
            else:
                assert float(row[metric]) == pytest.approx(want[metric], rel=1e-12, abs=0.0), (path.name, key, metric)


def test_markdown_uses_saved_nondefault_valid_reference_tolerance(tmp_path):
    from forward_dynamics.config import load_evaluation_config, protocol_mapping
    from forward_dynamics.evidence import EvaluationEvidence
    from forward_dynamics.report_markdown import report_markdown

    # Parse a real valid protocol with a nondefault tolerance, but no evaluation
    # is needed to test the narrative's use of its saved protocol value.
    root = DEFAULT_CONFIG.parents[3]
    config_path = tmp_path / "nondefault.toml"
    config_path.write_text(DEFAULT_CONFIG.read_text().replace(
        '"../../02-state-transition-dataset/data/dataset-v1"',
        f'"{root / "experiments/02-state-transition-dataset/data/dataset-v1"}"',
    ).replace('"../../01-feedback-control/models/one_link.xml"',
              f'"{root / "experiments/01-feedback-control/models/one_link.xml"}"')
      .replace("simulator_absolute_tolerance = 1e-12", "simulator_absolute_tolerance = 7.5e-9"))
    config = load_evaluation_config(config_path)
    payload = {"scope": "full", "protocol": protocol_mapping(config),
        "provenance": {"evaluator_commit": "0" * 40, "working_tree_dirty": True,
                       "dataset_source_commit": "1" * 40, "dataset_builder_commit": "2" * 40,
                       "runtime": {"python": "3.12.15", "mujoco": "3.16.0", "os": "Darwin", "architecture": "arm64"},
                       "command": "run_evaluation"},
        "plant_parameters": {"environment_version": config.environment_version,
                             "inertia_kg_m2": 0.23242512112403102, "damping_nm_s_per_rad": 0.05},
        "episode_outcomes": [], "rollout_aggregates": []}
    rendered = report_markdown(EvaluationEvidence(tmp_path, payload, ()))
    assert "7.5e-09 absolute tolerance" in rendered


def test_all_csv_groups_metrics_and_empty_rows_match_independent_states(full_evidence, tmp_path):
    from forward_dynamics.report_tables import write_tables

    write_tables(full_evidence, tmp_path)
    for mode, count in (("one_step", 126), ("rollouts", 756)):
        expected = independently_grouped_errors(full_evidence, mode)
        assert len(expected) == count
        assert {key[0] for key in expected} == set(ADAPTERS)
        assert {key[1] for key in expected} == {"episode", "scenario", "controller", "split"}
        assert {key[2] for key in expected} == set(SPLITS)
        assert {key[4] for key in expected} == ({1} if mode == "one_step" else set(HORIZONS))
        for name in ADAPTERS[1:]:
            assert any(row["position_rmse_rad"] > 0 for key, row in expected.items()
                       if key[0] == name and row["sample_count"])
        if mode == "rollouts":
            assert any(row["sample_count"] == 0 for row in expected.values())
        assert_csv_matches_independent_errors(tmp_path / f"{mode}.csv", expected)


def test_all_four_plot_series_match_independent_states_and_counts(full_evidence):
    from forward_dynamics.report_plots import plot_series

    plotted = plot_series(full_evidence)
    assert set(plotted) == {"one_step_position", "one_step_velocity", "rollout_position", "rollout_velocity"}
    for mode, prefix in (("one_step", "one_step"), ("rollouts", "rollout")):
        expected = independently_grouped_errors(full_evidence, mode)
        horizons = [1] if mode == "one_step" else list(HORIZONS)
        for component, metric in (("position", "position_rmse_rad"), ("velocity", "velocity_rmse_rad_s")):
            series = plotted[f"{prefix}_{component}"]
            indexed = {(row["predictor_id"], row["split"]): row for row in series}
            assert len(indexed) == len(series) == 18
            assert set(indexed) == {(name, split) for name in ADAPTERS for split in SPLITS}
            for (name, split), row in indexed.items():
                cells = [expected[(name, "split", split, split, horizon)] for horizon in horizons]
                assert row["horizons_steps"] == horizons
                assert row["sample_counts"] == [cell["sample_count"] for cell in cells]
                assert row["rmse"] == pytest.approx([cell[metric] for cell in cells], rel=1e-12, abs=0.0)


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
