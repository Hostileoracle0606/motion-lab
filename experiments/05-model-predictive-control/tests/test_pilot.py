from dataclasses import asdict
import json
from pathlib import Path
import tomllib

import pytest

from model_predictive_control import pilot
from model_predictive_control.pilot import run_pilot, write_evidence


EXPERIMENT = Path(__file__).resolve().parents[1]
MODEL = EXPERIMENT.parent / "01-feedback-control/models/one_link.xml"


def test_paired_fixed_work_run_reproduces_actions_and_writes_evidence(tmp_path):
    config = tomllib.loads((EXPERIMENT / "configs/pilot-v1.toml").read_text())
    config["steps"] = 4
    config["mpc"]["horizon"] = 2
    config["mpc"]["candidate_count"] = 3
    config["scenarios"] = config["scenarios"][:1]

    first = run_pilot(config, MODEL)
    second = run_pilot(config, MODEL)
    assert len(first) == 2
    assert first[0].scenario == first[1].scenario
    assert first[0].seed == first[1].seed
    assert [record.controller_id for record in first] == ["pd", "rk4_mpc"]
    assert sum(step.predictor_calls for step in first[1].steps) == 24

    def without_timing(records):
        data = [asdict(record) for record in records]
        for episode in data:
            for step in episode["steps"]:
                del step["decision_time_ns"]
                del step["action_ready_time_ns"]
        return data

    assert without_timing(first) == without_timing(second)
    write_evidence(tmp_path, config, first, {"source_commit": "source-fixture"})
    saved = json.loads((tmp_path / "episodes.json").read_text())
    assert saved["schema_version"] == 1
    assert len(saved["episodes"]) == 2
    result = json.loads((tmp_path / "metrics.json").read_text())
    assert result["cohorts"][1]["episodes"][0]["predictor_calls"] == 24
    assert (tmp_path / "paired.png").stat().st_size > 0
    assert json.loads((tmp_path / "manifest.json").read_text())["source_commit"] == "source-fixture"

    recorded = tmp_path / "recorded"
    assert pilot.run_recorded_pilot(
        recorded, config, MODEL, {"source_commit": "source-fixture"},
    ) == 2
    status = json.loads((recorded / "run-status.json").read_text())
    assert status == {
        "schema_version": 1, "status": "complete", "completed_episodes": 2,
        "expected_episodes": 2,
    }
    journal = [json.loads(line)["episode"] for line in
               (recorded / "completed-episodes.jsonl").read_text().splitlines()]
    assert journal == json.loads((recorded / "episodes.json").read_text())["episodes"]


def test_recorded_pilot_preserves_completed_episode_on_abort(tmp_path, monkeypatch):
    config = tomllib.loads((EXPERIMENT / "configs/pilot-v1.toml").read_text())
    config["steps"] = 2
    config["scenarios"] = config["scenarios"][:1]
    advance = pilot.MujocoPlant.advance
    calls = 0
    failure = RuntimeError("interrupted during the second episode")

    def fail_during_second_episode(self, torque):
        nonlocal calls
        calls += 1
        if calls == 4:
            raise failure
        return advance(self, torque)

    monkeypatch.setattr(pilot.MujocoPlant, "advance", fail_during_second_episode)
    output = tmp_path / "aborted"
    with pytest.raises(RuntimeError) as raised:
        pilot.run_recorded_pilot(output, config, MODEL, {"source_commit": "fixture"})
    assert raised.value is failure
    status = json.loads((output / "run-status.json").read_text())
    assert status["status"] == "aborted"
    assert status["completed_episodes"] == 1
    assert status["expected_episodes"] == 2
    assert status["exception_type"] == "RuntimeError"
    records = [json.loads(line) for line in (output / "completed-episodes.jsonl").read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["episode"]["controller_id"] == "pd"
    assert len(records[0]["episode"]["steps"]) == 2
    assert not (output / "metrics.json").exists()
    from model_predictive_control.attempts import audit_run
    audit = audit_run(output, require_complete=False)
    assert len(audit["completed"]) == 1
    assert audit["interrupted"] == [audit["planned"][1]]
    assert audit["missing"] == []
    assert audit["errors"] == []
    with pytest.raises(ValueError):
        audit_run(output)


def test_optional_cem_comparison_matches_candidate_budgets(tmp_path):
    config = tomllib.loads((EXPERIMENT / "configs/pilot-v1.toml").read_text())
    config["steps"] = 3
    config["mpc"]["horizon"] = 2
    config["mpc"]["candidate_count"] = 6
    config["mpc"]["cem"] = {"iterations": 2, "elite_count": 2, "min_std_nm": 0.25}
    config["scenarios"] = config["scenarios"][:1]
    output = tmp_path / "comparison"
    assert pilot.run_recorded_pilot(output, config, MODEL, {}) == 3
    records = json.loads((output / "episodes.json").read_text())["episodes"]
    assert [record["controller_id"] for record in records] == ["pd", "rk4_mpc", "rk4_cem"]
    assert records[1]["scenario"] == records[2]["scenario"]
    assert records[1]["seed"] == records[2]["seed"]
    for record in records[1:]:
        assert [step["candidate_slots"] for step in record["steps"]] == [6, 6, 6]
        assert sum(step["predictor_calls"] for step in record["steps"]) == 36
    cohorts = json.loads((output / "metrics.json").read_text())["cohorts"]
    assert [cohort["episodes"][0]["candidate_slots"] for cohort in cohorts] == [0, 18, 18]
    assert [cohort["episodes"][0]["rejected_candidates"] for cohort in cohorts] == [0, 0, 0]


def test_attempt_audit_complete_and_truncated(tmp_path):
    from model_predictive_control.attempts import audit_run

    config = tomllib.loads((EXPERIMENT / "configs/pilot-v1.toml").read_text())
    config["steps"] = 1
    config["mpc"]["horizon"] = 1
    config["mpc"]["candidate_count"] = 1
    config["scenarios"] = config["scenarios"][:1]
    output = tmp_path / "run"
    pilot.run_recorded_pilot(output, config, MODEL, {})
    audit = audit_run(output)
    assert len(audit["planned"]) == len(audit["started"]) == len(audit["completed"]) == 2
    assert audit["interrupted"] == audit["missing"] == []
    journal = output / "completed-episodes.jsonl"
    journal.write_bytes(journal.read_bytes()[:-1])
    audit = audit_run(output, require_complete=False)
    assert audit["truncated"]["completed"]
    assert len(audit["interrupted"]) == 1
    with pytest.raises(ValueError):
        audit_run(output)


def test_attempt_audit_rejects_duplicate_and_malformed_records(tmp_path):
    from model_predictive_control.attempts import audit_run
    config = tomllib.loads((EXPERIMENT / "configs/pilot-v1.toml").read_text())
    config["steps"] = 1
    config["mpc"]["horizon"] = config["mpc"]["candidate_count"] = 1
    config["scenarios"] = config["scenarios"][:1]
    output = tmp_path / "run"
    pilot.run_recorded_pilot(output, config, MODEL, {})
    journal = output / "started-attempts.jsonl"
    first = journal.read_text().splitlines()[0]
    journal.write_text(first + "\n" + first + "\n{}\n")
    audit = audit_run(output, require_complete=False)
    assert any("duplicate" in error for error in audit["errors"])
    assert any("malformed" in error for error in audit["errors"])
    with pytest.raises(ValueError):
        audit_run(output)


def test_incomplete_attempt_journal_cannot_publish_summary(tmp_path, monkeypatch):
    config = tomllib.loads((EXPERIMENT / "configs/pilot-v1.toml").read_text())
    config["steps"] = 1
    config["mpc"]["horizon"] = config["mpc"]["candidate_count"] = 1
    config["scenarios"] = config["scenarios"][:1]
    monkeypatch.setattr(pilot, "record_started", lambda output, attempt: None)
    output = tmp_path / "run"
    with pytest.raises(ValueError, match="incomplete attempt evidence"):
        pilot.run_recorded_pilot(output, config, MODEL, {})
    assert not (output / "metrics.json").exists()
    assert json.loads((output / "run-status.json").read_text())["status"] == "aborted"
