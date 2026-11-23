from dataclasses import asdict
import json
from pathlib import Path
import tomllib

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
