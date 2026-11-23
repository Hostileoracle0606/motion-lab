"""Compare scalar and batched learned planning on the settling development roster."""

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
from statistics import median

import numpy as np
import torch

from forward_dynamics.provenance import git_output
from forward_dynamics.types import DynamicsState
from model_predictive_control.closed_loop import PDController, run_episode
from model_predictive_control.contracts import NOMINAL_CONTEXT, Scenario
from model_predictive_control.metrics import summarize_episode
from model_predictive_control.pilot import planning_controller
from model_predictive_control.planner import CostWeights, plan
from model_predictive_control.plant import MujocoPlant
from model_predictive_control.predictor_comparison import load_comparators
from model_predictive_control.provenance import cpu_model
from run_settling_development import BASE, MODEL, PROTOCOL, configuration


ARMS = ("scalar", "batched", "batched_deadline", "pd")
DEADLINE_NS = 10_000_000
ACTION_TOLERANCE_NM = 1e-6
MIN_TIMELY_PLAN_FRACTION = 0.99


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def summarize_arm(records, metrics):
    episodes = [summarize_episode(record, **metrics) for record in records]
    steps = [step for record in records for step in record.steps]
    decisions = np.array([step.decision_time_ns for step in steps])
    action_ready = np.array([step.action_ready_time_ns for step in steps])
    settling = [row.settling_time_s for row in episodes if row.settling_time_s is not None]
    return {
        "holds": sum(row.success for row in episodes),
        "episodes": len(episodes),
        "decisions": len(steps),
        "truth_failures": sum(row.termination != "time_limit" for row in episodes),
        "decision_ms_p50": float(np.percentile(decisions, 50) / 1e6),
        "decision_ms_p99": float(np.percentile(decisions, 99) / 1e6),
        "action_ready_ms_p99": float(np.percentile(action_ready, 99) / 1e6),
        "decisions_over_10_ms": int(np.count_nonzero(decisions > DEADLINE_NS)),
        "action_ready_over_10_ms": int(np.count_nonzero(action_ready > DEADLINE_NS)),
        "deadline_discards": sum(step.deadline_missed for step in steps),
        "fallback_reasons": dict(Counter(step.fallback_reason for step in steps if step.fallback_reason)),
        "predicted_rows": sum(step.predictor_calls for step in steps),
        "candidate_slots": sum(step.candidate_slots for step in steps),
        "rejected_candidates": sum(step.rejected_candidates for step in steps),
        "median_settling_s": median(settling) if settling else None,
        "median_tracking_rmse_rad": median(row.tracking_rmse_rad for row in episodes),
        "median_effort_nm2_s": median(row.effort_nm2_s for row in episodes),
    }


def fixed_work_agreement(scalar_records, batch_records):
    pairs = []
    for scalar, batch in zip(scalar_records, batch_records, strict=True):
        assert scalar.scenario == batch.scenario and scalar.seed == batch.seed
        steps = list(zip(scalar.steps, batch.steps))
        torque_differences = [abs(a.applied_torque_nm - b.applied_torque_nm) for a, b in steps]
        pairs.append({
            "scenario_id": scalar.scenario.scenario_id,
            "scalar_decisions": len(scalar.steps), "batched_decisions": len(batch.steps),
            "steps_with_torque_difference_over_tolerance": sum(
                delta > ACTION_TOLERANCE_NM for delta in torque_differences
            ),
            "max_applied_torque_difference_nm": max(torque_differences),
            "steps_with_different_prediction_counts": sum(
                a.predictor_calls != b.predictor_calls for a, b in steps
            ),
            "steps_with_different_rejected_counts": sum(
                a.rejected_candidates != b.rejected_candidates for a, b in steps
            ),
        })
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--learned-checkpoint", required=True, type=Path)
    args = parser.parse_args()
    if git_output("status", "--porcelain"):
        parser.error("commit the experiment source before recording results")

    base, base_digest = configuration(BASE)
    protocol, protocol_digest = configuration(PROTOCOL)
    comparators, checkpoint_identity = load_comparators(MODEL, args.learned_checkpoint)
    predictor = comparators["learned_mpc"]
    plants = {"nominal": MujocoPlant(MODEL),
              "mild_friction": MujocoPlant(MODEL, frictionloss_nm=protocol["mild_frictionloss_nm"])}
    cases = [
        (condition, row["seed"], Scenario(
            f"{condition}-{row['seed']}", DynamicsState(row["position_rad"], row["velocity_rad_s"]),
            row["target_position_rad"], base["steps"],
        ))
        for condition in plants for row in protocol["feedback_confirmation"]
    ]
    run_order = [ARMS[index % len(ARMS):] + ARMS[:index % len(ARMS)] for index in range(len(cases))]
    pd = PDController(**base["pd"])
    weights = CostWeights(**base["mpc"]["weights"])

    def make_controller(arm, seed):
        if arm == "pd":
            return pd
        return planning_controller(plan, predictor, weights, base["mpc"], seed, {
            "feedback": pd, "batched": arm != "scalar",
        })

    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "manifest.json", {
        "scope": "settling_development_batching",
        "source_commit": git_output("rev-parse", "HEAD"),
        "base_config_sha256": base_digest, "protocol_sha256": protocol_digest,
        "base_config": base, "protocol": protocol, "checkpoint": checkpoint_identity,
        "predictor_context": asdict(NOMINAL_CONTEXT),
        "truth_sha256": {name: plant.model_sha256 for name, plant in plants.items()},
        "deadline_ns": DEADLINE_NS, "action_tolerance_nm": ACTION_TOLERANCE_NM,
        "minimum_timely_plan_fraction": MIN_TIMELY_PLAN_FRACTION,
        "platform": platform.platform(), "cpu_model": cpu_model(),
        "torch_version": str(torch.__version__), "torch_threads": torch.get_num_threads(),
        "roster": [{"scenario": asdict(scenario), "seed": seed, "condition": condition, "order": order}
                   for (condition, seed, scenario), order in zip(cases, run_order, strict=True)],
    })
    _, warm_seed, warm_scenario = cases[0]
    for arm in ("scalar", "batched"):
        make_controller(arm, warm_seed)(warm_scenario.initial_state, warm_scenario.target_position_rad)
    records = {arm: [] for arm in ARMS}
    with (args.output / "episodes.jsonl").open("w", encoding="utf-8") as stream:
        for (condition, seed, scenario), order in zip(cases, run_order, strict=True):
            for arm in order:
                record = run_episode(
                    plants[condition], scenario, make_controller(arm, seed), controller_id=arm,
                    seed=seed, fallback=pd, deadline_ns=DEADLINE_NS if arm == "batched_deadline" else None,
                )
                records[arm].append(record)
                stream.write(json.dumps({"condition": condition, "episode": asdict(record)}, allow_nan=False) + "\n")
                stream.flush()
            print(f"Completed {scenario.scenario_id}", flush=True)

    summaries = {arm: summarize_arm(rows, base["metrics"]) for arm, rows in records.items()}
    agreement = fixed_work_agreement(records["scalar"], records["batched"])
    deadline = summaries["batched_deadline"]
    expected_decisions = len(cases) * base["steps"]
    timely_plans = sum(
        step.fallback_reason is None and step.action_ready_time_ns <= DEADLINE_NS
        for record in records["batched_deadline"] for step in record.steps
    )
    timely_fraction = timely_plans / expected_decisions
    gate = (summaries["batched"]["holds"] == summaries["scalar"]["holds"]
            and summaries["batched"]["truth_failures"] == 0
            and deadline["holds"] == len(cases)
            and deadline["truth_failures"] == 0
            and timely_fraction >= MIN_TIMELY_PLAN_FRACTION)
    write_json(args.output / "results.json", {
        "arms": summaries, "fixed_work_agreement": agreement,
        "timely_plans_used": timely_plans, "expected_deadline_decisions": expected_decisions,
        "timely_plan_fraction": timely_fraction, "development_gate_passed": gate,
    })
    inventory = []
    for name in ("manifest.json", "episodes.jsonl", "results.json"):
        with (args.output / name).open("rb") as stream:
            inventory.append(f"{hashlib.file_digest(stream, 'sha256').hexdigest()}  {name}\n")
    (args.output / "SHA256SUMS").write_text("".join(inventory), encoding="utf-8")
    print(f"Used {timely_plans}/{expected_decisions} plans on time. Development gate passed: {gate}")


if __name__ == "__main__":
    main()
