"""Compare guided planners and standalone PD on fresh friction development cases."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
from statistics import median

import torch

from forward_dynamics.provenance import git_output
from model_predictive_control.closed_loop import PDController, run_episode
from model_predictive_control.contracts import NOMINAL_CONTEXT
from model_predictive_control.metrics import summarize_episode
from model_predictive_control.pilot import planning_controller
from model_predictive_control.planner import CostWeights, plan
from model_predictive_control.plant import MujocoPlant
from model_predictive_control.predictor_comparison import build_roster, load_comparators
from model_predictive_control.provenance import cpu_model
from run_batching_development import DEADLINE_NS, summarize_arm, write_json
from run_settling_development import EXPERIMENT, MODEL, configuration


FROZEN = EXPERIMENT / "configs/predictor-comparison-v1.toml"
PROTOCOL = EXPERIMENT / "configs/guided-friction-development-v1.toml"
FIXED_ARMS = ("pd", "rk4_fixed", "learned_fixed")
DEADLINE_ARMS = ("rk4_deadline", "learned_deadline")
PREDICTORS = {"rk4": "rk4_mpc", "learned": "learned_mpc"}


def compare_holds(candidate, baseline):
    wins = losses = 0
    for left, right in zip(candidate, baseline, strict=True):
        if (left.scenario_id, left.seed) != (right.scenario_id, right.seed):
            raise ValueError("hold comparison requires matching scenarios and seeds")
        wins += left.success and not right.success
        losses += right.success and not left.success
    return {"wins": wins, "losses": losses, "net_extra_holds": wins - losses}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--learned-checkpoint", required=True, type=Path)
    parser.add_argument("--phase", choices=("primary", "confirmation"), default="primary")
    args = parser.parse_args()
    if git_output("status", "--porcelain"):
        parser.error("commit the experiment source before recording results")

    frozen, frozen_digest = configuration(FROZEN)
    protocol, protocol_digest = configuration(PROTOCOL)
    conditions = [row for row in frozen["conditions"] if row["id"] in ("nominal", "friction")]
    config = {
        **frozen, "purpose": "development_smoke", "conditions": conditions,
        "roster": {**frozen["roster"], **protocol[args.phase]}, "fixed_work_repeats": 1,
    }
    if config["deadline_ns"] != DEADLINE_NS:
        raise ValueError("expected the common 10 ms cutoff")
    comparators, checkpoint_identity = load_comparators(MODEL, args.learned_checkpoint)
    plants = {row["id"]: MujocoPlant(MODEL, link_mass_kg=row["link_mass_kg"],
                                    frictionloss_nm=row["frictionloss_nm"]) for row in conditions}
    cases, _ = build_roster(config)
    cases = sorted(cases, key=lambda case: (case.seed, case.condition_id != "nominal"))
    orders = {}
    for index, case in enumerate(cases):
        arms = FIXED_ARMS if case.condition_id == "nominal" else (*FIXED_ARMS, *DEADLINE_ARMS)
        rotation = (index // len(conditions)) % len(arms)
        orders[case.scenario.scenario_id] = arms[rotation:] + arms[:rotation]
    pd = PDController(**config["pd"])
    weights = CostWeights(**config["mpc"]["weights"])

    def controller(arm, seed):
        if arm == "pd":
            return pd
        family, _ = arm.split("_")
        return planning_controller(plan, comparators[PREDICTORS[family]], weights, config["mpc"], seed,
                                   {"feedback": pd, "batched": family == "learned"})

    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "manifest.json", {
        "scope": "guided_friction_development", "phase": args.phase,
        "source_commit": git_output("rev-parse", "HEAD"),
        "source_config_hashes": {"frozen_reference": frozen_digest, "development": protocol_digest},
        "config": config, "development_protocol": protocol, "checkpoint": checkpoint_identity,
        "predictor_context": asdict(NOMINAL_CONTEXT),
        "truth_sha256": {name: plant.model_sha256 for name, plant in plants.items()},
        "platform": platform.platform(), "cpu_model": cpu_model(),
        "torch_version": str(torch.__version__), "torch_threads": torch.get_num_threads(),
        "roster": [{**asdict(case), "order": orders[case.scenario.scenario_id]} for case in cases],
    })
    warm = cases[0]
    for arm in FIXED_ARMS:
        controller(arm, warm.seed)(warm.scenario.initial_state, warm.scenario.target_position_rad)
    records = {condition: {} for condition in plants}
    with (args.output / "episodes.jsonl").open("w", encoding="utf-8") as stream:
        for case in cases:
            for arm in orders[case.scenario.scenario_id]:
                cutoff = DEADLINE_NS if arm in DEADLINE_ARMS else None
                record = run_episode(plants[case.condition_id], case.scenario, controller(arm, case.seed),
                                     controller_id=arm, seed=case.seed, fallback=pd, deadline_ns=cutoff)
                records[case.condition_id].setdefault(arm, []).append(record)
                stream.write(json.dumps({"condition": case.condition_id, "episode": asdict(record)},
                                        allow_nan=False) + "\n")
                stream.flush()
            print(f"Completed {case.scenario.scenario_id}", flush=True)

    metrics = config["metrics"]
    episode_metrics = {}
    summaries = {}
    for condition, arm_records in records.items():
        episode_metrics[condition] = {}
        summaries[condition] = {}
        for arm, episodes in arm_records.items():
            rows = [summarize_episode(record, **metrics) for record in episodes]
            episode_metrics[condition][arm] = rows
            final_errors = [abs(row.final_error_rad) for row in rows if row.final_error_rad is not None]
            summaries[condition][arm] = {
                **summarize_arm(episodes, metrics),
                "median_absolute_terminal_error_rad": median(final_errors) if final_errors else None,
            }

    nominal = summaries["nominal"]
    friction = episode_metrics["friction"]
    comparisons = {
        f"{mode}_vs_{baseline}": compare_holds(friction[f"learned_{mode}"], friction[baseline])
        for mode in ("fixed", "deadline") for baseline in ("pd", f"rk4_{mode}")
    }
    expected_deadline = len(friction["learned_deadline"]) * config["steps"]
    timely = sum(
        step.fallback_reason is None and step.action_ready_time_ns <= DEADLINE_NS
        for episode in records["friction"]["learned_deadline"] for step in episode.steps
    )
    checks = {
        "friction_hold_gain": all(row["net_extra_holds"] >= protocol["minimum_extra_paired_holds"]
                                  for row in comparisons.values()),
        "nominal_holds": nominal["learned_fixed"]["holds"] >= max(nominal[arm]["holds"] for arm in ("pd", "rk4_fixed")),
        "no_learned_truth_failures": all(summary["truth_failures"] == 0 for arms in summaries.values()
                                         for arm, summary in arms.items() if arm.startswith("learned_")),
        "timely_plans": timely / expected_deadline >= protocol["minimum_timely_plan_fraction"],
    }
    write_json(args.output / "results.json", {
        "phase": args.phase, "arms_by_condition": summaries, "paired_friction_holds": comparisons,
        "timely_learned_plans": timely, "expected_friction_deadline_decisions": expected_deadline,
        "acceptance_checks": checks, "development_gate_passed": all(checks.values()),
        "episodes": {condition: {arm: [asdict(row) for row in rows] for arm, rows in arms.items()}
                     for condition, arms in episode_metrics.items()},
    })
    inventory = []
    for name in ("manifest.json", "episodes.jsonl", "results.json"):
        with (args.output / name).open("rb") as stream:
            inventory.append(f"{hashlib.file_digest(stream, 'sha256').hexdigest()}  {name}\n")
    (args.output / "SHA256SUMS").write_text("".join(inventory), encoding="utf-8")
    print(f"Friction paired-hold comparisons {comparisons}", flush=True)
    print(f"Used {timely}/{expected_deadline} learned plans on time. Pass: {all(checks.values())}")


if __name__ == "__main__":
    main()
