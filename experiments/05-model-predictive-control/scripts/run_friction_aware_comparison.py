"""Evaluate learned and friction-aware analytical plans on 100 matched starts."""

import argparse
from dataclasses import asdict
import hashlib
from itertools import product
import json
import math
from pathlib import Path
import platform
from statistics import median

import numpy as np
import torch

from forward_dynamics.dry_friction import AnalyticalDryFrictionPredictor
from forward_dynamics.provenance import git_output
from forward_dynamics.types import DynamicsState
from model_predictive_control.closed_loop import PDController, run_episode
from model_predictive_control.contracts import NOMINAL_CONTEXT
from model_predictive_control.metrics import summarize_episode
from model_predictive_control.pilot import planning_controller
from model_predictive_control.planner import CostWeights, plan
from model_predictive_control.plant import MujocoPlant
from model_predictive_control.predictor_comparison import build_roster, load_comparators
from model_predictive_control.provenance import cpu_model
from run_batching_development import DEADLINE_NS, summarize_arm, write_json
from run_guided_friction_development import FROZEN, compare_holds
from run_settling_development import EXPERIMENT, MODEL, configuration


PROTOCOL = EXPERIMENT / "configs/friction-aware-baseline-v1.toml"
ARMS = ("pd", "nominal_fixed", "friction_fixed", "learned_fixed", "friction_deadline", "learned_deadline")


def check_forecasts(plant, predictors, grid):
    rows = []
    for velocity, torque in product(grid["velocities_rad_s"], grid["torques_nm"]):
        initial = plant.reset(DynamicsState(grid["position_rad"], velocity))
        truth = plant.advance(torque)
        if truth.termination is not None or truth.next_state is None:
            raise ValueError("preflight requires an observed valid next state")
        rows.append({
            "initial_state": asdict(initial), "applied_torque_nm": torque,
            "next_state": asdict(truth.next_state),
            "forecasts": {name: asdict(model.predict(initial, torque, NOMINAL_CONTEXT))
                          for name, model in predictors.items()},
        })
    summary = {
        name: {field: math.sqrt(sum((row["forecasts"][name][field] - row["next_state"][field]) ** 2
                                    for row in rows) / len(rows))
               for field in ("position_rad", "velocity_rad_s")}
        for name in predictors
    }
    return {"transitions": len(rows), "rmse": summary, "rows": rows}


def wilson_interval(holds, count):
    z = 1.959963984540054
    fraction = holds / count
    scale = 1 + z ** 2 / count
    center = (fraction + z ** 2 / (2 * count)) / scale
    half_width = z * math.sqrt(fraction * (1 - fraction) / count + z ** 2 / (4 * count ** 2)) / scale
    return [max(0.0, center - half_width), min(1.0, center + half_width)]


def compare(candidate, baseline, protocol):
    counts = compare_holds(candidate, baseline)
    changes = np.array([int(a.success) - int(b.success) for a, b in zip(candidate, baseline, strict=True)])
    rng = np.random.default_rng(protocol["bootstrap_seed"])
    sample = rng.choice(changes, size=(protocol["bootstrap_draws"], len(changes)))
    interval = np.percentile(sample.mean(axis=1), [2.5, 97.5]).tolist()
    return {**counts, "hold_rate_difference": float(changes.mean()), "paired_bootstrap_95": interval}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--learned-checkpoint", required=True, type=Path)
    args = parser.parse_args()
    if git_output("status", "--porcelain"):
        parser.error("commit the experiment source before recording results")

    frozen, frozen_digest = configuration(FROZEN)
    protocol, protocol_digest = configuration(PROTOCOL)
    friction, = [row for row in frozen["conditions"] if row["id"] == "friction"]
    roster = protocol["roster"]
    start = roster["first_seed"]
    count = roster["case_count"]
    config = {
        **frozen, "purpose": "held_out", "conditions": [friction], "fixed_work_repeats": 1,
        "roster": {**frozen["roster"], "generation_seed": roster["generation_seed"],
                   "seeds": list(range(start, start + count))},
    }
    if config["deadline_ns"] != DEADLINE_NS:
        raise ValueError("expected the common 10 ms cutoff")
    cases, _ = build_roster(config)
    comparators, checkpoint_identity = load_comparators(MODEL, args.learned_checkpoint)
    nominal = comparators["rk4_mpc"]
    predictors = {
        "nominal": nominal,
        "friction": AnalyticalDryFrictionPredictor(nominal.parameters, NOMINAL_CONTEXT, friction["frictionloss_nm"]),
        "learned": comparators["learned_mpc"],
    }
    plant = MujocoPlant(MODEL, link_mass_kg=friction["link_mass_kg"], frictionloss_nm=friction["frictionloss_nm"])
    preflight = check_forecasts(plant, predictors, protocol["preflight"])
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "preflight.json", preflight)
    if not all(preflight["rmse"]["friction"][field] < preflight["rmse"]["nominal"][field]
               for field in ("position_rad", "velocity_rad_s")):
        raise ValueError("friction baseline must improve both nominal forecast errors on the preflight grid")
    print(f"One-step grid RMSE {preflight['rmse']}", flush=True)

    orders = [ARMS[index % len(ARMS):] + ARMS[:index % len(ARMS)] for index in range(len(cases))]
    write_json(args.output / "manifest.json", {
        "scope": "held_out_friction_aware_comparison", "source_commit": git_output("rev-parse", "HEAD"),
        "source_config_hashes": {"frozen_reference": frozen_digest, "comparison": protocol_digest},
        "config": config, "comparison_protocol": protocol, "checkpoint": checkpoint_identity,
        "predictor_ids": {name: model.predictor_id for name, model in predictors.items()},
        "analytical_parameters": asdict(nominal.parameters),
        "known_friction_torque_nm": friction["frictionloss_nm"],
        "predictor_context": asdict(NOMINAL_CONTEXT), "truth_sha256": plant.model_sha256,
        "platform": platform.platform(), "cpu_model": cpu_model(),
        "torch_version": str(torch.__version__), "torch_threads": torch.get_num_threads(),
        "roster": [{**asdict(case), "order": order} for case, order in zip(cases, orders, strict=True)],
    })
    pd = PDController(**config["pd"])
    weights = CostWeights(**config["mpc"]["weights"])

    def controller(arm, seed):
        if arm == "pd":
            return pd
        family, _ = arm.split("_")
        return planning_controller(plan, predictors[family], weights, config["mpc"], seed,
                                   {"feedback": pd, "batched": family != "nominal"})

    warm = cases[0]
    for arm in ("nominal_fixed", "friction_fixed", "learned_fixed"):
        controller(arm, warm.seed)(warm.scenario.initial_state, warm.scenario.target_position_rad)
    records = {arm: [] for arm in ARMS}
    with (args.output / "episodes.jsonl").open("w", encoding="utf-8") as stream:
        for index, (case, order) in enumerate(zip(cases, orders, strict=True), start=1):
            for arm in order:
                cutoff = DEADLINE_NS if arm.endswith("_deadline") else None
                record = run_episode(plant, case.scenario, controller(arm, case.seed), controller_id=arm,
                                     seed=case.seed, fallback=pd, deadline_ns=cutoff)
                records[arm].append(record)
                stream.write(json.dumps(asdict(record), allow_nan=False) + "\n")
                stream.flush()
            if index % 10 == 0:
                print(f"Completed {index}/{len(cases)} starts across all six arms", flush=True)

    metrics = config["metrics"]
    summaries, episodes = {}, {}
    for arm, rows in records.items():
        episode_metrics = [summarize_episode(row, **metrics) for row in rows]
        episodes[arm] = episode_metrics
        summary = summarize_arm(rows, metrics)
        final_errors = [abs(row.final_error_rad) for row in episode_metrics if row.final_error_rad is not None]
        summaries[arm] = {
            **summary, "hold_rate_wilson_95": wilson_interval(summary["holds"], summary["episodes"]),
            "median_absolute_terminal_error_rad": median(final_errors) if final_errors else None,
        }

    pairs = {
        "learned_vs_friction_deadline": compare(episodes["learned_deadline"], episodes["friction_deadline"], protocol),
        **{f"learned_vs_{arm}": compare(episodes["learned_fixed"], episodes[arm], protocol)
           for arm in ("pd", "nominal_fixed", "friction_fixed")},
    }
    expected = len(cases) * config["steps"]
    timely = {arm: sum(step.fallback_reason is None and step.action_ready_time_ns <= DEADLINE_NS
                       for row in records[arm] for step in row.steps)
              for arm in ("learned_deadline", "friction_deadline")}
    primary = pairs["learned_vs_friction_deadline"]
    checks = {
        "hold_gain": primary["net_extra_holds"] >= protocol["minimum_extra_paired_holds"],
        "positive_paired_interval": primary["paired_bootstrap_95"][0] > 0,
        "no_truth_failures": summaries["learned_deadline"]["truth_failures"] == 0,
        "timely_plans": timely["learned_deadline"] / expected >= protocol["minimum_timely_plan_fraction"],
    }
    write_json(args.output / "results.json", {
        "arms": summaries, "paired_comparisons": pairs, "expected_deadline_decisions_per_arm": expected,
        "timely_plans_used": timely, "learned_advantage_checks": checks,
        "learned_advantage_gate_passed": all(checks.values()),
        "episodes_by_arm": {arm: [asdict(row) for row in rows] for arm, rows in episodes.items()},
    })
    inventory = []
    for name in ("manifest.json", "preflight.json", "episodes.jsonl", "results.json"):
        with (args.output / name).open("rb") as stream:
            inventory.append(f"{hashlib.file_digest(stream, 'sha256').hexdigest()}  {name}\n")
    (args.output / "SHA256SUMS").write_text("".join(inventory), encoding="utf-8")
    print(f"Primary paired comparison {primary}. Checks {checks}", flush=True)
    print(f"Timely plans used {timely} out of {expected} per arm", flush=True)


if __name__ == "__main__":
    main()
