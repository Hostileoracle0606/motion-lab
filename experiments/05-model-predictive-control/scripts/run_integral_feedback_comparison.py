"""Calibrate bounded integral feedback and compare it with the learned planner."""

import argparse
from dataclasses import asdict
import hashlib
from itertools import product
import json
from pathlib import Path
import platform
from statistics import median

import torch

from forward_dynamics.provenance import git_output
from model_predictive_control.closed_loop import IntegralFeedbackController, PDController, run_episode
from model_predictive_control.contracts import NOMINAL_CONTEXT
from model_predictive_control.metrics import summarize_episode
from model_predictive_control.pilot import planning_controller
from model_predictive_control.planner import CostWeights, plan
from model_predictive_control.plant import MujocoPlant
from model_predictive_control.predictor_comparison import build_roster, load_comparators
from model_predictive_control.provenance import cpu_model
from run_batching_development import DEADLINE_NS, summarize_arm, write_json
from run_friction_aware_comparison import compare, wilson_interval
from run_guided_friction_development import FROZEN
from run_settling_development import EXPERIMENT, MODEL, configuration


PROTOCOL = EXPERIMENT / "configs/integral-feedback-v1.toml"
SELECTION = EXPERIMENT / "configs/integral-feedback-selection-v1.toml"
ARMS = ("pd", "integral_feedback", "learned")


def make_cases(frozen, conditions, split, purpose):
    first_seed = split["first_seed"]
    config = {
        **frozen, "purpose": purpose, "conditions": conditions,
        "roster": {**frozen["roster"], "generation_seed": split["generation_seed"],
                   "seeds": list(range(first_seed, first_seed + split["case_count"]))},
    }
    return build_roster(config)[0]


def make_plant(condition):
    return MujocoPlant(MODEL, link_mass_kg=condition["link_mass_kg"], frictionloss_nm=condition["frictionloss_nm"])


def calibrate(output, frozen, protocol, source_config_hashes):
    split = protocol["calibration"]
    conditions = [{"id": f"calibration-{index}", "link_mass_kg": 1.0, "frictionloss_nm": value}
                  for index, value in enumerate(split["friction_nm"])]
    cases = make_cases(frozen, conditions, split, "development_smoke")
    plants = {row["id"]: make_plant(row) for row in conditions}
    pd = PDController(**frozen["pd"])
    evaluated = []
    for gain, limit in product(split["integral_gains"], split["integral_limits_nm"]):
        record = {"integral_gain": gain, "integral_limit_nm": limit}
        holds = dict.fromkeys(plants, 0)
        episodes = []
        for case in cases:
            controller = IntegralFeedbackController(pd, gain, limit)
            episode = run_episode(plants[case.condition_id], case.scenario, controller,
                                  controller_id="integral_feedback", seed=case.seed, fallback=pd)
            result = summarize_episode(episode, **frozen["metrics"])
            holds[case.condition_id] += result.success
            episodes.append({
                "scenario_id": result.scenario_id, "seed": result.seed, "condition": case.condition_id,
                "held": result.success, "termination": result.termination, "effort_nm2_s": result.effort_nm2_s,
            })
        evaluated.append({
            **record, "holds_by_condition": holds, "holds": sum(holds.values()),
            "min_condition_holds": min(holds.values()),
            "truth_failures": sum(row["termination"] != "time_limit" for row in episodes),
            "median_torque_effort_nm2_s": median(row["effort_nm2_s"] for row in episodes),
            "episodes": episodes,
        })
    best = min(
        (row for row in evaluated if row["truth_failures"] == 0),
        key=lambda row: (-row["min_condition_holds"], -row["holds"], row["median_torque_effort_nm2_s"],
                         row["integral_gain"], row["integral_limit_nm"]),
    )
    selected = {key: best[key] for key in ("integral_gain", "integral_limit_nm")}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "calibration.json", {
        "source_commit": git_output("rev-parse", "HEAD"), "source_config_hashes": source_config_hashes,
        "conditions": conditions, "roster": [asdict(case) for case in cases],
        "truth_sha256": {name: plant.model_sha256 for name, plant in plants.items()},
        "selected": selected, "candidates": evaluated,
    })
    print(f"Selected {selected} with {best['holds_by_condition']} holds", flush=True)


def load_selection(calibration_path, source_config_hashes):
    payload = calibration_path.read_bytes()
    tuning = json.loads(payload)
    locked, lock_digest = configuration(SELECTION)
    selected = {key: locked[key] for key in ("integral_gain", "integral_limit_nm")}
    if (hashlib.sha256(payload).hexdigest() != locked["calibration_results_sha256"]
            or tuning["source_commit"] != locked["calibration_source_commit"]
            or tuning["source_config_hashes"] != source_config_hashes
            or tuning["selected"] != selected):
        raise ValueError("calibration does not match the committed integral feedback selection")
    return selected, locked, lock_digest


def evaluate(args, frozen, protocol, source_config_hashes):
    selected, locked, lock_digest = load_selection(args.calibration, source_config_hashes)
    if frozen["deadline_ns"] != DEADLINE_NS:
        raise ValueError("expected the common 10 ms cutoff")
    comparators, checkpoint_identity = load_comparators(MODEL, args.learned_checkpoint)
    predictor = comparators["learned_mpc"]
    pd = PDController(**frozen["pd"])
    weights = CostWeights(**frozen["mpc"]["weights"])
    conditions = [{**row, "link_mass_kg": 1.0} for row in protocol["evaluation"]]
    cases = [case for condition in conditions for case in make_cases(frozen, [condition], condition, "held_out")]
    plants = {condition["id"]: make_plant(condition) for condition in conditions}
    orders = [ARMS[index % len(ARMS):] + ARMS[:index % len(ARMS)] for index in range(len(cases))]

    def controller(arm, seed):
        if arm == "pd":
            return pd
        if arm == "integral_feedback":
            return IntegralFeedbackController(pd, **selected)
        return planning_controller(plan, predictor, weights, frozen["mpc"], seed, {"feedback": pd, "batched": True})

    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "manifest.json", {
        "scope": "held_out_integral_feedback_comparison", "source_commit": git_output("rev-parse", "HEAD"),
        "source_config_hashes": source_config_hashes, "selection_hash": lock_digest, "calibration": locked,
        "pd": frozen["pd"], "mpc": frozen["mpc"], "steps": frozen["steps"], "metrics": frozen["metrics"],
        "comparison_protocol": protocol, "checkpoint": checkpoint_identity,
        "predictor_context": asdict(NOMINAL_CONTEXT), "deadline_ns": DEADLINE_NS,
        "truth_sha256": {name: plant.model_sha256 for name, plant in plants.items()},
        "platform": platform.platform(), "cpu_model": cpu_model(),
        "torch_version": str(torch.__version__), "torch_threads": torch.get_num_threads(),
        "roster": [{**asdict(case), "order": order} for case, order in zip(cases, orders, strict=True)],
    })
    warm = cases[0]
    for arm in ARMS:
        controller(arm, warm.seed)(warm.scenario.initial_state, warm.scenario.target_position_rad)
    records = {name: {arm: [] for arm in ARMS} for name in plants}
    with (args.output / "episodes.jsonl").open("w", encoding="utf-8") as stream:
        for index, (case, order) in enumerate(zip(cases, orders, strict=True), start=1):
            for arm in order:
                episode = run_episode(plants[case.condition_id], case.scenario, controller(arm, case.seed),
                                      controller_id=arm, seed=case.seed, fallback=pd, deadline_ns=DEADLINE_NS)
                records[case.condition_id][arm].append(episode)
                stream.write(json.dumps({"condition": case.condition_id, "episode": asdict(episode)},
                                        allow_nan=False) + "\n")
                stream.flush()
            if index % 20 == 0:
                print(f"Completed {index}/{len(cases)} starts across all three arms", flush=True)

    by_condition = {}
    for condition, arms in records.items():
        summaries, rows = {}, {}
        for arm, episodes in arms.items():
            rows[arm] = [summarize_episode(row, **frozen["metrics"]) for row in episodes]
            summary = summarize_arm(episodes, frozen["metrics"])
            expected = len(episodes) * frozen["steps"]
            timely = sum(step.fallback_reason is None and step.action_ready_time_ns <= DEADLINE_NS
                         for episode in episodes for step in episode.steps)
            summaries[arm] = {**summary, "timely_actions": timely, "expected_actions": expected,
                              "hold_rate_wilson_95": wilson_interval(summary["holds"], summary["episodes"])}
        primary = compare(rows["integral_feedback"], rows["learned"], protocol)
        integral, learned = summaries["integral_feedback"], summaries["learned"]
        checks = {
            "hold_margin": primary["hold_rate_difference"] >= -protocol["hold_margin_percentage_points"] / 100,
            "effort": integral["median_effort_nm2_s"] <= learned["median_effort_nm2_s"],
            "no_truth_failures": integral["truth_failures"] == 0,
            "timely": integral["timely_actions"] / integral["expected_actions"] >= protocol["minimum_timely_action_fraction"],
        }
        by_condition[condition] = {
            "arms": summaries, "integral_vs_learned": primary,
            "integral_vs_pd": compare(rows["integral_feedback"], rows["pd"], protocol),
            "integral_feedback_checks": checks,
        }

    favored = all(all(row["integral_feedback_checks"].values()) for row in by_condition.values())
    write_json(args.output / "results.json", {"by_condition": by_condition, "integral_feedback_favored": favored})
    inventory = []
    for name in ("manifest.json", "episodes.jsonl", "results.json"):
        with (args.output / name).open("rb") as stream:
            inventory.append(f"{hashlib.file_digest(stream, 'sha256').hexdigest()}  {name}\n")
    (args.output / "SHA256SUMS").write_text("".join(inventory), encoding="utf-8")
    print(f"Observed practical rule favors integral feedback: {favored}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="mode", required=True)
    for mode in ("calibrate", "evaluate"):
        command = subcommands.add_parser(mode)
        command.add_argument("--output", required=True, type=Path)
        if mode == "evaluate":
            command.add_argument("--learned-checkpoint", required=True, type=Path)
            command.add_argument("--calibration", required=True, type=Path)
    args = parser.parse_args()
    if git_output("status", "--porcelain"):
        parser.error("commit the experiment source before recording results")
    frozen, frozen_digest = configuration(FROZEN)
    protocol, protocol_digest = configuration(PROTOCOL)
    hashes = {"frozen_reference": frozen_digest, "integral_feedback_protocol": protocol_digest}
    if args.mode == "calibrate":
        calibrate(args.output, frozen, protocol, hashes)
    else:
        evaluate(args, frozen, protocol, hashes)


if __name__ == "__main__":
    main()
