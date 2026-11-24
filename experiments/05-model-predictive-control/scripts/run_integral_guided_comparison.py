"""Ablate integral guidance and compare integral-guided predictors on moving targets."""

import argparse
from copy import copy
from dataclasses import asdict, dataclass, replace
import hashlib
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
from model_predictive_control.closed_loop import IntegralFeedbackController, PDController, run_episode
from model_predictive_control.contracts import (
    CONTROL_INTERVAL_S, NOMINAL_CONTEXT, PHYSICS_INTERVAL_S, TORQUE_LIMIT_NM, ControlDecision, Scenario,
)
from model_predictive_control.guided_planning import plan_integral_guided
from model_predictive_control.metrics import summarize_episode
from model_predictive_control.planner import CostWeights, plan
from model_predictive_control.predictor_comparison import load_comparators
from model_predictive_control.provenance import cpu_model
from run_batching_development import DEADLINE_NS, write_json
from run_guided_friction_development import FROZEN
from run_integral_feedback_comparison import SELECTION, make_cases, make_plant
from run_settling_development import EXPERIMENT, MODEL, configuration


PROTOCOL = EXPERIMENT / "configs/integral-guided-planning-v1.toml"
TRACKING_ARMS = ("integral", "physics_guided", "learned_guided")
FIXED_ARMS = (*TRACKING_ARMS, "learned_pd_candidate", "learned_integral_candidate")


@dataclass(frozen=True)
class Trial:
    condition: str
    scenario: Scenario
    seed: int
    references: np.ndarray
    reference_spec: dict


def tracking_references(*, steps, horizon, amplitude, period, phase):
    omega = 2 * math.pi / period
    time_s = CONTROL_INTERVAL_S * np.arange(steps + horizon + 1)
    angle = omega * time_s + phase
    return np.column_stack((amplitude * np.sin(angle), amplitude * omega * np.cos(angle)))


def trials_for_phase(frozen, protocol, mode):
    horizon = frozen["mpc"]["horizon"]
    if mode == "fixed":
        conditions = [row for row in frozen["conditions"] if row["id"] == "friction"]
        cases = make_cases(frozen, conditions, protocol["fixed"], "held_out")
        return conditions, [
            Trial(case.condition_id, case.scenario, case.seed,
                  np.tile((case.scenario.target_position_rad, 0.0), (case.scenario.max_steps + horizon + 1, 1)),
                  {"position_rad": case.scenario.target_position_rad})
            for case in cases
        ]

    options = protocol["tracking"]
    split = options if mode == "tracking" else protocol["confirmation"]
    conditions = protocol["tracking_conditions"]
    if mode == "confirmation":
        conditions = [row for row in conditions if row["id"] == "payload_friction"]
    trials = []
    for level, condition in enumerate(conditions):
        first_seed = split["first_seed"] + level * split["case_count"]
        for seed in range(first_seed, first_seed + split["case_count"]):
            rng = np.random.default_rng(np.random.SeedSequence([split["generation_seed"], seed]))
            spec = {
                "amplitude": float(rng.uniform(*options["amplitude_rad"])),
                "period": float(rng.uniform(*options["period_s"])),
                "phase": float(rng.uniform(0, 2 * math.pi)),
            }
            refs = tracking_references(steps=options["steps"], horizon=horizon, **spec)
            initial = DynamicsState(float(refs[0, 0] + rng.uniform(*options["initial_position_error_rad"])),
                                    float(refs[0, 1] + rng.uniform(*options["initial_velocity_error_rad_s"])))
            name = condition["id"]
            trials.append(Trial(name, Scenario(f"{name}-{seed}", initial, float(refs[0, 0]), options["steps"]),
                                seed, refs, spec))
    return conditions, trials


def run_trial(trial, arm, plant, predictor, pd, selected, frozen, protocol):
    integral = IntegralFeedbackController(pd, **selected)
    rng = np.random.default_rng(trial.seed)
    weights = CostWeights(**frozen["mpc"]["weights"])
    horizon = frozen["mpc"]["horizon"]
    budget = frozen["mpc"]["candidate_count"]
    baseline = []

    def controller(state, _):
        tick = len(baseline)
        position, velocity = trial.references[tick]
        snapshot = copy(integral)
        default = integral(state, float(position), target_velocity_rad_s=float(velocity))
        baseline.append(default.requested_torque_nm)
        if arm == "integral":
            return default
        if arm in ("learned_pd_candidate", "learned_integral_candidate"):
            result = plan(
                predictor, NOMINAL_CONTEXT, state, float(position), weights,
                horizon=horizon, candidate_count=budget, rng=rng, batched=True,
                feedback=pd if arm == "learned_pd_candidate" else snapshot,
            )
        else:
            result = plan_integral_guided(
                predictor, NOMINAL_CONTEXT, state, weights, trial.references[tick:tick + horizon + 1],
                feedback=snapshot, candidate_count=budget, rng=rng,
                correction_limit_nm=protocol["correction_limit_nm"], correction_block_steps=protocol["correction_block_steps"],
                minimum_relative_improvement=protocol["minimum_relative_improvement"],
            )
        return ControlDecision(
            result.actions_nm[0] if result.actions_nm is not None else None, result.predictor_calls,
            result.candidates, result.candidates - result.feasible_candidates,
        )

    def fallback(_state, _target):
        return ControlDecision(baseline[-1])

    record = run_episode(plant, trial.scenario, controller, controller_id=arm, seed=trial.seed,
                         fallback=fallback, deadline_ns=DEADLINE_NS)
    return record, baseline


def episode_metrics(record, baseline, trial, fixed, metrics):
    finish = record.termination == "time_limit" and len(record.steps) == trial.scenario.max_steps
    expected = trial.scenario.max_steps
    steps = record.steps
    rmse = None
    if finish:
        errors = [step.next_state.position_rad - trial.references[tick + 1, 0] for tick, step in enumerate(steps)]
        rmse = math.sqrt(sum(error ** 2 for error in errors) / len(errors))
    standard = summarize_episode(record, **metrics) if fixed else None
    return {
        "seed": trial.seed, "completed": finish, "decisions": len(steps), "expected": expected,
        "holds": standard.success if standard else None,
        "settling_s": standard.settling_time_s if standard else None,
        "tracking_rmse_rad": rmse,
        "effort_nm2_s": sum(step.applied_torque_nm ** 2 * step.physics_steps * PHYSICS_INTERVAL_S for step in steps),
        "deadlines": sum(step.deadline_missed for step in steps),
        "timely": sum(step.fallback_reason is None and step.action_ready_time_ns <= DEADLINE_NS for step in steps),
        "no_plan": sum(step.fallback_reason == "no_plan" for step in steps),
        "rows_predicted": sum(step.predictor_calls for step in steps),
        "actions_different_from_integral": sum(
            abs(step.applied_torque_nm - min(TORQUE_LIMIT_NM, max(-TORQUE_LIMIT_NM, request))) > 1e-12
            for step, request in zip(steps, baseline, strict=True)
        ),
    }


def summarize(rows, timings):
    errors = [row["tracking_rmse_rad"] for row in rows if row["tracking_rmse_rad"] is not None]
    settling = [row["settling_s"] for row in rows if row["settling_s"] is not None]
    return {
        "episodes": len(rows), "complete": sum(row["completed"] for row in rows),
        "holds": sum(bool(row["holds"]) for row in rows) if rows[0]["holds"] is not None else None,
        "median_rmse_rad": median(errors) if errors else None, "median_settling_s": median(settling) if settling else None,
        "median_effort_nm2_s": median(row["effort_nm2_s"] for row in rows),
        "p99_action_ready_ms": float(np.percentile(timings, 99) / 1e6),
        **{key: sum(row[key] for row in rows) for key in (
            "decisions", "expected", "deadlines", "timely", "no_plan", "rows_predicted", "actions_different_from_integral",
        )},
    }


def compare_tracking(arms, protocol):
    learned = arms["learned_guided"]
    baselines = ("integral", "physics_guided")
    complete_baselines = [arm for arm in baselines if arms[arm]["complete"] == arms[arm]["episodes"]]
    strongest = min(complete_baselines, key=lambda arm: arms[arm]["median_rmse_rad"], default=None)
    gain = effort = None
    complete = learned["complete"] == learned["episodes"]
    if strongest is not None and complete:
        baseline = arms[strongest]
        gain = 1 - learned["median_rmse_rad"] / baseline["median_rmse_rad"]
        effort = learned["median_effort_nm2_s"] / baseline["median_effort_nm2_s"] - 1
    checks = {
        "tracking_gain": gain is not None and gain >= protocol["minimum_tracking_gain"],
        "effort": effort is not None and effort <= protocol["maximum_effort_increase"],
        "on_time": learned["timely"] / learned["expected"] >= protocol["minimum_timely_fraction"],
        "complete": complete,
    }
    return {"strongest_baseline": strongest, "relative_rmse_gain": gain, "relative_effort_increase": effort,
            "checks": checks, "passed": all(checks.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("fixed", "tracking", "confirmation"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--learned-checkpoint", type=Path, required=True)
    args = parser.parse_args()
    if git_output("status", "--porcelain"):
        parser.error("commit the experiment source before recording results")
    frozen, frozen_digest = configuration(FROZEN)
    protocol, protocol_digest = configuration(PROTOCOL)
    selection, selection_digest = configuration(SELECTION)
    selected = {key: selection[key] for key in ("integral_gain", "integral_limit_nm")}
    if frozen["deadline_ns"] != DEADLINE_NS:
        raise ValueError("expected the common 10 ms cutoff")
    models, checkpoint_identity = load_comparators(MODEL, args.learned_checkpoint)
    learned = models["learned_mpc"]
    parameters = models["rk4_mpc"].parameters
    pd = PDController(**frozen["pd"])
    conditions, trials = trials_for_phase(frozen, protocol, args.phase)
    plants = {row["id"]: make_plant(row) for row in conditions}
    physics = {row["id"]: AnalyticalDryFrictionPredictor(parameters, NOMINAL_CONTEXT, row["frictionloss_nm"])
               for row in conditions}
    fixed = args.phase == "fixed"
    arms = FIXED_ARMS if fixed else TRACKING_ARMS
    orders = [arms[index % len(arms):] + arms[:index % len(arms)] for index in range(len(trials))]
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "manifest.json", {
        "scope": "integral_guided_planning", "phase": args.phase, "source_commit": git_output("rev-parse", "HEAD"),
        "config_sha256": {"frozen": frozen_digest, "protocol": protocol_digest, "integral_selection": selection_digest},
        "protocol": protocol, "integral_selection": selection, "checkpoint": checkpoint_identity, "mpc": frozen["mpc"],
        "hold_metrics": frozen["metrics"], "pd": frozen["pd"], "deadline_ns": DEADLINE_NS,
        "conditions": conditions, "analytical_parameters": asdict(parameters), "predictor_context": asdict(NOMINAL_CONTEXT),
        "truth_sha256": {name: plant.model_sha256 for name, plant in plants.items()},
        "platform": platform.platform(), "cpu_model": cpu_model(), "torch_version": str(torch.__version__),
        "torch_threads": torch.get_num_threads(),
        "roster": [{"condition": trial.condition, "scenario": asdict(trial.scenario), "seed": trial.seed,
                    "reference": trial.reference_spec, "order": order}
                   for trial, order in zip(trials, orders, strict=True)],
    })

    first = trials[0]
    warm = replace(first, scenario=replace(first.scenario, max_steps=1))
    for arm in arms:
        predictor = physics[warm.condition] if arm == "physics_guided" else learned
        run_trial(warm, arm, plants[warm.condition], predictor, pd, selected, frozen, protocol)

    rows = {condition: {arm: [] for arm in arms} for condition in plants}
    times = {condition: {arm: [] for arm in arms} for condition in plants}
    with (args.output / "episodes.jsonl").open("w", encoding="utf-8") as stream:
        for index, (trial, order) in enumerate(zip(trials, orders, strict=True), start=1):
            for arm in order:
                predictor = physics[trial.condition] if arm == "physics_guided" else learned
                record, baseline = run_trial(trial, arm, plants[trial.condition], predictor,
                                             pd, selected, frozen, protocol)
                metrics = episode_metrics(record, baseline, trial, fixed, frozen["metrics"])
                rows[trial.condition][arm].append(metrics)
                times[trial.condition][arm].extend(step.action_ready_time_ns for step in record.steps)
                stream.write(json.dumps({"condition": trial.condition, "episode": asdict(record),
                                         "integral_requests_nm": baseline}, allow_nan=False) + "\n")
                stream.flush()
            if index % 10 == 0:
                print(f"Completed {index}/{len(trials)} {args.phase} starts", flush=True)
    summaries = {condition: {arm: summarize(results, times[condition][arm]) for arm, results in group.items()}
                 for condition, group in rows.items()}
    comparisons = {}
    if fixed:
        results = rows["friction"]
        for arm in ("learned_integral_candidate", "learned_guided", "physics_guided"):
            opponent = "learned_pd_candidate" if arm == "learned_integral_candidate" else "integral"
            pairs = list(zip(results[arm], results[opponent], strict=True))
            both = [left["settling_s"] - right["settling_s"] for left, right in pairs if left["holds"] and right["holds"]]
            comparisons[arm] = {
                "baseline": opponent, "hold_wins": sum(left["holds"] and not right["holds"] for left, right in pairs),
                "hold_losses": sum(right["holds"] and not left["holds"] for left, right in pairs),
                "joint_successes": len(both), "median_paired_settling_delta_s": median(both) if both else None,
            }
    else:
        comparisons = {condition: compare_tracking(results, protocol) for condition, results in summaries.items()}
    write_json(args.output / "results.json", {"phase": args.phase, "arms_by_condition": summaries,
                                             "comparisons": comparisons, "episode_metrics": rows})
    inventory = []
    for name in ("manifest.json", "episodes.jsonl", "results.json"):
        with (args.output / name).open("rb") as stream:
            inventory.append(f"{hashlib.file_digest(stream, 'sha256').hexdigest()}  {name}\n")
    (args.output / "SHA256SUMS").write_text("".join(inventory), encoding="utf-8")
    print(json.dumps(comparisons), flush=True)


if __name__ == "__main__":
    main()
