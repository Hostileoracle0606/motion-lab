"""Compare ways to improve MPC settling on paired development cases."""

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import platform
from statistics import median
import tomllib

from forward_dynamics.analytical import AnalyticalRK4Predictor
from forward_dynamics.plant import expected_plant_parameters
from forward_dynamics.provenance import git_output
from forward_dynamics.types import DynamicsState
from model_predictive_control.closed_loop import PDController, run_episode
from model_predictive_control.contracts import NOMINAL_CONTEXT, PHYSICS_INTERVAL_S, Scenario
from model_predictive_control.metrics import summarize_episode
from model_predictive_control.pilot import planning_controller
from model_predictive_control.planner import CostWeights, plan
from model_predictive_control.plant import MujocoPlant
from model_predictive_control.provenance import cpu_model


EXPERIMENT = Path(__file__).resolve().parents[1]
MODEL = EXPERIMENT.parent / "01-feedback-control/models/one_link.xml"
BASE = EXPERIMENT / "configs/optimizer-comparison-v1.toml"
PROTOCOL = EXPERIMENT / "configs/settling-development-v1.toml"


def final_hold_samples(record, tolerances):
    target = record.scenario.target_position_rad
    count = 0
    for step in reversed(record.steps):
        state = step.next_state
        if (state is None
                or abs(state.position_rad - target) > tolerances["position_tolerance_rad"]
                or abs(state.velocity_rad_s) > tolerances["velocity_tolerance_rad_s"]):
            break
        count += 1
    return count


def summarize(rows):
    return {
        "episodes": len(rows), "holds": sum(row["success"] for row in rows),
        "median_final_hold_samples": median(row["final_hold_samples"] for row in rows),
        "median_tracking_rmse_rad": median(row["tracking_rmse_rad"] for row in rows),
        "median_effort_nm2_s": median(row["effort_nm2_s"] for row in rows),
        "median_episode_action_ready_p99_ms": median(row["action_ready_time_p99_ms"] for row in rows),
        "truth_failures": sum(row["termination"] != "time_limit" for row in rows),
        "predictor_calls": sum(row["predictor_calls"] for row in rows),
        "candidate_slots": sum(row["candidate_slots"] for row in rows),
    }


def configuration(path):
    payload = path.read_bytes()
    return tomllib.loads(payload.decode("utf-8")), hashlib.sha256(payload).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--feedback-proposal", action="store_true")
    parser.add_argument("--learned-checkpoint", type=Path)
    args = parser.parse_args()
    if args.learned_checkpoint is not None and not args.feedback_proposal:
        parser.error("--learned-checkpoint requires --feedback-proposal")
    if git_output("status", "--porcelain"):
        parser.error("commit the experiment source before recording results")
    base, base_digest = configuration(BASE)
    protocol, protocol_digest = configuration(PROTOCOL)
    args.output.mkdir(parents=True, exist_ok=False)
    nominal = MujocoPlant(MODEL)
    friction = MujocoPlant(MODEL, frictionloss_nm=protocol["mild_frictionloss_nm"])
    predictor = AnalyticalRK4Predictor(
        expected_plant_parameters(NOMINAL_CONTEXT.environment_version), PHYSICS_INTERVAL_S,
    )
    feedback_predictor = predictor
    checkpoint_identity = None
    if args.learned_checkpoint is not None:
        from model_predictive_control.predictor_comparison import load_comparators

        comparators, checkpoint_identity = load_comparators(MODEL, args.learned_checkpoint)
        feedback_predictor = comparators["learned_mpc"]
    pd = PDController(**base["pd"])
    weights = CostWeights(**base["mpc"]["weights"])
    tolerances = base["metrics"]
    manifest = {
        "scope": "settling_development",
        "experiment": "feedback_proposal" if args.feedback_proposal else "terminal_speed",
        "source_commit": git_output("rev-parse", "HEAD"),
        "base_config_sha256": base_digest, "protocol_sha256": protocol_digest,
        "base_config": base, "protocol": protocol, "predictor_context": asdict(NOMINAL_CONTEXT),
        "predictor_id": predictor.predictor_id,
        "feedback_predictor_id": feedback_predictor.predictor_id,
        "checkpoint": checkpoint_identity,
        "truth_sha256": {"nominal": nominal.model_sha256, "mild_friction": friction.model_sha256},
        "cpu_model": cpu_model(), "platform": platform.platform(),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")

    def run_case(plant, condition, setup, stage, weight, *, use_feedback=False):
        seed = setup["seed"]
        scenario = Scenario(
            setup.get("scenario_id", f"{condition}-{seed}"),
            DynamicsState(setup["position_rad"], setup["velocity_rad_s"]),
            setup.get("target_position_rad", base["target_position_rad"]),
            base["steps"],
        )
        name = "pd" if weight is None else ("feedback-proposal" if use_feedback else f"terminal-speed-{weight:g}")
        active_predictor = feedback_predictor if use_feedback else predictor
        controller = pd if weight is None else planning_controller(
            plan, active_predictor, replace(weights, terminal_velocity=weight), base["mpc"], seed,
            {"feedback": pd} if use_feedback else {},
        )
        record = run_episode(
            plant, scenario, controller, controller_id=name, seed=seed, fallback=pd,
        )
        metadata = {"stage": stage, "condition": condition, "terminal_velocity_weight": weight,
                    "feedback_proposal": use_feedback,
                    "predictor_id": active_predictor.predictor_id if weight is not None else None}
        with (args.output / "episodes.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({**metadata, "episode": asdict(record)}, allow_nan=False) + "\n")
        return {
            **metadata, **asdict(summarize_episode(record, **tolerances)),
            "final_hold_samples": final_hold_samples(record, tolerances),
        }

    calibration = {}
    selected = 0.0
    if args.feedback_proposal:
        confirmation_cases = protocol["feedback_confirmation"]
    else:
        confirmation_cases = protocol["confirmation"]
        for weight in protocol["terminal_velocity_weights"]:
            calibration[weight] = [
                run_case(nominal, "nominal", setup, "calibration", weight) for setup in base["scenarios"]
            ]
            print(f"Calibration weight {weight:g}: {summarize(calibration[weight])}", flush=True)
        selected = max(
            (weight for weight in calibration if weight > 0),
            key=lambda w: (sum(r["success"] for r in calibration[w]),
                           median(r["final_hold_samples"] for r in calibration[w]), -w),
        )
    confirmation = {}
    pairs = []
    for condition, plant in (("nominal", nominal), ("mild_friction", friction)):
        sets = {
            name: [
                run_case(plant, condition, setup, "confirmation", weight,
                         use_feedback=args.feedback_proposal and name == "selected")
                for setup in confirmation_cases
            ]
            for name, weight in (("baseline", 0.0), ("selected", selected), ("pd", None))
        }
        confirmation[condition] = sets
        for baseline, candidate in zip(sets["baseline"], sets["selected"], strict=True):
            pairs.append({
                "condition": condition, "seed": baseline["seed"], "baseline_hold": baseline["success"],
                "candidate_hold": candidate["success"],
                "new_truth_failure": baseline["termination"] == "time_limit"
                and candidate["termination"] != "time_limit",
            })
        print(f"Confirmation {condition}: { {name: summarize(rows) for name, rows in sets.items()} }", flush=True)
    extra_holds = sum(pair["candidate_hold"] - pair["baseline_hold"] for pair in pairs)
    passes = extra_holds >= protocol["minimum_extra_holds"] and not any(p["new_truth_failure"] for p in pairs)
    result = {
        "selected_weight": selected, "extra_holds": extra_holds, "improvement_gate_passed": passes,
        "feedback_proposal": args.feedback_proposal,
        "calibration": {weight: summarize(rows) for weight, rows in calibration.items()},
        "confirmation": {
            condition: {name: summarize(rows) for name, rows in sets.items()}
            for condition, sets in confirmation.items()
        },
        "paired_holds": pairs,
        "episode_metrics": [
            row for rows in calibration.values() for row in rows
        ] + [row for sets in confirmation.values() for rows in sets.values() for row in rows],
    }
    (args.output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Confirmation change {extra_holds:+d} holds, gate passed: {passes}")


if __name__ == "__main__":
    main()
