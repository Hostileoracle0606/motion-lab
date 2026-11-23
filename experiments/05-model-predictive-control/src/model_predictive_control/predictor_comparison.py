"""Pin model identity and compare predictors under one closed-loop protocol."""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from forward_dynamics.analytical import AnalyticalRK4Predictor
from forward_dynamics.mujoco_predictor import MujocoPredictor
from forward_dynamics.plant import expected_plant_parameters
from forward_dynamics.serialization import require_equal
from forward_dynamics.types import DynamicsState, validate_finite_scalar
from learned_dynamics.checkpoint import load_checkpoint
from learned_dynamics.predictor import LearnedPredictor
from model_predictive_control.attempts import audit_run, identity
from model_predictive_control.closed_loop import PDController, run_episode
from model_predictive_control.contracts import (
    NOMINAL_CONTEXT, PHYSICS_INTERVAL_S, POSITION_LIMIT_RAD, Scenario,
)
from model_predictive_control.pilot import planning_controller, record_run
from model_predictive_control.planner import CostWeights, plan
from model_predictive_control.plant import MujocoPlant


CHECKPOINT_RUN_ID = "canonical32-mps-seed0-repeat0"
CHECKPOINT_MANIFEST_SHA256 = "a30d771e770e9aabcb1d2f5fa7014bb9125d5487e0034dc901f4f7ded411960c"
CHECKPOINT_RELEASE_SHA256 = "f769e45bac9c2b83d6c3b2a13643adfec51b3016f62976586cdfe4d44634330a"
CONTROLLER_IDS = ("pd", "rk4_mpc", "mujoco_mpc", "learned_mpc")
ATTEMPT_POLICY = "experiments/05-model-predictive-control/README.md#attempt-accounting-and-recovery"
REPETITION_KEYS = {"fixed-work": "fixed_work_repeats", "deadline-discard": "deadline_repeats"}


@dataclass(frozen=True)
class Case:
    condition_id: str
    scenario: Scenario
    seed: int


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _interval(values, name, minimum=None, maximum=None):
    low, high = values
    validate_finite_scalar(name, low)
    validate_finite_scalar(name, high)
    if low > high or (minimum is not None and low < minimum) or (maximum is not None and high > maximum):
        raise ValueError(f"invalid {name} interval")
    return low, high


def build_roster(config: dict) -> tuple[tuple[Case, ...], list[dict]]:
    if type(config["schema_version"]) is not int or config["schema_version"] != 1:
        raise ValueError("expected predictor protocol schema_version 1")
    if config["purpose"] not in ("held_out", "development_smoke"):
        raise ValueError("unknown protocol purpose")
    for name in ("steps", "fixed_work_repeats", "deadline_repeats", "deadline_ns"):
        _integer(config[name], name, 1)
    if config["mpc"]["optimizer"] != "random_shooting":
        raise ValueError("predictor comparison requires common random shooting")
    for name in ("horizon", "candidate_count"):
        _integer(config["mpc"][name], name, 1)
    generation = config["roster"]
    generation_seed = _integer(generation["generation_seed"], "generation_seed")
    seeds = [_integer(seed, "seed") for seed in generation["seeds"]]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("expected a nonempty roster of unique seeds")
    names = [condition["id"] for condition in config["conditions"]]
    if not names or any(not isinstance(name, str) or not name.strip() for name in names) or len(set(names)) != len(names):
        raise ValueError("expected nonempty unique truth condition IDs")
    initial = _interval(generation["initial_position_magnitude_rad"], "position", 0, POSITION_LIMIT_RAD)
    target = _interval(generation["target_position_magnitude_rad"], "target", 0, POSITION_LIMIT_RAD)
    velocity = _interval(generation["initial_velocity_rad_s"], "velocity")

    base_cases = {}
    for seed in seeds:
        rng = np.random.default_rng(np.random.SeedSequence([generation_seed, seed]))
        sign = -1 if rng.integers(2) == 0 else 1
        base_cases[seed] = (
            DynamicsState(sign * float(rng.uniform(*initial)), float(rng.uniform(*velocity))),
            -sign * float(rng.uniform(*target)),
        )
    cases = tuple(
        Case(name, Scenario(f"{name}-{seed}", state, target_rad, config["steps"]), seed)
        for name in names for seed, (state, target_rad) in base_cases.items()
    )
    attempts = [
        identity(case.scenario.scenario_id, controller_id, case.seed)
        for case in cases for controller_id in CONTROLLER_IDS
    ]
    return cases, attempts


def run_identity(config: dict, mode: str, repetition: int) -> str:
    if mode not in REPETITION_KEYS:
        raise ValueError("unknown comparison mode")
    _integer(repetition, "repetition", 1)
    if repetition > config[REPETITION_KEYS[mode]]:
        raise ValueError(f"repetition exceeds configured count for {mode}")
    return f"{mode}-{repetition:02d}"


def planned_runs(config: dict) -> list[dict]:
    return [
        {
            "run_id": run_identity(config, mode, repetition),
            "mode": mode, "repetition": repetition,
            "deadline_ns": config["deadline_ns"] if mode == "deadline-discard" else None,
        }
        for mode, key in REPETITION_KEYS.items()
        for repetition in range(1, config[key] + 1)
    ]


def load_comparators(model_path: Path, checkpoint: Path):
    bundle = load_checkpoint(checkpoint, device="cpu")
    metadata = bundle.metadata
    expected_run = {
        "variant_id": "canonical32", "seed": 0, "repeat": 0, "device": "mps",
        "loss": "one_step", "membership": "all", "model": {"hidden_width": 32, "history_steps": 0},
    }
    if (bundle.bundle_sha256 != CHECKPOINT_MANIFEST_SHA256
            or metadata["run_spec"] != expected_run
            or metadata["selection_metric"] != "validation_normalized_one_step_mse"
            or metadata["selected_epoch"] != 75):
        raise ValueError("checkpoint does not match the frozen published canonical run")
    rk4 = AnalyticalRK4Predictor(
        expected_plant_parameters(NOMINAL_CONTEXT.environment_version), PHYSICS_INTERVAL_S,
    )
    learned = LearnedPredictor(bundle.model, bundle.normalizer, NOMINAL_CONTEXT, CHECKPOINT_RUN_ID, "cpu")
    comparators = {
        "rk4_mpc": rk4, "mujoco_mpc": MujocoPredictor(model_path), "learned_mpc": learned,
    }
    learned_identity = {
        "run_id": CHECKPOINT_RUN_ID, "bundle_sha256": bundle.bundle_sha256,
        "release_archive_sha256": CHECKPOINT_RELEASE_SHA256, "runtime_device": "cpu",
        "training_run": metadata["run_spec"], "selected_epoch": metadata["selected_epoch"],
        "selection_metric": metadata["selection_metric"],
        "selected_validation_mse": metadata["selected_validation_mse"],
    }
    return comparators, learned_identity


def prepare_comparison(config: dict, model_path: Path, checkpoint: Path):
    cases, attempts = build_roster(config)
    comparators, checkpoint_identity = load_comparators(model_path, checkpoint)
    plants = {
        row["id"]: MujocoPlant(
            model_path, link_mass_kg=row["link_mass_kg"], frictionloss_nm=row["frictionloss_nm"],
        )
        for row in config["conditions"]
    }
    protocol = {
        "model_context": asdict(NOMINAL_CONTEXT),
        "truth_conditions": [
            {**row, "truth_model_sha256": plants[row["id"]].model_sha256, "automatic_reset": False}
            for row in config["conditions"]
        ],
        "checkpoint": checkpoint_identity,
        "predictors": {key: {"predictor_id": value.predictor_id, "device": "cpu"}
                       for key, value in comparators.items()},
        "observation": "exact_scalar_truth_angle_and_angular_velocity_each_control_step",
        "information_policy": "all_predictors_use_nominal_dynamics_context_for_every_truth_condition",
        "attempt_policy": ATTEMPT_POLICY,
        "cases": [asdict(case) for case in cases],
        "planned_attempts": attempts,
        "planned_runs": planned_runs(config),
    }
    return cases, plants, comparators, protocol


def verify_frozen_manifest(protocol: dict, frozen: dict) -> None:
    if not isinstance(frozen, dict) or type(frozen.get("schema_version")) is not int or frozen["schema_version"] != 1:
        raise ValueError("expected frozen manifest schema_version 1")
    require_equal(frozen.get("protocol"), protocol, "frozen manifest")


def audit_comparison_runs(frozen_path: Path, outputs: list[Path]) -> list[str]:
    """Require every planned run exactly once, with matching source protocol and roster."""
    frozen_bytes = frozen_path.read_bytes()
    frozen = json.loads(frozen_bytes)
    protocol = frozen["protocol"]
    verify_frozen_manifest(protocol, frozen)
    digest = hashlib.sha256(frozen_bytes).hexdigest()
    expected = {row["run_id"]: row for row in protocol["planned_runs"]}
    observed = set()
    for output in outputs:
        manifest = json.loads((output / "manifest.json").read_text())
        name = manifest["run_id"]
        if name not in expected or name in observed:
            raise ValueError(f"unexpected or duplicate run ID: {name}")
        require_equal(manifest["frozen_manifest_sha256"], digest, "frozen manifest digest")
        require_equal(manifest["protocol"], protocol, "run protocol")
        require_equal({key: manifest[key] for key in expected[name]}, expected[name], "run identity")
        audit = audit_run(output)
        require_equal(audit["planned"], protocol["planned_attempts"], "run attempt roster")
        observed.add(name)
    if observed != expected.keys():
        raise ValueError(f"missing planned runs: {sorted(expected.keys() - observed)}")
    return list(expected)


def execute_comparison(output, config, cases, plants, comparators, manifest, deadline_ns):
    pd = PDController(**config["pd"])
    weights = CostWeights(**config["mpc"]["weights"])

    def episodes(on_start):
        for case in cases:
            controllers = {"pd": pd}
            controllers.update({
                key: planning_controller(plan, predictor, weights, config["mpc"], case.seed, {})
                for key, predictor in comparators.items()
            })
            for controller_id in CONTROLLER_IDS:
                on_start(identity(case.scenario.scenario_id, controller_id, case.seed))
                yield run_episode(
                    plants[case.condition_id], case.scenario, controllers[controller_id],
                    controller_id=controller_id, seed=case.seed, fallback=pd, deadline_ns=deadline_ns,
                )

    return record_run(
        output, config, manifest, manifest["protocol"]["planned_attempts"], episodes,
    )
