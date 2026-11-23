"""Run and report the paired, nominal one-link development pilot."""

from dataclasses import asdict
import json
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np

from forward_dynamics.analytical import AnalyticalRK4Predictor
from forward_dynamics.plant import expected_plant_parameters
from forward_dynamics.types import DynamicsState
from model_predictive_control.closed_loop import PDController, run_episode
from model_predictive_control.contracts import (
    NOMINAL_CONTEXT, PHYSICS_INTERVAL_S, ControlDecision, EpisodeRecord, Scenario,
)
from model_predictive_control.metrics import summarize_cohort
from model_predictive_control.planner import CostWeights, plan
from model_predictive_control.plant import MujocoPlant


def run_pilot(
    config: dict, model_path: Path, *, deadline_ns: int | None = None,
) -> tuple[EpisodeRecord, ...]:
    """Run both controllers from the same reset and scenario seed."""
    if type(config["schema_version"]) is not int or config["schema_version"] != 1:
        raise ValueError("expected pilot configuration schema_version 1")
    setups = config["scenarios"]
    if not setups or len({setup["scenario_id"] for setup in setups}) != len(setups):
        raise ValueError("expected nonempty scenarios with unique IDs")
    plant = MujocoPlant(model_path)
    predictor = AnalyticalRK4Predictor(
        expected_plant_parameters(NOMINAL_CONTEXT.environment_version), PHYSICS_INTERVAL_S,
    )
    pd = PDController(**config["pd"])
    mpc = config["mpc"]
    weights = CostWeights(**mpc["weights"])
    records = []
    for setup in setups:
        seed = setup["seed"]
        rng = np.random.default_rng(seed)
        scenario = Scenario(
            setup["scenario_id"], DynamicsState(setup["position_rad"], setup["velocity_rad_s"]),
            config["target_position_rad"], config["steps"],
        )

        def mpc_controller(state: DynamicsState, target: float) -> ControlDecision:
            proposal = plan(
                predictor, NOMINAL_CONTEXT, state, target, weights,
                horizon=mpc["horizon"], candidate_count=mpc["candidate_count"], rng=rng,
            )
            torque = proposal.actions_nm[0] if proposal.actions_nm is not None else None
            return ControlDecision(torque, proposal.predictor_calls)

        for controller_id, controller in (("pd", pd), ("rk4_mpc", mpc_controller)):
            records.append(run_episode(
                plant, scenario, controller, controller_id=controller_id, seed=seed,
                fallback=pd, deadline_ns=deadline_ns,
            ))
    return tuple(records)


def write_evidence(
    output: Path, config: dict, records: tuple[EpisodeRecord, ...], manifest: dict,
) -> None:
    """Save raw episodes, summaries and a plot computed from their state samples."""
    cohorts = []
    for controller_id in ("pd", "rk4_mpc"):
        selected = (record for record in records if record.controller_id == controller_id)
        cohorts.append(asdict(summarize_cohort(selected, **config["metrics"])))
    _write_json(output / "episodes.json", {"episodes": [asdict(record) for record in records]})
    _write_json(output / "metrics.json", {"cohorts": cohorts})
    _plot_pairs(output / "paired.png", records)
    _write_json(output / "manifest.json", manifest)


def _write_json(path: Path, data: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump({**data, "schema_version": 1}, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _plot_pairs(path: Path, records: tuple[EpisodeRecord, ...]) -> None:
    scenarios = dict.fromkeys(record.scenario.scenario_id for record in records)
    figure = Figure(figsize=(8, 2.8 * len(scenarios)), layout="constrained")
    canvas = FigureCanvasAgg(figure)
    panels = figure.subplots(len(scenarios), 1, squeeze=False)[:, 0]
    for scenario_id, axes in zip(scenarios, panels, strict=True):
        for record in records:
            scenario = record.scenario
            if scenario.scenario_id != scenario_id:
                continue
            times = [0.0]
            positions = [scenario.initial_state.position_rad]
            substeps = 0
            for step in record.steps:
                substeps += step.physics_steps
                times.append(substeps * PHYSICS_INTERVAL_S)
                positions.append(step.next_state.position_rad if step.next_state else float("nan"))
            axes.plot(times, positions, label=f"{record.controller_id} ({record.termination})")
        axes.axhline(scenario.target_position_rad, color="0.4", linestyle="--", label="Target")
        axes.set(title=scenario_id, xlabel="Simulated time (s)", ylabel="Hinge position (rad)")
        axes.legend()
    with path.open("xb") as stream:
        canvas.print_png(stream)
