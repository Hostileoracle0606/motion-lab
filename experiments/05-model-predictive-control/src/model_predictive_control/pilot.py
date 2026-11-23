"""Run and report the paired, nominal one-link development pilot."""

from collections.abc import Iterator
from dataclasses import asdict
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np

from forward_dynamics.analytical import AnalyticalRK4Predictor
from forward_dynamics.plant import expected_plant_parameters
from forward_dynamics.types import DynamicsState
from model_predictive_control.attempts import audit_run, identity, record_started, write_roster
from model_predictive_control.cem import plan_cem
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
    return tuple(_iter_pilot(config, model_path, deadline_ns=deadline_ns))


def _iter_pilot(
    config: dict, model_path: Path, *, deadline_ns: int | None, on_start=None,
) -> Iterator[EpisodeRecord]:
    """Run configured controllers from the same reset and scenario seed."""
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
    for setup in setups:
        seed = setup["seed"]
        scenario = Scenario(
            setup["scenario_id"], DynamicsState(setup["position_rad"], setup["velocity_rad_s"]),
            config["target_position_rad"], config["steps"],
        )

        shooting = planning_controller(plan, predictor, weights, mpc, seed, {})
        controllers = [("pd", pd), ("rk4_mpc", shooting)]
        if "cem" in mpc:
            cem = planning_controller(plan_cem, predictor, weights, mpc, seed, mpc["cem"])
            controllers.append(("rk4_cem", cem))
        for controller_id, controller in controllers:
            if on_start is not None:
                on_start(identity(scenario.scenario_id, controller_id, seed))
            yield run_episode(
                plant, scenario, controller, controller_id=controller_id, seed=seed,
                fallback=pd, deadline_ns=deadline_ns,
            )


def planning_controller(optimizer, predictor, weights, mpc, seed, parameters):
    rng = np.random.default_rng(seed)

    def choose(state: DynamicsState, target: float) -> ControlDecision:
        proposal = optimizer(
            predictor, NOMINAL_CONTEXT, state, target, weights,
            horizon=mpc["horizon"], candidate_count=mpc["candidate_count"], rng=rng,
            **parameters,
        )
        torque = proposal.actions_nm[0] if proposal.actions_nm is not None else None
        return ControlDecision(
            torque, proposal.predictor_calls, proposal.candidates,
            proposal.candidates - proposal.feasible_candidates,
        )

    return choose


def _controller_ids(config: dict) -> tuple[str, ...]:
    if "cem" in config["mpc"]:
        return ("pd", "rk4_mpc", "rk4_cem")
    return ("pd", "rk4_mpc")


def run_recorded_pilot(
    output: Path, config: dict, model_path: Path, manifest: dict, *,
    deadline_ns: int | None = None,
) -> int:
    """Preserve completed episodes as they finish and propagate unexpected errors."""
    roster = [identity(setup["scenario_id"], controller_id, setup["seed"])
              for setup in config["scenarios"] for controller_id in _controller_ids(config)]
    return record_run(
        output, config, manifest, roster,
        lambda on_start: _iter_pilot(config, model_path, deadline_ns=deadline_ns, on_start=on_start),
    )


def record_run(output: Path, config: dict, manifest: dict, roster: list[dict], make_episodes) -> int:
    """Journal the shared attempt roster and publish summaries after a complete run."""
    output.mkdir(parents=True, exist_ok=False)
    status_path = output / "run-status.json"
    expected = len(roster)
    _write_json(status_path, {"status": "in_progress", "expected_episodes": expected})
    records = []
    try:
        _write_json(output / "manifest.json", manifest)
        write_roster(output, roster)
        for record in make_episodes(lambda attempt: record_started(output, attempt)):
            entry = {"schema_version": 1, "episode": asdict(record)}
            line = json.dumps(entry, allow_nan=False) + "\n"
            with (output / "completed-episodes.jsonl").open("a", encoding="utf-8") as journal:
                journal.write(line)
            records.append(record)
        audit = audit_run(output, require_complete=False)
        if (audit["errors"] or any(audit["truncated"].values())
                or audit["started"] != audit["planned"] or audit["completed"] != audit["planned"]):
            raise ValueError(f"incomplete attempt evidence: {audit}")
        with TemporaryDirectory(prefix=".results-", dir=output) as staged:
            staged_path = Path(staged)
            _write_results(staged_path, config, tuple(records))
            for name in ("episodes.json", "metrics.json", "paired.png"):
                (staged_path / name).replace(output / name)
    except BaseException as error:
        try:
            _replace_status(status_path, {
                "status": "aborted", "completed_episodes": len(records),
                "expected_episodes": expected, "exception_type": type(error).__name__,
            })
        except OSError as status_error:
            error.add_note(f"Could not record abort status: {status_error}")
        raise
    _replace_status(status_path, {
        "status": "complete", "completed_episodes": len(records), "expected_episodes": expected,
    })
    return len(records)


def write_evidence(
    output: Path, config: dict, records: tuple[EpisodeRecord, ...], manifest: dict,
) -> None:
    """Save raw episodes, summaries and a plot computed from their state samples."""
    _write_results(output, config, records)
    _write_json(output / "manifest.json", manifest)


def _write_results(output: Path, config: dict, records: tuple[EpisodeRecord, ...]) -> None:
    cohorts = []
    for controller_id in dict.fromkeys(record.controller_id for record in records):
        selected = (record for record in records if record.controller_id == controller_id)
        cohorts.append(asdict(summarize_cohort(selected, **config["metrics"])))
    _write_json(output / "episodes.json", {"episodes": [asdict(record) for record in records]})
    _write_json(output / "metrics.json", {"cohorts": cohorts})
    _plot_pairs(output / "paired.png", records)


def _replace_status(path: Path, data: dict) -> None:
    temporary = path.with_suffix(".tmp")
    _write_json(temporary, data)
    temporary.replace(path)


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
