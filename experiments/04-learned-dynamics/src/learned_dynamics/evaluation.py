"""Pure Stage 4 attempts, recursion, coverage and physical-unit summaries."""

from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
import math

from forward_dynamics.dataset import PredictionCase
from forward_dynamics.metrics import StateResidual, residual, summarize_errors
from forward_dynamics.rollouts import _validated_episodes
from forward_dynamics.types import DynamicsState, validate_state
from learned_dynamics.config import RunSpec
from learned_dynamics.types import ModelSpec, validate_hash


MEMBERSHIPS = ("all", "history_common", "loss_common")
BASELINE_IDS = (
    "mujoco", "no_change", "constant_velocity",
    "analytical_rk4_0.002", "analytical_rk4_0.001", "analytical_rk4_0.0005",
)


@dataclass(frozen=True)
class EvaluationCandidate:
    predictor_id: str
    predictor: object
    kind: str
    run_spec: dict | None
    checkpoint_dir: str | None
    checkpoint_sha256: str | None
    selected_epoch: int | None
    selected_validation_mse: float | None
    training_config_sha256: str | None
    training_failure_reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.predictor_id, str) or not self.predictor_id:
            raise ValueError("predictor_id must be nonempty")
        if self.kind == "baseline":
            if any(value is not None for value in (
                    self.run_spec, self.checkpoint_dir, self.checkpoint_sha256,
                    self.selected_epoch, self.selected_validation_mse,
                    self.training_config_sha256, self.training_failure_reason)):
                raise ValueError("baseline identity cannot contain training fields")
        elif self.kind in ("learned", "learned_failed"):
            if not isinstance(self.run_spec, dict):
                raise ValueError("learned identity requires a run specification")
            model = self.run_spec.get("model")
            if not isinstance(model, dict):
                raise ValueError("learned run model is missing")
            run = RunSpec(
                self.run_spec.get("variant_id"), ModelSpec(**model),
                self.run_spec.get("loss"), self.run_spec.get("membership"),
                self.run_spec.get("seed"), self.run_spec.get("device"),
                self.run_spec.get("repeat"),
            )
            if run.mapping() != self.run_spec:
                raise ValueError("learned run specification is not canonical")
            validate_hash(self.training_config_sha256)
            if self.kind == "learned":
                if (not isinstance(self.checkpoint_dir, str) or not self.checkpoint_dir or
                        Path(self.checkpoint_dir).is_absolute() or
                        ".." in Path(self.checkpoint_dir).parts):
                    raise ValueError("learned checkpoint_dir must be local and relative")
                validate_hash(self.checkpoint_sha256)
                if type(self.selected_epoch) is not int or self.selected_epoch <= 0:
                    raise ValueError("learned selected_epoch must be positive")
                if (type(self.selected_validation_mse) not in (int, float) or
                        not math.isfinite(self.selected_validation_mse) or
                        self.selected_validation_mse < 0):
                    raise ValueError("learned selected metric must be finite and nonnegative")
                if self.training_failure_reason is not None:
                    raise ValueError("successful learned candidate cannot have a failure reason")
            elif (any(value is not None for value in (
                    self.checkpoint_dir, self.checkpoint_sha256, self.selected_epoch,
                    self.selected_validation_mse)) or
                    not isinstance(self.training_failure_reason, str) or
                    not self.training_failure_reason.strip()):
                raise ValueError("failed learned candidate must retain only its failure reason")
        else:
            raise ValueError("candidate kind must be baseline or learned")

    @classmethod
    def baseline(cls, predictor_id: str, predictor: object) -> "EvaluationCandidate":
        return cls(predictor_id, predictor, "baseline", None, None, None, None, None, None, None)

    @classmethod
    def learned(cls, *, predictor_id: str, predictor: object, run_spec: dict,
                checkpoint_dir: str, checkpoint_sha256: str, selected_epoch: int,
                selected_validation_mse: float,
                training_config_sha256: str) -> "EvaluationCandidate":
        return cls(predictor_id, predictor, "learned", run_spec, checkpoint_dir,
                   checkpoint_sha256, selected_epoch, selected_validation_mse,
                   training_config_sha256, None)

    @classmethod
    def failed(cls, *, predictor_id: str, predictor: object, run_spec: dict,
               training_config_sha256: str, failure_reason: str) -> "EvaluationCandidate":
        return cls(predictor_id, predictor, "learned_failed", run_spec, None, None,
                   None, None, training_config_sha256, failure_reason)

    def identity(self) -> dict:
        return {
            "predictor_id": self.predictor_id, "kind": self.kind,
            "run_spec": self.run_spec, "checkpoint_dir": self.checkpoint_dir,
            "checkpoint_sha256": self.checkpoint_sha256,
            "selected_epoch": self.selected_epoch,
            "selected_validation_mse": self.selected_validation_mse,
            "training_config_sha256": self.training_config_sha256,
            "training_failure_reason": self.training_failure_reason,
        }


def evaluate(cases: tuple[PredictionCase, ...], predictors: tuple[EvaluationCandidate, ...],
             horizons: tuple[int, ...], membership: str) -> tuple[dict, ...]:
    """Return every attempted start, including unavailable and failed attempts."""
    if membership not in MEMBERSHIPS:
        raise ValueError("unsupported evaluation membership")
    if (not predictors or any(not isinstance(row, EvaluationCandidate) for row in predictors) or
            len({row.predictor_id for row in predictors}) != len(predictors)):
        raise ValueError("evaluation candidates must be nonempty and uniquely identified")
    if (not horizons or tuple(sorted(set(horizons))) != horizons or
            any(type(value) is not int or value <= 0 for value in horizons)):
        raise ValueError("horizons must be increasing unique positive integers")
    episodes = _validated_episodes(cases, "stage4")
    records = []
    for candidate in predictors:
        history_steps = (candidate.run_spec or {}).get("model", {}).get("history_steps", 0)
        declared_membership = (candidate.run_spec or {}).get("membership")
        if candidate.kind != "baseline" and declared_membership != membership:
            raise ValueError("learned candidate membership disagrees with evaluation membership")
        if history_steps and membership != "history_common":
            raise ValueError("history predictor requires history_common membership")
        for episode in episodes:
            for horizon in horizons:
                for start, case in enumerate(episode):
                    records.append(_attempt(candidate, episode, start, horizon,
                                            membership, history_steps))
    return tuple(records)


def _attempt(candidate: EvaluationCandidate, episode: tuple[PredictionCase, ...],
             start: int, horizon: int, membership: str, history_steps: int) -> dict:
    case = episode[start]
    base = {
        "predictor_id": candidate.predictor_id, "candidate": candidate.identity(),
        "membership": membership, "split": case.split, "episode_id": case.episode_id,
        "scenario_id": case.scenario_id, "controller_id": case.controller_id,
        "data_seed": case.seed, "start_step_index": case.step_index,
        "horizon_steps": horizon,
        "horizon_s": horizon * float(case.context.control_interval_s),
    }
    reason = None
    if membership == "history_common" and start == 0:
        reason = "history_common requires a previous row"
    required = max(horizon, 5 if membership == "loss_common" else horizon)
    if reason is None and len(episode) - start < required:
        reason = "insufficient episode-local future rows"
    if reason is not None:
        return base | {
            "status": "unavailable", "unavailable_reason": reason,
            "applied_actions_nm": [], "expected_endpoint_state": None,
            "predicted_endpoint_state": None, "residual": None,
            "failure_step": None, "failure_reason": None,
        }
    window = episode[start:start + horizon]
    actions = [float(row.applied_torque_nm) for row in window]
    expected = window[-1].expected_next_state
    state = DynamicsState(float(case.state.position_rad), float(case.state.velocity_rad_s))
    previous_state = episode[start - 1].state if start else None
    previous_action = episode[start - 1].applied_torque_nm if start else None
    for offset, action_case in enumerate(window, start=1):
        try:
            if candidate.kind == "learned_failed":
                raise RuntimeError(f"training failed: {candidate.training_failure_reason}")
            before = state
            if history_steps:
                state = candidate.predictor.predict_with_history(
                    state, float(action_case.applied_torque_nm), previous_state,
                    float(previous_action), action_case.context)
                previous_state, previous_action = before, action_case.applied_torque_nm
            else:
                state = candidate.predictor.predict(
                    state, float(action_case.applied_torque_nm), action_case.context)
            validate_state(state)
            state = DynamicsState(float(state.position_rad), float(state.velocity_rad_s))
        except (ValueError, TypeError, ArithmeticError, RuntimeError, NotImplementedError) as error:
            return base | {
                "status": "failed", "unavailable_reason": None,
                "applied_actions_nm": actions,
                "expected_endpoint_state": asdict(expected),
                "predicted_endpoint_state": None, "residual": None,
                "failure_step": offset,
                "failure_reason": f"{type(error).__name__}: {error}",
            }
    error = residual(state, expected)
    return base | {
        "status": "success", "unavailable_reason": None,
        "applied_actions_nm": actions, "expected_endpoint_state": asdict(expected),
        "predicted_endpoint_state": asdict(state), "residual": asdict(error),
        "failure_step": None, "failure_reason": None,
    }


def summarize_records(records: tuple[dict, ...]) -> tuple[dict, ...]:
    if not records:
        raise ValueError("evaluation records cannot be empty")
    grouped = defaultdict(list)
    for row in records:
        grouped[(row["predictor_id"], row["membership"], row["horizon_steps"])].append(row)
    result = []
    for (predictor_id, membership, horizon), rows in sorted(grouped.items()):
        successes = [row for row in rows if row["status"] == "success"]
        failures = [row for row in rows if row["status"] == "failed"]
        unavailable = [row for row in rows if row["status"] == "unavailable"]
        errors = tuple(StateResidual(**row["residual"]) for row in successes)
        result.append({
            "predictor_id": predictor_id, "membership": membership,
            "horizon_steps": horizon,
            "units": {"position": "rad", "velocity": "rad/s", "horizon": "s"},
            "counts": {"attempted": len(rows), "eligible": len(successes) + len(failures),
                       "success": len(successes), "failed": len(failures),
                       "unavailable": len(unavailable)},
            "success_metrics": asdict(summarize_errors(errors)) if errors else None,
        })
    return tuple(result)


def run_evaluation(config_path: Path, roster_path: Path, output_dir: Path) -> Path:
    """Evaluate the frozen test protocol; callers gate full-study execution."""
    from forward_dynamics.adapters import make_predictor
    from forward_dynamics.config import DEFAULT_CONFIG, load_evaluation_config as load_stage3_config
    from learned_dynamics.checkpoint import load_checkpoint
    from learned_dynamics.data import load_views
    from learned_dynamics.evaluation_config import load_evaluation_config
    from learned_dynamics.evidence import write_evidence
    from learned_dynamics.predictor import LearnedPredictor, SUPPORTED_CONTEXT
    from learned_dynamics.training_evidence import load_run_manifest

    config = load_evaluation_config(config_path)
    if Path(roster_path).resolve() != config.training_roster_dir:
        raise ValueError("roster path must equal the configured training roster")
    if Path(output_dir).resolve() != config.output_dir:
        raise ValueError("output path must equal the configured evaluation output")
    full_roster = load_run_manifest(config.training_roster_dir)
    roster = {key: full_roster[key] for key in (
        "schema_version", "training_config_sha256", "planned_runs", "results")}
    views = load_views(config.dataset_dir, config.dataset_version)
    cases = views.test.cases
    stage3 = load_stage3_config(DEFAULT_CONFIG)
    baseline_predictors = {name: make_predictor(name, stage3) for name in config.baseline_ids}
    planned = {row["run_id"]: row["run_spec"] for row in roster["planned_runs"]}
    records = []
    for membership in config.memberships:
        candidates = [EvaluationCandidate.baseline(name, baseline_predictors[name])
                      for name in config.baseline_ids]
        for result in roster["results"]:
            run_spec = planned[result["run_id"]]
            if run_spec["membership"] != membership:
                continue
            if result["status"] == "failed":
                candidates.append(EvaluationCandidate.failed(
                    predictor_id=result["run_id"], predictor=object(), run_spec=run_spec,
                    training_config_sha256=roster["training_config_sha256"],
                    failure_reason=result["failure_reason"]))
                continue
            checkpoint_dir = config.training_roster_dir / result["checkpoint_dir"]
            bundle = load_checkpoint(checkpoint_dir, device=run_spec["device"])
            predictor = LearnedPredictor(bundle.model, bundle.normalizer, SUPPORTED_CONTEXT,
                                         result["run_id"], run_spec["device"])
            candidates.append(EvaluationCandidate.learned(
                predictor_id=result["run_id"], predictor=predictor, run_spec=run_spec,
                checkpoint_dir=result["checkpoint_dir"],
                checkpoint_sha256=result["checkpoint_sha256"],
                selected_epoch=result["selected_epoch"],
                selected_validation_mse=result["selected_validation_mse"],
                training_config_sha256=roster["training_config_sha256"]))
        records.extend(evaluate(cases, tuple(candidates), config.horizons, membership))
    return write_evidence(config.output_dir, config, roster, tuple(records), cases)
