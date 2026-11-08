"""Validate, evaluate, stage, audit and publish a fresh prediction bundle."""

from dataclasses import asdict
from pathlib import Path
import tempfile

from forward_dynamics.adapters import make_predictor
from forward_dynamics.config import EvaluationConfig, REPOSITORY_ROOT, load_evaluation_config
from forward_dynamics.dataset import load_prediction_cases, validate_case_contexts
from forward_dynamics.evaluation import evaluate_one_step
from forward_dynamics.plant import expected_plant_parameters, load_plant_parameters
from forward_dynamics.rollouts import evaluate_rollouts
from forward_dynamics.evidence import evaluation_mapping, load_evaluation_evidence, write_evidence
from forward_dynamics.provenance import capture_provenance
from forward_dynamics.serialization import require_equal
from forward_dynamics.types import Predictor, DynamicsState, DynamicsContext, validate_prediction_inputs


def run_evaluation(
    config_path: Path, output_dir: Path, *, allow_dirty: bool = False,
) -> Path:
    """Return manifest.json; never overwrite an existing evidence directory."""
    config_path = config_path.resolve()
    config = load_evaluation_config(config_path)
    output = output_dir.resolve()
    _validate_output(output, config_path, config)
    if output.exists():
        raise ValueError("output directory already exists; choose a fresh output path")
    provenance = capture_provenance(config_path, config, allow_dirty)
    cases = load_prediction_cases(config.dataset_dir, config.dataset_version)
    validate_case_contexts(cases, config.environment_id, config.environment_version)
    if config.scope == "full":
        require_equal(asdict(load_plant_parameters(config.model_path, config.environment_version)),
                      asdict(expected_plant_parameters(config.environment_version)), "compiled plant parameters")
    predictions = []
    rollouts = []
    for name in config.predictor_ids:
        predictor = make_predictor(name, config)
        if name == "mujoco" and config.scope == "full":
            predictor = _RunReferenceCache(predictor)
        predictions.extend(evaluate_one_step(cases, predictor))
        if config.scope == "full":
            rollouts.extend(evaluate_rollouts(cases, predictor, config.horizons).predictions)
    predictions, rollouts = tuple(predictions), tuple(rollouts)
    # Detect input, source, HEAD or dirty-state changes during evaluation.
    require_equal(capture_provenance(config_path, config, allow_dirty), provenance, "inputs changed during evaluation")
    payload = evaluation_mapping(config, cases, predictions, provenance, rollouts)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".one-step-staging-", dir=output.parent) as staging:
        stage = Path(staging)
        write_evidence(stage, payload, predictions, rollouts)
        load_evaluation_evidence(stage)
        if output.exists():
            raise ValueError("output directory was created during evaluation")
        stage.rename(output)
    return output / "manifest.json"


class _RunReferenceCache:
    """Run-local memoryless reference memo; values come only from real queries.

    Constructed once per run and discarded afterwards. Keys contain every
    predictor input and never observed targets, case IDs or residuals.
    """

    def __init__(self, predictor: Predictor) -> None:
        self.predictor_id = predictor.predictor_id
        self._predictor = predictor
        self._cache = {}

    def predict(self, state: DynamicsState, applied_torque_nm: float,
                context: DynamicsContext) -> DynamicsState:
        validate_prediction_inputs(state, applied_torque_nm, context)
        key = (state, applied_torque_nm, context)
        if key not in self._cache:
            self._cache[key] = self._predictor.predict(state, applied_torque_nm, context)
        return self._cache[key]


def _validate_output(output: Path, config_path: Path, config: EvaluationConfig) -> None:
    protected = [config.dataset_dir, config.model_path.parent, config_path.parent,
                 REPOSITORY_ROOT / "uv.lock", REPOSITORY_ROOT / "pyproject.toml"]
    for name in ("01-feedback-control", "02-state-transition-dataset", "03-forward-dynamics"):
        protected.append(REPOSITORY_ROOT / "experiments" / name / "src")
        protected.append(REPOSITORY_ROOT / "experiments" / name / "configs")
    experiment = REPOSITORY_ROOT / "experiments/03-forward-dynamics"
    protected.extend(experiment / name for name in ("scripts", "tests"))
    for source in protected:
        source = source.resolve()
        if output == source or output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError("input and output trees must not overlap")
