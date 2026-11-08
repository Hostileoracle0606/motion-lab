"""Independent audit of saved predictions, source identity and derivations."""

from dataclasses import asdict, dataclass
from pathlib import Path

from forward_dynamics.adapters import adapter_descriptions, make_predictor
from forward_dynamics.config import EvaluationConfig, load_evaluation_config, protocol_mapping
from forward_dynamics.dataset import PredictionCase, load_prediction_cases, validate_case_contexts
from forward_dynamics.evaluation import OneStepPrediction, summarize_predictions
from forward_dynamics.metrics import residual
from forward_dynamics.plant import PlantParameters, expected_plant_parameters
from forward_dynamics.rollouts import RolloutPrediction, WindowCount
from forward_dynamics.rollout_evidence import (
    episode_outcomes, window_counts, summarize_rollouts, validate_rollout_records,
    validate_rollout_baselines,
)
from forward_dynamics.provenance import input_path, validate_provenance
from forward_dynamics.serialization import (
    decode_object, json_bytes, require_equal, require_fields, sha256_file,
)
from forward_dynamics.types import DynamicsState, validate_state


@dataclass(frozen=True)
class EvaluationEvidence:
    input_dir: Path
    evaluation: dict
    predictions: tuple[OneStepPrediction, ...]
    rollouts: tuple[RolloutPrediction, ...] = ()
    windows: tuple[WindowCount, ...] = ()
    plant_parameters: PlantParameters | None = None


def evaluation_mapping(
    config: EvaluationConfig, cases: tuple[PredictionCase, ...],
    predictions: tuple[OneStepPrediction, ...], provenance: dict,
    rollouts: tuple[RolloutPrediction, ...] = (),
) -> dict:
    payload = {
        "schema_version": 1, "scope": config.scope, "protocol": protocol_mapping(config),
        "provenance": provenance, "adapters": adapter_descriptions(config),
        "case_count": len(cases), "prediction_count": len(predictions),
        "split_counts": {split: sum(case.split == split for case in cases)
                         for split in ("train", "validation", "test")},
        "aggregates": summarize_predictions(predictions),
    }
    if config.scope == "full":
        payload.update(rollout_prediction_count=len(rollouts),
            windows=tuple(asdict(row) for row in window_counts(cases, config.horizons)),
            rollout_aggregates=summarize_rollouts(rollouts),
            episode_outcomes=episode_outcomes(cases),
            plant_parameters=asdict(expected_plant_parameters(config.environment_version)))
    return payload


def write_evidence(
    output: Path, evaluation: dict, predictions: tuple[OneStepPrediction, ...],
    rollouts: tuple[RolloutPrediction, ...] = (),
) -> None:
    """Write into a fresh staging directory; caller validates before publishing."""
    (output / "evaluation.json").write_bytes(json_bytes(evaluation, indent=2))
    (output / "one_step.jsonl").write_bytes(b"".join(json_bytes(asdict(row)) for row in predictions))
    if evaluation["scope"] == "full":
        (output / "rollouts.jsonl").write_bytes(b"".join(json_bytes(asdict(row)) for row in rollouts))
    (output / "manifest.json").write_bytes(json_bytes(_manifest(output), indent=2))


def _manifest(output: Path) -> dict:
    scope = decode_object((output / "evaluation.json").read_text())["scope"]
    names = ("evaluation.json", "one_step.jsonl") + (("rollouts.jsonl",) if scope == "full" else ())
    return {
        "schema_version": 1, "scope": scope,
        "files": {name: {"sha256": sha256_file(output / name)}
                  for name in names},
    }


def load_evaluation_evidence(input_dir: Path) -> EvaluationEvidence:
    """Audit saved records against their original checked-out inputs.

    This does not call simulator dynamics. Reference predictions are checked
    against recorded next states; other baselines are checked algebraically.
    Checksums establish byte identity, not authenticity of an untrusted author.
    """
    output = input_dir.resolve()
    try:
        manifest = decode_object((output / "manifest.json").read_text())
        require_equal(manifest, _manifest(output), "evidence manifest/checksums")
        payload = decode_object((output / "evaluation.json").read_text())
        fields = {
            "schema_version", "scope", "protocol", "provenance", "adapters",
            "case_count", "prediction_count", "split_counts", "aggregates",
        }
        if payload.get("scope") == "full":
            fields.update({"rollout_prediction_count", "windows", "rollout_aggregates", "episode_outcomes", "plant_parameters"})
        require_fields(payload, fields, "evaluation")
        require_equal(payload["schema_version"], 1, "evaluation schema_version")
        provenance = payload["provenance"]
        if not isinstance(provenance, dict):
            raise ValueError("provenance must be an object")
        config_path = input_path(provenance.get("config_file"))
        config = load_evaluation_config(config_path)
        require_equal(payload["scope"], config.scope, "evaluation scope")
        validate_provenance(provenance, config_path, config)
        require_equal(payload["protocol"], protocol_mapping(config), "recorded protocol")
        cases = load_prediction_cases(config.dataset_dir, config.dataset_version)
        validate_case_contexts(cases, config.environment_id, config.environment_version)
        records = [decode_object(line) for line in (output / "one_step.jsonl").read_text().splitlines()]
        expected_keys = [(name, case) for name in config.predictor_ids for case in cases]
        if len(records) != len(expected_keys):
            raise ValueError("prediction membership/count mismatch")
        rows = tuple(
            _validate_record(record, name, case, config)
            for record, (name, case) in zip(records, expected_keys, strict=True)
        )
        rollouts = ()
        windows = ()
        parameters = None
        if config.scope == "full":
            parameters = expected_plant_parameters(config.environment_version)
            require_equal(payload["plant_parameters"], asdict(parameters), "independent plant parameters")
            records = [decode_object(line) for line in (output / "rollouts.jsonl").read_text().splitlines()]
            rollouts = validate_rollout_records(records, cases, config)
            windows = window_counts(cases, config.horizons)
        require_equal(payload, evaluation_mapping(config, cases, rows, provenance, rollouts), "evaluation derivations/counts/aggregates")
        if config.scope == "full":
            validate_rollout_baselines(rollouts, cases, config)
        return EvaluationEvidence(output, payload, rows, rollouts, windows, parameters)
    except (OSError, UnicodeError) as error:
        raise ValueError("cannot read complete evaluation evidence") from error


def _validate_record(
    payload: dict, predictor_id: str, case: PredictionCase, config: EvaluationConfig,
) -> OneStepPrediction:
    require_fields(payload, {"case", "predictor_id", "predicted_next_state", "residual"}, "prediction")
    require_equal(payload["case"], asdict(case), "prediction case membership/expected state")
    require_equal(payload["predictor_id"], predictor_id, "prediction adapter membership")
    state = payload["predicted_next_state"]
    require_fields(state, {"position_rad", "velocity_rad_s"}, "predicted state")
    predicted = DynamicsState(**state)
    validate_state(predicted)
    error = residual(predicted, case.expected_next_state)
    require_equal(payload["residual"], asdict(error), "prediction residual")
    if predictor_id == "mujoco":
        if (abs(error.position_error_rad) > config.simulator_absolute_tolerance or
                abs(error.velocity_error_rad_s) > config.simulator_absolute_tolerance):
            raise ValueError("simulator reference exceeds recorded-state tolerance")
    else:
        predictor = make_predictor(predictor_id, config)
        expected = predictor.predict(case.state, case.applied_torque_nm, case.context)
        require_equal(state, asdict(expected), "baseline prediction")
    return OneStepPrediction(case, predictor_id, predicted, error)
