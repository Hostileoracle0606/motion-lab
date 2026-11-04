"""Independent audit of saved predictions, source identity and derivations."""

from dataclasses import asdict, dataclass
from pathlib import Path

from forward_dynamics.adapters import adapter_descriptions
from forward_dynamics.baselines import ConstantVelocityPredictor, NoChangePredictor
from forward_dynamics.config import EvaluationConfig, load_evaluation_config, protocol_mapping
from forward_dynamics.dataset import PredictionCase, load_prediction_cases, validate_case_contexts
from forward_dynamics.evaluation import OneStepPrediction, summarize_predictions
from forward_dynamics.metrics import residual
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


def evaluation_mapping(
    config: EvaluationConfig, cases: tuple[PredictionCase, ...],
    predictions: tuple[OneStepPrediction, ...], provenance: dict,
) -> dict:
    return {
        "schema_version": 1, "scope": "one_step", "protocol": protocol_mapping(config),
        "provenance": provenance, "adapters": adapter_descriptions(config),
        "case_count": len(cases), "prediction_count": len(predictions),
        "split_counts": {split: sum(case.split == split for case in cases)
                         for split in ("train", "validation", "test")},
        "aggregates": summarize_predictions(predictions),
    }


def write_evidence(
    output: Path, evaluation: dict, predictions: tuple[OneStepPrediction, ...],
) -> None:
    """Write into a fresh staging directory; caller validates before publishing."""
    (output / "evaluation.json").write_bytes(json_bytes(evaluation, indent=2))
    (output / "one_step.jsonl").write_bytes(b"".join(json_bytes(asdict(row)) for row in predictions))
    (output / "manifest.json").write_bytes(json_bytes(_manifest(output), indent=2))


def _manifest(output: Path) -> dict:
    return {
        "schema_version": 1, "scope": "one_step",
        "files": {name: {"sha256": sha256_file(output / name)}
                  for name in ("evaluation.json", "one_step.jsonl")},
    }


def load_evaluation_evidence(input_dir: Path) -> EvaluationEvidence:
    """Audit saved records against their original checked-out inputs.

    This does not call simulator dynamics. Reference predictions are checked
    against recorded next states; simple baselines are checked algebraically.
    Checksums establish byte identity, not authenticity of an untrusted author.
    """
    output = input_dir.resolve()
    try:
        manifest = decode_object((output / "manifest.json").read_text())
        require_equal(manifest, _manifest(output), "evidence manifest/checksums")
        payload = decode_object((output / "evaluation.json").read_text())
        require_fields(payload, {
            "schema_version", "scope", "protocol", "provenance", "adapters",
            "case_count", "prediction_count", "split_counts", "aggregates",
        }, "evaluation")
        require_equal(payload["schema_version"], 1, "evaluation schema_version")
        require_equal(payload["scope"], "one_step", "evaluation scope")
        provenance = payload["provenance"]
        if not isinstance(provenance, dict):
            raise ValueError("provenance must be an object")
        config_path = input_path(provenance.get("config_file"))
        config = load_evaluation_config(config_path)
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
        require_equal(payload, evaluation_mapping(config, cases, rows, provenance), "evaluation derivations/counts/aggregates")
        return EvaluationEvidence(output, payload, rows)
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
        predictor = NoChangePredictor() if predictor_id == "no_change" else ConstantVelocityPredictor()
        expected = predictor.predict(case.state, case.applied_torque_nm, case.context)
        require_equal(state, asdict(expected), "simple baseline prediction")
    return OneStepPrediction(case, predictor_id, predicted, error)
