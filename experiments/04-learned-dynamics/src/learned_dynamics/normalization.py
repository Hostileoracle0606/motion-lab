"""Training-only scaling, immutable fitted parameters and strict finite JSON."""

from dataclasses import asdict
from pathlib import Path

import numpy as np

from forward_dynamics.serialization import decode_object, json_bytes, require_fields
from learned_dynamics.data import training_arrays
from learned_dynamics.types import (
    CONSTANT_THRESHOLD, FEATURE_ORDER, FEATURE_UNITS, Normalizer, TrainingView,
)


def fit_normalizer(view: TrainingView) -> Normalizer:
    """Fit float64 population statistics on training rows, never an evaluation view."""
    inputs, deltas = training_arrays(view)
    with np.errstate(over="raise", invalid="raise"):
        try:
            x_mean, x_std = inputs.mean(axis=0), inputs.std(axis=0, ddof=0)
            y_mean, y_std = deltas.mean(axis=0), deltas.std(axis=0, ddof=0)
        except FloatingPointError as error:
            raise ValueError("normalization statistics exceed finite float64 range") from error
    x_constant, y_constant = x_std < CONSTANT_THRESHOLD, y_std < CONSTANT_THRESHOLD
    return Normalizer(
        schema_version=1, feature_order=FEATURE_ORDER, feature_units=FEATURE_UNITS,
        input_mean=tuple(float(v) for v in x_mean),
        input_scale=tuple(float(v) for v in np.where(x_constant, 1.0, x_std)),
        input_constant=tuple(bool(v) for v in x_constant),
        delta_mean=tuple(float(v) for v in y_mean),
        delta_scale=tuple(float(v) for v in np.where(y_constant, 1.0, y_std)),
        delta_constant=tuple(bool(v) for v in y_constant),
        fit_row_count=len(view.cases), dataset_version=view.view.dataset_version,
        manifest_sha256=view.view.manifest_sha256, train_sha256=view.view.train_sha256,
        fit_keys=tuple((c.episode_id, c.step_index) for c in view.cases),
    )


def _array(values: np.ndarray, widths: tuple[int, ...]) -> np.ndarray:
    if (not isinstance(values, np.ndarray) or values.ndim != 2 or
            values.shape[0] == 0 or values.shape[1] not in widths or
            values.dtype.kind not in "iuf"):
        raise ValueError(f"values must be a nonempty real numeric [N,{widths}] array")
    with np.errstate(over="ignore", invalid="ignore"):
        result = values.astype(np.float64)
    if not np.isfinite(result).all():
        raise ValueError("array values must be finite float64")
    return result


def transform_inputs(values: np.ndarray, normalizer: Normalizer) -> np.ndarray:
    """Scale canonical [N,3] or explicit current+previous [N,6] input; never clip."""
    if not isinstance(normalizer, Normalizer):
        raise ValueError("normalizer must be a Normalizer")
    values = _array(values, (3, 6))
    repeats = values.shape[1] // 3
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        result = (values - np.tile(normalizer.input_mean, repeats)) / np.tile(normalizer.input_scale, repeats)
    if not np.isfinite(result).all():
        raise ValueError("transformed inputs must be finite")
    return result


def inverse_delta(values: np.ndarray, normalizer: Normalizer) -> np.ndarray:
    """Restore normalized [N,2] outputs to physical delta rad and rad/s."""
    if not isinstance(normalizer, Normalizer):
        raise ValueError("normalizer must be a Normalizer")
    values = _array(values, (2,))
    with np.errstate(over="ignore", invalid="ignore"):
        result = values * normalizer.delta_scale + normalizer.delta_mean
    if not np.isfinite(result).all():
        raise ValueError("restored deltas must be finite")
    return result


def write_normalizer(normalizer: Normalizer, path: Path) -> None:
    """Write deterministic finite JSON without overwriting an existing artifact."""
    if not isinstance(normalizer, Normalizer):
        raise ValueError("normalizer must be a Normalizer")
    content = json_bytes(asdict(normalizer), indent=2)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(content)
    except FileExistsError as error:
        raise ValueError("normalizer output already exists") from error


def load_normalizer(path: Path) -> Normalizer:
    """Validate structure/identity fields; upstream bundle audit binds the source hashes."""
    try:
        payload = decode_object(Path(path).read_text())
        require_fields(payload, set(Normalizer.__dataclass_fields__), "normalizer")
        for field in ("feature_order", "feature_units", "input_mean", "input_scale",
                      "input_constant", "delta_mean", "delta_scale", "delta_constant", "fit_keys"):
            if not isinstance(payload[field], list):
                raise ValueError(f"normalizer {field} must be a JSON array")
            if field == "fit_keys":
                if any(not isinstance(key, list) for key in payload[field]):
                    raise ValueError("fit_keys must be JSON arrays")
                payload[field] = tuple(tuple(key) for key in payload[field])
            else:
                payload[field] = tuple(payload[field])
        return Normalizer(**payload)
    except (OSError, UnicodeError) as error:
        raise ValueError("cannot read normalizer") from error
