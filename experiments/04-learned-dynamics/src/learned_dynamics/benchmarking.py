"""Synchronized, raw-sample CPU/MPS inference benchmarking."""

import math
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import torch

from forward_dynamics.serialization import json_bytes, sha256_file
from learned_dynamics.checkpoint import load_checkpoint
from learned_dynamics.normalization import transform_inputs
from learned_dynamics.predictor import LearnedPredictor, SUPPORTED_CONTEXT
from learned_dynamics.types import SplitView


WARMUPS = 20
MEASURED = 100
BATCH_SIZES = (1, 256)
ROUND_DEVICE_ORDER = (("cpu", "mps"), ("mps", "cpu"), ("cpu", "mps"))


def summarize_timings(samples: tuple[float, ...]) -> dict:
    if (not isinstance(samples, tuple) or not samples or
            any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0
                for value in samples)):
        raise ValueError("timing samples must be a nonempty tuple of positive finite seconds")
    values = np.asarray(samples, dtype=np.float64)
    return {
        "sample_count": len(samples),
        "median_s": float(np.median(values)),
        "p95_s": float(np.percentile(values, 95, method="linear")),
    }


def _measure_calls(operation, synchronize, clock, *, warmups: int, measured: int) -> tuple[float, ...]:
    if type(warmups) is not int or warmups < 0 or type(measured) is not int or measured <= 0:
        raise ValueError("warmups/measured must be nonnegative/positive integers")
    for _ in range(warmups):
        operation()
        synchronize()
    samples = []
    for _ in range(measured):
        synchronize()
        started = clock()
        operation()
        synchronize()
        elapsed = float(clock()) - float(started)
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise ValueError("timed call duration must be positive and finite")
        samples.append(elapsed)
    return tuple(samples)


def _synchronizer(device: str):
    return torch.mps.synchronize if device == "mps" else (lambda: None)


def _summary(samples: tuple[float, ...], batch_size: int) -> dict:
    result = summarize_timings(samples)
    return result | {
        "batch_latency_s": result["median_s"],
        "items_per_s": float(batch_size / result["median_s"]),
    }


def _benchmark_device(checkpoint_dir: Path, validation: SplitView,
                      device: str, round_index: int) -> tuple[dict, ...]:
    if not isinstance(validation, SplitView) or validation.split != "validation":
        raise ValueError("inference benchmark requires the validation split")
    if len(validation.cases) < max(BATCH_SIZES):
        raise ValueError("validation split is too small for the frozen batch sizes")
    bundle = load_checkpoint(checkpoint_dir, device=device)
    run = bundle.metadata.get("run_spec", {})
    if (run.get("variant_id"), run.get("seed"), run.get("device"), run.get("repeat")) != (
            "canonical32", 0, "mps", 0):
        raise ValueError("benchmark requires the canonical MPS seed-0 repeat-0 checkpoint")
    predictor = LearnedPredictor(bundle.model, bundle.normalizer, SUPPORTED_CONTEXT,
                                 "canonical32-benchmark", device)
    synchronize = _synchronizer(device)
    rows = []
    for batch_size in BATCH_SIZES:
        selected = validation.cases[:batch_size]
        states = np.asarray([(case.state.position_rad, case.state.velocity_rad_s)
                             for case in selected], dtype=np.float64)
        actions = np.asarray([case.applied_torque_nm for case in selected], dtype=np.float64)

        def end_to_end():
            predictor.predict_batch(states, actions, SUPPORTED_CONTEXT)

        end_samples = _measure_calls(end_to_end, synchronize, time.perf_counter,
                                     warmups=WARMUPS, measured=MEASURED)
        rows.append({
            "status": "success", "device": device, "round": round_index,
            "mode": "end_to_end", "batch_size": batch_size, "dtype": "float32",
            "warmup_count": WARMUPS, "samples_s": list(end_samples),
            "summary": _summary(end_samples, batch_size),
        })

        physical = np.column_stack((states, actions))
        normalized = transform_inputs(physical, bundle.normalizer).astype(np.float32)
        resident = torch.from_numpy(normalized).to(device)

        def model_only():
            with torch.inference_mode():
                bundle.model(resident)

        model_samples = _measure_calls(model_only, synchronize, time.perf_counter,
                                       warmups=WARMUPS, measured=MEASURED)
        rows.append({
            "status": "success", "device": device, "round": round_index,
            "mode": "model_only", "batch_size": batch_size, "dtype": "float32",
            "warmup_count": WARMUPS, "samples_s": list(model_samples),
            "summary": _summary(model_samples, batch_size),
        })
    return tuple(rows)


def _source_sha() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[4],
                            capture_output=True, text=True, check=True)
    return result.stdout.strip()


def _environment() -> dict:
    return {
        "os": platform.system(), "os_release": platform.release(),
        "architecture": platform.machine(), "processor": platform.processor() or "unavailable",
        "python": platform.python_version(), "torch": torch.__version__,
        "numpy": np.__version__, "cpu_intraop_threads": torch.get_num_threads(),
        "cpu_interop_threads": torch.get_num_interop_threads(),
        "power_state": os.environ.get("MOTION_LAB_POWER_STATE", "not_recorded"),
        "competing_load": os.environ.get("MOTION_LAB_COMPETING_LOAD", "not_recorded"),
    }


def _pin_cpu_threads() -> None:
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError as error:
            raise ValueError("CPU inter-op threads must be pinned before benchmark work") from error


def benchmark_inference(checkpoint_dir: Path, validation: SplitView, output_dir: Path) -> Path:
    """Run three alternating rounds and retain every raw timing sample."""
    checkpoint = Path(checkpoint_dir).resolve()
    output = Path(output_dir)
    if output.exists() or output.is_symlink():
        raise ValueError("benchmark output must be a fresh directory")
    if not isinstance(validation, SplitView) or validation.split != "validation":
        raise ValueError("inference benchmark requires the validation split")
    manifest = checkpoint / "manifest.json"
    if manifest.is_symlink() or not manifest.is_file():
        raise ValueError("checkpoint manifest is missing or unsafe")
    _pin_cpu_threads()
    results = []
    for round_index, order in enumerate(ROUND_DEVICE_ORDER):
        for device in order:
            try:
                results.extend(_benchmark_device(checkpoint, validation, device, round_index))
            except (ValueError, RuntimeError, NotImplementedError) as error:
                results.append({
                    "status": "failed", "device": device, "round": round_index,
                    "failure_reason": f"{type(error).__name__}: {error}",
                })
    payload = {
        "schema_version": 1, "source_sha": _source_sha(),
        "checkpoint_manifest_sha256": sha256_file(manifest),
        "validation": {
            "dataset_version": validation.dataset_version,
            "manifest_sha256": validation.manifest_sha256,
            "train_sha256": validation.train_sha256,
            "row_count": len(validation.cases),
        },
        "protocol": {
            "batch_sizes": list(BATCH_SIZES), "warmup_count": WARMUPS,
            "measured_count": MEASURED,
            "round_device_order": [list(order) for order in ROUND_DEVICE_ORDER],
            "modes": ["end_to_end", "model_only"],
        },
        "round_device_order": [list(order) for order in ROUND_DEVICE_ORDER],
        "environment": _environment(), "results": results,
    }
    output.mkdir(parents=True)
    benchmark_path = output / "benchmark.json"
    benchmark_path.write_bytes(json_bytes(payload, indent=2))
    (output / "manifest.json").write_bytes(json_bytes({
        "schema_version": 1,
        "files": {"benchmark.json": sha256_file(benchmark_path)},
    }, indent=2))
    return benchmark_path
