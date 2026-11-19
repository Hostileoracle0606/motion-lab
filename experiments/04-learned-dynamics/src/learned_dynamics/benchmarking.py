"""Synchronized, raw-sample CPU/MPS inference benchmarking."""

import math
import os
from pathlib import Path
import platform
import re
import subprocess
import time

import numpy as np
import torch

from forward_dynamics.serialization import (
    decode_object, json_bytes, require_equal, require_fields, sha256_file,
)
from learned_dynamics.checkpoint import load_checkpoint
from learned_dynamics.normalization import transform_inputs
from learned_dynamics.predictor import LearnedPredictor, SUPPORTED_CONTEXT
from learned_dynamics.types import DATASET_VERSION, SplitView, validate_hash


WARMUPS = 20
MEASURED = 100
BATCH_SIZES = (1, 256)
ROUND_DEVICE_ORDER = (("cpu", "mps"), ("mps", "cpu"), ("cpu", "mps"))
BENCHMARK_FIELDS = {
    "schema_version", "source_sha", "checkpoint_manifest_sha256", "validation",
    "protocol", "round_device_order", "environment", "results",
}


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


def _timing_result(device: str, round_index: int, mode: str, batch_size: int,
                   samples: tuple[float, ...]) -> dict:
    return {
        "status": "success", "device": device, "round": round_index,
        "mode": mode, "batch_size": batch_size, "dtype": "float32",
        "warmup_count": WARMUPS, "samples_s": list(samples),
        "summary": _summary(samples, batch_size),
    }


def _failed_workload(device: str, round_index: int, mode: str,
                     batch_size: int, error: Exception) -> dict:
    return {
        "status": "failed", "device": device, "round": round_index,
        "mode": mode, "batch_size": batch_size,
        "failure_reason": f"{type(error).__name__}: {error}",
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

        try:
            end_samples = _measure_calls(end_to_end, synchronize, time.perf_counter,
                                         warmups=WARMUPS, measured=MEASURED)
            rows.append(_timing_result(
                device, round_index, "end_to_end", batch_size, end_samples))
        except (ValueError, RuntimeError, NotImplementedError) as error:
            rows.append(_failed_workload(
                device, round_index, "end_to_end", batch_size, error))

        try:
            physical = np.column_stack((states, actions))
            normalized = transform_inputs(physical, bundle.normalizer).astype(np.float32)
            resident = torch.from_numpy(normalized).to(device)

            def model_only():
                with torch.inference_mode():
                    bundle.model(resident)

            model_samples = _measure_calls(model_only, synchronize, time.perf_counter,
                                           warmups=WARMUPS, measured=MEASURED)
            rows.append(_timing_result(
                device, round_index, "model_only", batch_size, model_samples))
        except (ValueError, RuntimeError, NotImplementedError) as error:
            rows.append(_failed_workload(
                device, round_index, "model_only", batch_size, error))
    return tuple(rows)


def _source_sha() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[4],
                            capture_output=True, text=True, check=True)
    return result.stdout.strip()


def _hardware_model() -> str:
    if platform.system() != "Darwin":
        return platform.processor() or platform.machine() or "unavailable"
    result = subprocess.run(
        ["sysctl", "-n", "machdep.cpu.brand_string"],
        capture_output=True, text=True, check=False)
    value = result.stdout.strip()
    return value if result.returncode == 0 and value else "unavailable"


def _environment() -> dict:
    return {
        "os": platform.system(), "os_release": platform.release(),
        "architecture": platform.machine(), "processor": platform.processor() or "unavailable",
        "hardware_model": _hardware_model(),
        "python": platform.python_version(), "torch": torch.__version__,
        "numpy": np.__version__, "cpu_intraop_threads": torch.get_num_threads(),
        "cpu_interop_threads": torch.get_num_interop_threads(),
        "power_state": os.environ.get("MOTION_LAB_POWER_STATE", "not_recorded"),
        "competing_load": os.environ.get("MOTION_LAB_COMPETING_LOAD", "not_recorded"),
    }


def pin_cpu_threads() -> None:
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
    if not os.environ.get("MOTION_LAB_POWER_STATE") or not os.environ.get("MOTION_LAB_COMPETING_LOAD"):
        raise ValueError("power state and competing load must be reported before benchmarking")
    manifest = checkpoint / "manifest.json"
    if manifest.is_symlink() or not manifest.is_file():
        raise ValueError("checkpoint manifest is missing or unsafe")
    pin_cpu_threads()
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
        "schema_version": 2, "source_sha": _source_sha(),
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


def load_benchmark_evidence(input_dir: Path) -> dict:
    """Independently recompute timing summaries without running a model."""
    directory = Path(input_dir)
    try:
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError("benchmark evidence must be a real directory")
        if {path.name for path in directory.iterdir()} != {"benchmark.json", "manifest.json"}:
            raise ValueError("benchmark evidence files do not match schema")
        if any(path.is_symlink() or not path.is_file() for path in directory.iterdir()):
            raise ValueError("benchmark evidence files must be regular")
        outer = decode_object((directory / "manifest.json").read_text())
        require_equal(outer, {"schema_version": 1,
            "files": {"benchmark.json": sha256_file(directory / "benchmark.json")}},
            "benchmark evidence checksums")
        payload = decode_object((directory / "benchmark.json").read_text())
        require_fields(payload, BENCHMARK_FIELDS, "benchmark evidence")
        if type(payload["schema_version"]) is not int or payload["schema_version"] not in (1, 2):
            raise ValueError("unsupported benchmark evidence schema")
        if not isinstance(payload["source_sha"], str) or re.fullmatch(
                r"[0-9a-f]{40}", payload["source_sha"]) is None:
            raise ValueError("benchmark source SHA is invalid")
        validate_hash(payload["checkpoint_manifest_sha256"])
        require_fields(payload["validation"], {
            "dataset_version", "manifest_sha256", "train_sha256", "row_count",
        }, "benchmark validation identity")
        if (payload["validation"]["dataset_version"] != DATASET_VERSION or
                type(payload["validation"]["row_count"]) is not int or
                payload["validation"]["row_count"] != 800):
            raise ValueError("benchmark validation membership does not match protocol")
        validate_hash(payload["validation"]["manifest_sha256"])
        validate_hash(payload["validation"]["train_sha256"])
        require_equal(payload["round_device_order"], [list(row) for row in ROUND_DEVICE_ORDER],
                      "benchmark round order")
        require_equal(payload["protocol"], {
            "batch_sizes": list(BATCH_SIZES), "warmup_count": WARMUPS,
            "measured_count": MEASURED,
            "round_device_order": [list(row) for row in ROUND_DEVICE_ORDER],
            "modes": ["end_to_end", "model_only"],
        }, "benchmark protocol")
        environment = payload["environment"]
        required_environment = {
            "os", "os_release", "architecture", "processor", "python", "torch", "numpy",
            "cpu_intraop_threads", "cpu_interop_threads", "power_state", "competing_load",
        }
        if payload["schema_version"] == 2:
            required_environment.add("hardware_model")
        require_fields(environment, required_environment, "benchmark environment")
        if (payload["schema_version"] == 2 and
                (not isinstance(environment["hardware_model"], str) or
                 not environment["hardware_model"])):
            raise ValueError("benchmark hardware model must be nonempty")
        if environment["cpu_intraop_threads"] != 1 or environment["cpu_interop_threads"] != 1:
            raise ValueError("benchmark CPU threads are not pinned")
        grouped = {(round_index, device): [] for round_index, order in enumerate(ROUND_DEVICE_ORDER)
                   for device in order}
        for row in payload["results"]:
            key = (row.get("round"), row.get("device")) if isinstance(row, dict) else None
            if key not in grouped:
                raise ValueError("benchmark result has undeclared round/device")
            grouped[key].append(row)
            if row.get("status") == "failed":
                generic_fields = {"status", "device", "round", "failure_reason"}
                workload_fields = generic_fields | {"mode", "batch_size"}
                fields = workload_fields if "mode" in row or "batch_size" in row else generic_fields
                require_fields(row, fields, "failed benchmark result")
                if not isinstance(row["failure_reason"], str) or not row["failure_reason"]:
                    raise ValueError("benchmark failure reason must be nonempty")
                if fields == workload_fields and (
                        row["mode"] not in ("end_to_end", "model_only") or
                        row["batch_size"] not in BATCH_SIZES):
                    raise ValueError("failed benchmark workload does not match protocol")
                continue
            require_fields(row, {"status", "device", "round", "mode", "batch_size", "dtype",
                                 "warmup_count", "samples_s", "summary"},
                           "successful benchmark result")
            if (row["status"] != "success" or row["mode"] not in ("end_to_end", "model_only") or
                    row["batch_size"] not in BATCH_SIZES or row["dtype"] != "float32" or
                    row["warmup_count"] != WARMUPS or not isinstance(row["samples_s"], list) or
                    len(row["samples_s"]) != MEASURED):
                raise ValueError("benchmark workload does not match protocol")
            samples = tuple(row["samples_s"])
            require_equal(row["summary"], _summary(samples, row["batch_size"]),
                          "benchmark timing summary")
        expected_workloads = {(mode, batch) for mode in ("end_to_end", "model_only")
                              for batch in BATCH_SIZES}
        for rows in grouped.values():
            if len(rows) == 1 and rows[0]["status"] == "failed":
                continue
            if len(rows) != len(expected_workloads) or {
                    (row.get("mode"), row.get("batch_size")) for row in rows
                    } != expected_workloads:
                raise ValueError("benchmark round/device workload coverage is incomplete")
        return payload
    except (OSError, UnicodeError, KeyError, TypeError) as error:
        raise ValueError("cannot audit benchmark evidence") from error
