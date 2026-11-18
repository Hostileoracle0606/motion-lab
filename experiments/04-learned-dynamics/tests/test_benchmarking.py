import json
from pathlib import Path
import subprocess
import sys
import hashlib

import pytest

from learned_dynamics.benchmarking import (
    _measure_calls, benchmark_inference, load_benchmark_evidence, summarize_timings,
)


def test_summary_uses_linear_percentile_and_retains_sample_count():
    assert summarize_timings((1.0, 2.0, 3.0, 4.0)) == {
        "sample_count": 4, "median_s": 2.5, "p95_s": 3.8499999999999996,
    }
    for bad in ((), (0.0,), (True,), (float("inf"),)):
        with pytest.raises(ValueError):
            summarize_timings(bad)


def test_measurement_synchronizes_immediately_around_timed_async_work():
    events = []
    times = iter((10.0, 10.25, 20.0, 20.5))

    def operation():
        events.append("call")

    def synchronize():
        events.append("sync")

    def clock():
        events.append("clock")
        return next(times)

    samples = _measure_calls(operation, synchronize, clock, warmups=2, measured=2)
    assert samples == (0.25, 0.5)
    assert events == [
        "call", "sync", "call", "sync",
        "sync", "clock", "call", "sync", "clock",
        "sync", "clock", "call", "sync", "clock",
    ]


def test_benchmark_keeps_mps_failure_identified_and_cpu_results(
        tmp_path, synthetic_views, monkeypatch):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "manifest.json").write_text("checkpoint\n")
    output = tmp_path / "benchmark"

    def fake_device(checkpoint_dir, validation, device, round_index):
        if device == "mps":
            raise ValueError("synthetic MPS failure")
        return ({
            "status": "success", "device": device, "round": round_index,
            "mode": "end_to_end", "batch_size": 1, "dtype": "float32",
            "warmup_count": 20, "samples_s": [0.001] * 100,
            "summary": {"sample_count": 100, "median_s": 0.001, "p95_s": 0.001,
                        "batch_latency_s": 0.001, "items_per_s": 1000.0},
        },)

    monkeypatch.setattr("learned_dynamics.benchmarking._benchmark_device", fake_device)
    monkeypatch.setenv("MOTION_LAB_POWER_STATE", "AC")
    monkeypatch.setenv("MOTION_LAB_COMPETING_LOAD", "none")
    path = benchmark_inference(checkpoint, synthetic_views[1], output)
    payload = json.loads(path.read_text())
    successes = [row for row in payload["results"] if row["status"] == "success"]
    failures = [row for row in payload["results"] if row["status"] == "failed"]
    assert len(successes) == 3
    assert all(row["device"] == "cpu" for row in successes)
    assert len(failures) == 3
    assert all(row["device"] == "mps" and "MPS" in row["failure_reason"]
               for row in failures)
    assert payload["round_device_order"] == [["cpu", "mps"], ["mps", "cpu"],
                                               ["cpu", "mps"]]


def test_benchmark_refuses_unreported_power_or_competing_load(
        tmp_path, synthetic_views, monkeypatch):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "manifest.json").write_text("checkpoint\n")
    monkeypatch.delenv("MOTION_LAB_POWER_STATE", raising=False)
    monkeypatch.delenv("MOTION_LAB_COMPETING_LOAD", raising=False)
    monkeypatch.setattr("learned_dynamics.benchmarking._benchmark_device", lambda *args: ())
    with pytest.raises(ValueError, match="power state"):
        benchmark_inference(checkpoint, synthetic_views[1], tmp_path / "benchmark")


def test_device_workloads_include_both_timing_boundaries(tmp_path, monkeypatch):
    from learned_dynamics import benchmarking

    class FakePredictor:
        def predict_batch(self, states, actions, context):
            return states

    class FakeModel:
        def __call__(self, values):
            return values[:, :2]

    bundle = type("Bundle", (), {
        "model": FakeModel(), "normalizer": object(),
        "metadata": {"run_spec": {"variant_id": "canonical32", "seed": 0,
                                    "device": "mps", "repeat": 0}},
        "bundle_sha256": "a" * 64,
    })()
    monkeypatch.setattr(benchmarking, "load_checkpoint", lambda path, device: bundle)
    monkeypatch.setattr(benchmarking, "LearnedPredictor",
                        lambda model, normalizer, context, predictor_id, device: FakePredictor())
    monkeypatch.setattr(benchmarking, "transform_inputs",
                        lambda values, normalizer: values.astype("float32"))
    def fake_measure(operation, synchronize, clock, warmups, measured):
        operation()
        return (0.001,) * measured

    monkeypatch.setattr(benchmarking, "_measure_calls", fake_measure)
    from learned_dynamics.data import load_views
    root = Path(__file__).resolve().parents[3]
    validation = load_views(
        root / "experiments/02-state-transition-dataset/data/dataset-v1",
        "experiment-02-transition-v1").validation

    # This fixture proves the public benchmark surface constructs both the
    # physical CPU-array path and an already-normalized model-only path.
    rows = benchmarking._benchmark_device(
        tmp_path / "checkpoint", validation, device="cpu", round_index=0)
    assert {row["mode"] for row in rows} == {"end_to_end", "model_only"}
    assert {row["batch_size"] for row in rows} == {1, 256}

    calls = 0
    def partial_measure(operation, synchronize, clock, warmups, measured):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("synthetic model-only failure")
        operation()
        return (0.001,) * measured

    monkeypatch.setattr(benchmarking, "_measure_calls", partial_measure)
    partial = benchmarking._benchmark_device(
        tmp_path / "checkpoint", validation, device="cpu", round_index=0)
    assert partial[0]["status"] == "success"
    assert partial[0]["mode"] == "end_to_end"
    assert partial[1]["status"] == "failed"
    assert partial[1]["mode"] == "model_only"
    assert "synthetic model-only failure" in partial[1]["failure_reason"]


def test_environment_records_concrete_hardware_model(monkeypatch):
    from learned_dynamics import benchmarking

    completed = subprocess.CompletedProcess([], 0, stdout="Apple M4\n", stderr="")
    monkeypatch.setattr(benchmarking.subprocess, "run", lambda *args, **kwargs: completed)
    assert benchmarking._environment()["hardware_model"] == "Apple M4"


def test_benchmark_cli_documents_required_evidence_inputs():
    root = Path(__file__).resolve().parents[3]
    result = subprocess.run([
        sys.executable,
        str(root / "experiments/04-learned-dynamics/scripts/benchmark_devices.py"),
        "--help",
    ], cwd=root, capture_output=True, text=True)
    assert result.returncode == 0
    for option in ("--checkpoint", "--dataset", "--output", "--power-state",
                   "--competing-load"):
        assert option in result.stdout


def write_benchmark_bundle(path):
    path.mkdir()
    results = []
    orders = [["cpu", "mps"], ["mps", "cpu"], ["cpu", "mps"]]
    for round_index, order in enumerate(orders):
        for device in order:
            for mode in ("end_to_end", "model_only"):
                for batch in (1, 256):
                    samples = [0.001] * 100
                    results.append({
                        "status": "success", "device": device, "round": round_index,
                        "mode": mode, "batch_size": batch, "dtype": "float32",
                        "warmup_count": 20, "samples_s": samples,
                        "summary": {"sample_count": 100, "median_s": 0.001,
                                    "p95_s": 0.001, "batch_latency_s": 0.001,
                                    "items_per_s": batch / 0.001},
                    })
    payload = {
        "schema_version": 2, "source_sha": "a" * 40,
        "checkpoint_manifest_sha256": "b" * 64,
        "validation": {"dataset_version": "experiment-02-transition-v1",
                       "manifest_sha256": "c" * 64, "train_sha256": "d" * 64,
                       "row_count": 800},
        "protocol": {"batch_sizes": [1, 256], "warmup_count": 20,
                     "measured_count": 100, "round_device_order": orders,
                     "modes": ["end_to_end", "model_only"]},
        "round_device_order": orders,
        "environment": {"os": "Darwin", "os_release": "test", "architecture": "arm64",
                        "hardware_model": "Apple M4",
                        "processor": "test", "python": "3.12", "torch": "test",
                        "numpy": "test", "cpu_intraop_threads": 1,
                        "cpu_interop_threads": 1, "power_state": "AC",
                        "competing_load": "none"},
        "results": results,
    }
    (path / "benchmark.json").write_text(json.dumps(payload))
    digest = hashlib.sha256((path / "benchmark.json").read_bytes()).hexdigest()
    (path / "manifest.json").write_text(json.dumps(
        {"schema_version": 1, "files": {"benchmark.json": digest}}))
    return payload


@pytest.mark.parametrize("target", ["summary", "checkpoint"])
def test_benchmark_audit_rejects_semantic_tampering_after_hash_repair(tmp_path, target):
    bundle = tmp_path / "benchmark"
    payload = write_benchmark_bundle(bundle)
    assert load_benchmark_evidence(bundle)["results"]
    if target == "summary":
        payload["results"][0]["summary"]["median_s"] = 0.5
    else:
        payload["checkpoint_manifest_sha256"] = "not-a-hash"
    (bundle / "benchmark.json").write_text(json.dumps(payload))
    digest = hashlib.sha256((bundle / "benchmark.json").read_bytes()).hexdigest()
    (bundle / "manifest.json").write_text(json.dumps(
        {"schema_version": 1, "files": {"benchmark.json": digest}}))
    with pytest.raises(ValueError):
        load_benchmark_evidence(bundle)


def test_benchmark_audit_retains_partial_workload_failure(tmp_path):
    bundle = tmp_path / "benchmark"
    payload = write_benchmark_bundle(bundle)
    payload["results"][2] = {
        "status": "failed", "device": "cpu", "round": 0,
        "mode": "model_only", "batch_size": 1,
        "failure_reason": "RuntimeError: synthetic failure",
    }
    (bundle / "benchmark.json").write_text(json.dumps(payload))
    digest = hashlib.sha256((bundle / "benchmark.json").read_bytes()).hexdigest()
    (bundle / "manifest.json").write_text(json.dumps(
        {"schema_version": 1, "files": {"benchmark.json": digest}}))
    loaded = load_benchmark_evidence(bundle)
    assert loaded["results"][0]["status"] == "success"
    assert loaded["results"][2]["status"] == "failed"
