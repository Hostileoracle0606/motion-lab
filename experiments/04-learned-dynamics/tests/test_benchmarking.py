import json
from pathlib import Path
import subprocess
import sys

import pytest

from learned_dynamics.benchmarking import (
    _measure_calls, benchmark_inference, summarize_timings,
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
