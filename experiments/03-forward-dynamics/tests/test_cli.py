from pathlib import Path
import subprocess
import sys
import shutil

import pytest

from forward_dynamics.config import DEFAULT_CONFIG
from forward_dynamics.evidence import load_evaluation_evidence
from forward_dynamics.generation import run_evaluation
import forward_dynamics.provenance as provenance
import forward_dynamics.generation as generation


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "experiments/03-forward-dynamics/scripts/run_evaluation.py"


def test_dirty_tree_refusal_happens_before_output_creation(monkeypatch, tmp_path) -> None:
    original = provenance.git_output
    monkeypatch.setattr(provenance, "git_output", lambda *args: " M dirty" if args[0] == "status" else original(*args))
    output = tmp_path / "refused"
    with pytest.raises(ValueError, match="uncommitted"):
        run_evaluation(DEFAULT_CONFIG, output)
    assert not output.exists()


def test_explicit_dirty_run_is_labeled(monkeypatch, tmp_path) -> None:
    original = provenance.git_output
    monkeypatch.setattr(provenance, "git_output", lambda *args: " M dirty" if args[0] == "status" else original(*args))
    output = tmp_path / "allowed"
    run_evaluation(DEFAULT_CONFIG, output, allow_dirty=True)
    evidence = load_evaluation_evidence(output)
    assert evidence.evaluation["provenance"]["working_tree_dirty"] is True
    assert evidence.evaluation["provenance"]["allow_dirty"] is True


def test_existing_output_is_preserved(tmp_path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "important.txt"
    marker.write_text("preserve me")
    with pytest.raises(ValueError, match="exist"):
        run_evaluation(DEFAULT_CONFIG, output, allow_dirty=True)
    assert marker.read_text() == "preserve me"
    assert tuple(output.iterdir()) == (marker,)


@pytest.mark.parametrize("relative", [".", "experiments/02-state-transition-dataset/data/dataset-v1/new-output",
                                        "experiments/03-forward-dynamics/configs/new-output"])
def test_refuses_input_output_tree_overlap(relative) -> None:
    with pytest.raises(ValueError, match="overlap"):
        run_evaluation(DEFAULT_CONFIG, ROOT / relative, allow_dirty=True)


def test_cli_is_independent_of_working_directory_and_output_path(tmp_path) -> None:
    outputs = (tmp_path / "first", tmp_path / "second")
    for output in outputs:
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--config", str(DEFAULT_CONFIG), "--output", str(output), "--allow-dirty"],
            cwd=tmp_path, capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
        assert (output / "manifest.json").exists()
    for name in ("manifest.json", "evaluation.json", "one_step.jsonl"):
        assert (outputs[0] / name).read_bytes() == (outputs[1] / name).read_bytes()


def test_cli_help_uses_ordinary_package_imports(tmp_path) -> None:
    result = subprocess.run([sys.executable, str(SCRIPT), "--help"], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0
    assert "--allow-dirty" in result.stdout


def test_partition_checksum_failure_creates_no_output(tmp_path) -> None:
    dataset = tmp_path / "copied-dataset"
    shutil.copytree(ROOT / "experiments/02-state-transition-dataset/data/dataset-v1", dataset)
    partition = dataset / "train.jsonl"
    partition.write_bytes(partition.read_bytes() + b"\n")
    protocol_dir = tmp_path / "protocol"
    protocol_dir.mkdir()
    config = protocol_dir / "invalid.toml"
    text = DEFAULT_CONFIG.read_text().replace(
        '"../../02-state-transition-dataset/data/dataset-v1"', f'"{dataset}"',
    ).replace('"../../01-feedback-control/models/one_link.xml"',
              f'"{ROOT / "experiments/01-feedback-control/models/one_link.xml"}"')
    config.write_text(text)
    output = tmp_path / "not-written"
    with pytest.raises(ValueError, match="checksum"):
        run_evaluation(config, output, allow_dirty=True)
    assert not output.exists()


def test_staging_validation_failure_never_publishes_partial_evidence(monkeypatch, tmp_path) -> None:
    def rejected(stage):
        assert (stage / "manifest.json").is_file()
        raise ValueError("injected validation failure")

    monkeypatch.setattr(generation, "load_evaluation_evidence", rejected)
    output = tmp_path / "not-published"
    with pytest.raises(ValueError, match="injected validation"):
        run_evaluation(DEFAULT_CONFIG, output, allow_dirty=True)
    assert not output.exists()
    assert not tuple(tmp_path.glob(".one-step-staging-*"))


@pytest.mark.parametrize("experiment", ["01-feedback-control", "02-state-transition-dataset"])
def test_output_cannot_be_nested_inside_any_participating_source_tree(monkeypatch, experiment) -> None:
    def reached_evaluation(*args, **kwargs):
        raise AssertionError("source overlap must fail before evaluation begins")

    monkeypatch.setattr(generation, "capture_provenance", reached_evaluation)
    output = ROOT / "experiments" / experiment / "src" / "invalid-evidence-output"
    with pytest.raises(ValueError, match="overlap"):
        run_evaluation(DEFAULT_CONFIG, output, allow_dirty=True)
    assert not output.exists()
