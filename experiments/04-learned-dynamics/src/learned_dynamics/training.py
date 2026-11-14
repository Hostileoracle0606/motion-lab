"""Seeded fitting, validation-only selection and fail-closed publication."""

from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import torch

from forward_dynamics.serialization import json_bytes
from learned_dynamics.checkpoint import save_checkpoint
from learned_dynamics.checkpoint_metadata import REPOSITORY_ROOT
from learned_dynamics.config import RunSpec, TrainingConfig, load_training_config
from learned_dynamics.data import load_views
from learned_dynamics.devices import require_device
from learned_dynamics.losses import OneStepBatch, one_step_loss, rollout_loss
from learned_dynamics.model import ResidualMLP
from learned_dynamics.normalization import fit_normalizer
from learned_dynamics.predictor import SUPPORTED_CONTEXT
from learned_dynamics.training_evidence import RunResult, write_run_manifest
from learned_dynamics.types import Normalizer, SplitView, TrainingView, Window
from learned_dynamics.windows import make_windows


@dataclass(frozen=True)
class TrainingMembership:
    windows: tuple[Window, ...]
    updates_per_epoch: int


@dataclass(frozen=True)
class TrainingOutcome:
    model: ResidualMLP
    normalizer: Normalizer
    curve: tuple[dict, ...]
    selected_epoch: int
    selected_validation_mse: float
    device: str
    deterministic_algorithms_requested: bool


class TrainingFailure(RuntimeError):
    """A failed fit plus every epoch that completed before the failure."""

    def __init__(self, error: Exception, curve: tuple[dict, ...]):
        super().__init__(str(error))
        self.error = error
        self.curve = curve


def training_membership(view: TrainingView | SplitView, run: RunSpec,
                        batch_size: int) -> TrainingMembership:
    if not isinstance(view, (TrainingView, SplitView)) or not isinstance(run, RunSpec):
        raise ValueError("training membership requires a checked view and RunSpec")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if run.membership == "all":
        horizon, common_history = 1, 0
    elif run.membership == "history_common":
        horizon, common_history = 1, 1
    elif run.membership == "loss_common":
        horizon, common_history = 5, 0
    else:
        raise ValueError("unsupported training membership")
    windows, _ = make_windows(view, horizon, common_history)
    if not windows:
        raise ValueError("training membership contains no eligible windows")
    return TrainingMembership(windows, math.ceil(len(windows) / batch_size))


def _batch(windows: tuple[Window, ...], run: RunSpec) -> OneStepBatch:
    inputs, deltas = [], []
    for window in windows:
        case = window.cases[0]
        row = [case.state.position_rad, case.state.velocity_rad_s, case.applied_torque_nm]
        if run.model.history_steps:
            if window.previous is None:
                raise ValueError("history training requires explicit previous state/action")
            row.extend((window.previous.state.position_rad, window.previous.state.velocity_rad_s,
                        window.previous.applied_torque_nm))
        inputs.append(row)
        deltas.append((case.expected_next_state.position_rad - case.state.position_rad,
                       case.expected_next_state.velocity_rad_s - case.state.velocity_rad_s))
    return OneStepBatch(torch.tensor(inputs, dtype=torch.float32),
                        torch.tensor(deltas, dtype=torch.float32))


def _validate_views(train: TrainingView, validation: SplitView) -> None:
    if not isinstance(train, TrainingView):
        raise ValueError("training API requires a TrainingView")
    if not isinstance(validation, SplitView) or validation.split != "validation":
        raise ValueError("selection API requires a validation SplitView; test access is forbidden")
    identity = (train.view.dataset_version, train.view.manifest_sha256, train.view.train_sha256)
    if (validation.dataset_version, validation.manifest_sha256, validation.train_sha256) != identity:
        raise ValueError("training and validation provenance disagree")


def select_earliest_epoch(values: tuple[float, ...]) -> tuple[int, float]:
    if (not isinstance(values, tuple) or not values or
            any(type(value) not in (int, float) or not math.isfinite(value) or value < 0
                for value in values)):
        raise ValueError("validation losses must be finite nonnegative values")
    best = min(values)
    return values.index(best) + 1, float(best)


def _synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def _validation_loss(model: ResidualMLP, membership: TrainingMembership,
                     run: RunSpec, normalizer: Normalizer, batch_size: int) -> float:
    total = 0.0
    with torch.no_grad():
        for start in range(0, len(membership.windows), batch_size):
            selected = membership.windows[start:start + batch_size]
            value = one_step_loss(model, _batch(selected, run), normalizer)
            total += float(value.detach().cpu()) * len(selected)
    result = total / len(membership.windows)
    if not math.isfinite(result):
        raise ValueError("validation loss is non-finite")
    return result


def fit_model(config: TrainingConfig, run: RunSpec, train: TrainingView,
              validation: SplitView) -> TrainingOutcome:
    """Request deterministic kernels, restoring the caller's process setting."""
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        return _fit_model_deterministic(config, run, train, validation)
    finally:
        torch.use_deterministic_algorithms(previous)


def _fit_model_deterministic(config: TrainingConfig, run: RunSpec, train: TrainingView,
                             validation: SplitView) -> TrainingOutcome:
    """Fit in memory; synthetic fixtures stop here and publish no checkpoint."""
    if not isinstance(config, TrainingConfig) or not isinstance(run, RunSpec):
        raise ValueError("fit_model requires TrainingConfig and RunSpec")
    _validate_views(train, validation)
    normalizer = fit_normalizer(train)
    training_set = training_membership(train, run, config.batch_size)
    validation_set = training_membership(validation, run, config.batch_size)
    target = require_device(run.device)
    model = ResidualMLP(run.model, seed=run.seed).to(target)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.optimizer.learning_rate,
        betas=config.optimizer.betas, eps=config.optimizer.epsilon,
        weight_decay=config.optimizer.weight_decay,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.sampler_seed_offset + run.seed)
    permutations = tuple(torch.randperm(len(training_set.windows), generator=generator).tolist()
                         for _ in range(config.epochs))
    curve, best_value, best_state, best_epoch, updates = [], float("inf"), None, None, 0
    try:
        for epoch, order in enumerate(permutations, start=1):
            _synchronize(target)
            started = time.perf_counter()
            total, batch_sizes = 0.0, []
            model.train()
            for start in range(0, len(order), config.batch_size):
                indices = order[start:start + config.batch_size]
                selected = tuple(training_set.windows[index] for index in indices)
                optimizer.zero_grad(set_to_none=True)
                if run.loss == "rollout_5":
                    loss = rollout_loss(model, selected, normalizer, horizon=5)
                else:
                    loss = one_step_loss(model, _batch(selected, run), normalizer)
                if not torch.isfinite(loss).item():
                    raise ValueError("training loss is non-finite")
                loss.backward()
                if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all().item()
                       for parameter in model.parameters()):
                    raise ValueError("training gradient is non-finite")
                optimizer.step()
                if any(not torch.isfinite(parameter).all().item() for parameter in model.parameters()):
                    raise ValueError("model parameters became non-finite")
                size = len(selected)
                total += float(loss.detach().cpu()) * size
                batch_sizes.append(size)
                updates += 1
            model.eval()
            validation_mse = _validation_loss(model, validation_set, run, normalizer, config.batch_size)
            _synchronize(target)
            elapsed = time.perf_counter() - started
            training_loss = total / len(training_set.windows)
            curve.append({
                "epoch": epoch, "training_loss": training_loss,
                "validation_mse": validation_mse, "batch_sizes": batch_sizes,
                "updates_this_epoch": len(batch_sizes), "optimizer_updates": updates,
                "training_seconds": elapsed,
            })
            if validation_mse < best_value:
                best_value, best_epoch = validation_mse, epoch
                best_state = {name: tensor.detach().cpu().clone()
                              for name, tensor in model.state_dict().items()}
    except (ValueError, RuntimeError, NotImplementedError, ArithmeticError,
            FloatingPointError) as error:
        raise TrainingFailure(error, tuple(curve)) from error
    if best_state is None or best_epoch is None:
        raise ValueError("training selected no finite validation checkpoint")
    declared_epoch, declared_value = select_earliest_epoch(
        tuple(row["validation_mse"] for row in curve))
    if best_epoch != declared_epoch or best_value != declared_value:
        raise ValueError("checkpoint selection disagrees with the earliest-minimum policy")
    selected = ResidualMLP(run.model, seed=run.seed)
    selected.load_state_dict(best_state, strict=True)
    return TrainingOutcome(selected.eval(), normalizer, tuple(curve), best_epoch,
                           float(best_value), run.device, True)


def _source_state() -> tuple[str, bool]:
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT,
                                     text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"],
                                         cwd=REPOSITORY_ROOT, text=True).strip())
    return source, dirty


def _curve_payload(config: TrainingConfig, run: RunSpec, curve: tuple[dict, ...],
                   selected_epoch: int | None,
                   selected_validation_mse: float | None) -> dict:
    return {
        "schema_version": 1, "run_id": run.run_id, "run_spec": run.mapping(),
        "training_config_sha256": config.config_sha256,
        "selected_epoch": selected_epoch,
        "selected_validation_mse": selected_validation_mse,
        "deterministic_algorithms_requested": True,
        "epochs": list(curve),
    }


def _failure(run_dir: Path, config: TrainingConfig, run: RunSpec, error: Exception,
             curve: tuple[dict, ...] = ()) -> RunResult:
    reason = f"{type(error).__name__}: {error}"
    (run_dir / "learning-curve.json").write_bytes(json_bytes(
        _curve_payload(config, run, curve, None, None), indent=2))
    (run_dir / "failure.json").write_bytes(json_bytes({
        "schema_version": 1, "run_id": run.run_id, "status": "failed",
        "failure_reason": reason, "completed_epochs": len(curve),
    }, indent=2))
    return RunResult(run.run_id, "failed", None, None, None, None, reason)


def train_run(config: TrainingConfig, run: RunSpec, train: TrainingView,
              validation: SplitView, output_dir: Path) -> RunResult:
    """Train one declared run from scratch and retain success or failure evidence."""
    return _train_run(config, run, train, validation, output_dir, _source_state())


def _train_run(config: TrainingConfig, run: RunSpec, train: TrainingView,
               validation: SplitView, output_dir: Path,
               source_state: tuple[str, bool]) -> RunResult:
    _validate_views(train, validation)
    if run.run_id not in {planned.run_id for planned in config.runs}:
        raise ValueError("run is not declared by the training config")
    output = Path(output_dir).resolve()
    if output != config.output_dir or output.is_symlink():
        raise ValueError("run output must equal the configured real output")
    output.mkdir(parents=True, exist_ok=True)
    run_dir = output / run.run_id
    try:
        run_dir.mkdir()
    except FileExistsError as error:
        raise ValueError("run output already exists; restart requires a fresh directory") from error
    try:
        outcome = fit_model(config, run, train, validation)
        (run_dir / "learning-curve.json").write_bytes(json_bytes(
            _curve_payload(config, run, outcome.curve, outcome.selected_epoch,
                           outcome.selected_validation_mse), indent=2))
        source, dirty = source_state
        metadata = {
            "schema_version": 1, "model_spec": asdict(run.model),
            "run_spec": run.mapping(), "context": asdict(SUPPORTED_CONTEXT),
            "selected_epoch": outcome.selected_epoch,
            "selection_metric": config.selection_metric,
            "selected_validation_mse": outcome.selected_validation_mse,
            "dataset_version": outcome.normalizer.dataset_version,
            "dataset_manifest_sha256": outcome.normalizer.manifest_sha256,
            "train_sha256": outcome.normalizer.train_sha256,
            "training_config_sha256": config.config_sha256,
            "source_sha": source, "working_tree_dirty": dirty,
            "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                        "numpy": np.__version__, "os": platform.system(),
                        "architecture": platform.machine(), "device": run.device},
        }
        checkpoint = save_checkpoint(run_dir / "checkpoint", outcome.model,
                                     outcome.normalizer, metadata)
        from learned_dynamics.checkpoint import load_checkpoint
        bundle = load_checkpoint(checkpoint)
        return RunResult(run.run_id, "success", checkpoint, bundle.bundle_sha256,
                         outcome.selected_epoch, outcome.selected_validation_mse, None)
    except TrainingFailure as failure:
        return _failure(run_dir, config, run, failure.error, failure.curve)
    except (ValueError, RuntimeError, NotImplementedError, ArithmeticError,
            FloatingPointError) as error:
        return _failure(run_dir, config, run, error)


def train(config_path: Path, output_dir: Path, *, allow_full_study: bool = False) -> Path:
    """Run a configured roster. The 45-run study requires Task 9's explicit gate."""
    config = load_training_config(config_path)
    output = Path(output_dir).resolve()
    if output != config.output_dir:
        raise ValueError("--output must match the file-relative configured output")
    if output.exists() or output.is_symlink():
        raise ValueError("training output already exists; restart from a fresh path")
    if len(config.runs) == 45 and not allow_full_study:
        raise ValueError("the full 45-run study is gated to Task 9; pass the explicit study gate")
    source, dirty = _source_state()
    views = load_views(config.dataset_dir, config.dataset_version)
    normalizer = fit_normalizer(views.train)
    normalizer_hash = hashlib.sha256(json_bytes(asdict(normalizer), indent=2)).hexdigest()
    output.mkdir(parents=True)
    results = tuple(_train_run(config, run, views.train, views.validation, output,
                               (source, dirty))
                    for run in config.runs)
    provenance = {
        "source_sha": source, "working_tree_dirty": dirty,
        "dataset_manifest_sha256": normalizer.manifest_sha256,
        "train_sha256": normalizer.train_sha256,
        "normalizer_sha256": normalizer_hash,
        "environment_sha256": SUPPORTED_CONTEXT.environment_version,
        "actual_command": list(sys.argv),
    }
    return write_run_manifest(output, config, results, provenance)
