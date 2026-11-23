"""Audit matched validation results once both 45-run rosters are present."""

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch

from learned_dynamics.checkpoint import load_checkpoint
from learned_dynamics.config import load_training_config
from learned_dynamics.data import load_views
from learned_dynamics.predictor import LearnedPredictor, SUPPORTED_CONTEXT
from learned_dynamics.training import training_membership
from learned_dynamics.training_evidence import load_run_manifest


TOLERANCE = np.array([1e-4, 1e-3])


def validation_mae(directory, row, run, windows):
    checkpoint = load_checkpoint(directory / row["checkpoint_dir"], device=run.device)
    if checkpoint.bundle_sha256 != row["checkpoint_sha256"]:
        raise ValueError(f"{run.run_id} checkpoint hash differs from roster")
    model = LearnedPredictor(checkpoint.model, checkpoint.normalizer, SUPPORTED_CONTEXT,
                             run.run_id, run.device)
    predicted = []
    expected = []
    for window in windows:
        case = window.cases[0]
        if run.model.history_steps:
            previous = window.previous
            state = model.predict_with_history(
                case.state, case.applied_torque_nm, previous.state,
                previous.applied_torque_nm, SUPPORTED_CONTEXT,
            )
        else:
            state = model.predict(case.state, case.applied_torque_nm, SUPPORTED_CONTEXT)
        predicted.append((state.position_rad, state.velocity_rad_s))
        expected.append((case.expected_next_state.position_rad, case.expected_next_state.velocity_rad_s))
    return np.abs(np.asarray(predicted) - np.asarray(expected)).mean(axis=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", required=True, type=Path)
    parser.add_argument("--retrained", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    config = load_training_config(args.config)
    rosters = [load_run_manifest(path) for path in (args.original, args.retrained)]
    expected = [run.run_id for run in config.runs]
    if len(expected) != 45 or any(bundle["training_config_sha256"] != config.config_sha256
                                 for bundle in rosters):
        raise ValueError("both rosters must bind the original 45-entry config")
    identities = ["dataset_manifest_sha256", "train_sha256", "normalizer_sha256", "environment_sha256"]
    compatible = all(rosters[0]["provenance"][key] == rosters[1]["provenance"][key]
                     for key in identities)
    if not compatible:
        raise ValueError("training dataset, normalization or environment identity differs")
    indexed = [{row["run_id"]: row for row in roster["results"]} for roster in rosters]
    if any(set(index) != set(expected) for index in indexed):
        raise ValueError("a roster is missing declared runs")
    view = load_views(config.dataset_dir, config.dataset_version).validation
    rows = []
    for run in config.runs:
        old, new = (index[run.run_id] for index in indexed)
        row = {"run_id": run.run_id,
               "original_status": old["status"], "retrained_status": new["status"],
               "original_failure": old["failure_reason"], "retrained_failure": new["failure_reason"],
               "original_epoch": old["selected_epoch"], "retrained_epoch": new["selected_epoch"]}
        if old["status"] == new["status"] == "success":
            windows = training_membership(view, run, config.batch_size).windows
            original_mae = validation_mae(args.original, old, run, windows)
            retrained_mae = validation_mae(args.retrained, new, run, windows)
            drift = np.abs(retrained_mae - original_mae)
            row.update({
                "windows": len(windows), "original_mae": original_mae.tolist(),
                "retrained_mae": retrained_mae.tolist(), "absolute_delta_mae": drift.tolist(),
                "pass": bool(np.all(drift <= TOLERANCE)),
            })
        else:
            row["pass"] = False
        rows.append(row)
        gc.collect()
    print(json.dumps({"thresholds_rad_rad_s": TOLERANCE.tolist(), "runs": rows,
                      "pass_count": sum(row["pass"] for row in rows)}, indent=2, allow_nan=False))
    return 0 if all(row["pass"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
