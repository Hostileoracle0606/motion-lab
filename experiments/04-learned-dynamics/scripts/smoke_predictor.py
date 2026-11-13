"""Exercise a seeded, untrained predictor on the explicitly selected device."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform

import torch

from forward_dynamics.types import DynamicsState
from learned_dynamics.data import load_views
from learned_dynamics.model import ResidualMLP
from learned_dynamics.normalization import fit_normalizer
from learned_dynamics.predictor import LearnedPredictor, SUPPORTED_CONTEXT
from learned_dynamics.types import ModelSpec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "mps"), required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    try:
        views = load_views(root / "experiments/02-state-transition-dataset/data/dataset-v1",
                           "experiment-02-transition-v1")
        normalizer = fit_normalizer(views.train)
        predictor = LearnedPredictor(ResidualMLP(ModelSpec(), seed=0), normalizer,
                                     SUPPORTED_CONTEXT, "synthetic_seed_0", args.device)
        state = predictor.predict(DynamicsState(.3, -.2), .1, SUPPORTED_CONTEXT)
        if args.device == "mps":
            torch.mps.synchronize()
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps({
        "scope": "synthetic_operation_check", "device": args.device,
        "python": platform.python_version(), "torch": torch.__version__,
        "fit_rows": normalizer.fit_row_count,
        "predicted_state": [state.position_rad, state.velocity_rad_s],
        "context": asdict(SUPPORTED_CONTEXT),
    }, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
