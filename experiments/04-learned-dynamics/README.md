# Experiment 04 — Learned dynamics

Status: data/preprocessing implementation in progress (#25). Model training,
CPU/MPS comparisons and prediction results remain planned.

The [approved design](../../docs/superpowers/specs/2026-11-10-learned-dynamics-design.md)
and [implementation plan](../../docs/superpowers/plans/2026-11-11-learned-dynamics.md)
define the experiment. Input is the unchanged
[Experiment 02 dataset](../02-state-transition-dataset/DATASET_CARD.md):
940 train, 800 validation and 800 test transitions, split by whole scenario.

## Data ownership

`load_views(dataset_dir, expected_dataset_version)` first invokes the existing
independent dataset validator. It retains the manifest and train-partition
SHA-256 identifiers, complete episode/step/scenario/controller/collection-seed
identity and terminal boundaries. A `TrainingView` only wraps train cases;
fitting entry points reject plain evaluation views. These Python value types
prevent accidental mixing, not a malicious caller deliberately relabeling data.
Original validated loading remains the provenance boundary.

The canonical model inputs are current position (rad), velocity (rad/s), and
applied torque (N m). Training targets are changes in position and velocity
after a 0.010 s held action. Targets, controller identity and future observations
never enter the input feature array. Terminal transitions remain present.

## Setup and verification

From the repository root, using Python 3.12 and the committed lock:

```bash
uv sync --locked
uv run python -c "import learned_dynamics, forward_dynamics, state_transition_dataset"
uv run pytest -q experiments/04-learned-dynamics/tests
```

The current supported verification environment is macOS arm64. This step adds
no learning framework. PyTorch/MPS dependency and operation verification belongs
to #26. The small, simulated one-link dataset has one collection seed and only
four scenarios; it supports no physical fidelity or broad-generalization claim.
