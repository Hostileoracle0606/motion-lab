# Experiment 04 — Learned dynamics

Status: data/preprocessing APIs implemented (#25). Model training,
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

## Train-only normalization

`fit_normalizer(views.train)` fits per-feature means and population standard
deviations in float64 on the original 940 train transitions. A standard
deviation below 1e-12 uses scale 1.0 and a saved constant-feature flag.
`transform_inputs` supports canonical [N,3] inputs and explicit
current/previous [N,6] inputs using the same three fitted feature scales.
`inverse_delta` restores [N,2] model outputs to delta rad and rad/s.
All functions reject invalid shapes/dtypes, booleans and non-finite values;
they do not clip unseen inputs. The future predictor owns float32 conversion
and adding the restored delta to the current physical state.

`write_normalizer(normalizer, path)` emits deterministic schema-v1 JSON and
refuses an existing destination. `load_normalizer(path)` checks schema,
feature order/units, finite positive scales, constant flags, fitted row count,
ordered episode/step keys, dataset hashes and declared target/statistics
transforms. These fields link to source data; loading the JSON alone does not
prove its statistics were fitted honestly. A future checkpoint/evidence audit
must bind it to the original checked dataset.

## Boundary-aware windows

`make_windows(view, horizon, history_steps=0)` returns immutable windows and
per-episode coverage records. A terminal transition may end a window but
cannot precede another transition. Inputs are checked for contiguous steps,
state continuity and fixed episode/context identity. Supported history is
zero or one preceding state/action; the previous row stays separate from the
future labels. Recursive inference is owned by later predictor/evaluator work.

Every supplied start belongs to exactly one category: eligible, unavailable
future, or missing history. Unavailable future takes priority when both are
missing. Training membership counts are 940 (horizon 1), 936 (horizon 1 plus
history) and 924 (horizon 5). Validation/test counts are 800/798/792 each.
History and loss studies must use the same eligible starts in both arms.

For example, from the repository root:

```python
from pathlib import Path
from learned_dynamics.data import load_views
from learned_dynamics.normalization import fit_normalizer
from learned_dynamics.windows import make_windows

views = load_views(
    Path("experiments/02-state-transition-dataset/data/dataset-v1"),
    "experiment-02-transition-v1",
)
normalizer = fit_normalizer(views.train)
windows, coverage = make_windows(views.train, horizon=5)
assert normalizer.fit_row_count == 940
assert len(windows) == 924
```

Check yourself: why would fitting scales on all three partitions cause
leakage? The test distribution would influence preprocessing before final
evaluation, even if the neural network never received a test row.

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
