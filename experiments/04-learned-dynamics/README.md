# Experiment 04 — Learned dynamics

Status: data/preprocessing APIs (#25) and the residual predictor (#26) are implemented. Model training,
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
PyTorch 2.15.0 with explicit CPU/MPS selection. The small, simulated one-link dataset has one collection seed and only
four scenarios; it supports no physical fidelity or broad-generalization claim.

## Residual model and explicit devices

`ResidualMLP(ModelSpec(), seed=0)` creates the declared float32
3->32->32->2 tanh network on CPU. Width 16 and one-step explicit history
(six inputs) are the other declared v1 options. Initialization restores the
caller's CPU RNG state. `LearnedPredictor(model, normalizer, context, id,
device)` owns an inference copy, applies saved scaling, predicts a normalized
delta and restores it to rad/rad/s before adding the current state.
A zero **normalized** output means the saved mean physical delta; zero
physical delta corresponds to subtracting the saved mean and dividing by scale.

Use `device="mps"` explicitly to request the GPU or `device="cpu"` for
the CPU. Missing/unsupported MPS raises an error and never substitutes CPU.
Launch Python without `PYTORCH_ENABLE_MPS_FALLBACK=1`; changing that variable
after importing PyTorch does not reconfigure the backend. GPU timing remains
a later benchmark task.

The canonical `predict` and `predict_batch` receive current state, applied
action and the exact one-link environment/context. History uses
`predict_with_history` with an explicit previous state/action and never
stores hidden trajectory state. Batched physical arrays are [N,2] states and
[N] actions; model tensors are float32. Outputs are finite physical values
without clipping. No simulator calls, fitting, targets or expected future
states enter inference.

Exercise ordinary installed imports, train-only preprocessing and an untrained
seeded network from any working directory:

```bash
uv run python experiments/04-learned-dynamics/scripts/smoke_predictor.py --device cpu
uv run python experiments/04-learned-dynamics/scripts/smoke_predictor.py --device mps
```

These commands print structured synthetic operation results, not model-quality
measurements. Setup and CLI tests on macOS 26.4 arm64 / Python 3.12.15 confirmed
PyTorch 2.15.0 MPS availability, prediction, and a separate forward/backward/Adam
step. CPU/MPS speed, learned accuracy and training repeatability are later
experiments and are not established by this smoke check.

## Inference checkpoint bundles

`save_checkpoint(output_dir, model, normalizer, metadata)` creates a fresh
directory with metadata.json, normalizer.json, weights.npz and manifest.json.
Supply metadata including model/run/context, selected epoch and validation
metric, dataset/config/source identities and runtime; the saver derives
normalizer/tensor hashes and the tensor filename. `load_checkpoint(input_dir,
device="cpu")` returns a checked model, the complete normalizer (including
all fitted episode/step keys), metadata and bundle digest. A checkpoint can
be replayed on CPU or explicitly requested MPS.

The loader checks fixed names, regular files, exact schemas, reachable source
commit, environment/timing/action context, architecture/run consistency,
finite float32 weights and complete tensor names/shapes. Before creating a
model it checks NPZ headers and resource limits (2 MiB per JSON file, 16 MiB
compressed/decompressed tensor budget); it loads no Python pickle and
extracts no arbitrary archive paths. It also validates the original committed
v1 dataset and independently refits its training normalizer to bind statistics,
hashes and fitted identities. The original dataset must remain available.

The source/config/seed metadata are declared provenance. File hashes and
source-commit existence alone do not prove which source/config actually
trained weights; the training-selection/evidence audit and reproduction
tasks must verify those claims. Changing all coherent declarations cannot
be detected as forgery by a self-contained unsigned bundle.

Saves are staged and audited before exclusively reserving the final directory.
An interruption while copying can leave an incomplete output, which loading
rejects; retry into a new directory after inspecting the interrupted one.
Source/input locations and existing outputs cannot be overwritten. Checkpoints
are inference-only; optimizer state and resume are unsupported in schema v1.
