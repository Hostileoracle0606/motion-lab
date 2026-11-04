# Experiment 03 — Forward dynamics

## Question and current scope

How well can simple state-transition models predict the next joint position
and velocity, and how should their errors be measured?

Issue [#12](https://github.com/Hostileoracle0606/motion-lab/issues/12) implements
two deterministic baselines and pure error metrics.
[#13](https://github.com/Hostileoracle0606/motion-lab/issues/13) adds the frozen
one-step evaluation command, saved predictions, provenance and independent
evidence loading. It includes the required
[#11](https://github.com/Hostileoracle0606/motion-lab/issues/11) dataset converter
and MuJoCo reference. Issue
[#14](https://github.com/Hostileoracle0606/motion-lab/issues/14) implements
recursive recorded-action rollout evaluation as a Python API.
[#15](https://github.com/Hostileoracle0606/motion-lab/issues/15)
adds an independent analytical RK4 adapter and static plant-parameter extraction.
Both features are available through Python interfaces; evaluation registry and
CLI/evidence integration, comparison reports and Stage 3
clean-reproduction closure remain planned.

The [design](../../docs/superpowers/specs/2026-11-03-forward-dynamics-design.md)
and [ticket plan](../../docs/superpowers/plans/2026-11-03-forward-dynamics.md)
describe those later deliverables. Experiment 02's
[dataset card](../02-state-transition-dataset/DATASET_CARD.md) owns the recorded
transition and split contracts.

## Setup and verification

From the repository root:

```bash
uv sync
uv run pytest -q experiments/03-forward-dynamics/tests
uv run python -c "from forward_dynamics.baselines import NoChangePredictor, ConstantVelocityPredictor; from forward_dynamics.metrics import residual, summarize_errors"
```

Use Python `>=3.12,<3.13` and the committed `uv.lock`. Repository dependencies
include MuJoCo `3.16.0`; these baseline and metric modules use only the standard
library and do not import MuJoCo. Current verification is on macOS arm64 with
Python 3.12.15. Run the complete regression suite with `uv run pytest -q`.

## Prediction contract

`DynamicsState(position_rad, velocity_rad_s)` is a frozen scalar record.
`DynamicsContext(environment_id, environment_version, control_interval_s,
torque_limit_nm)` supplies the environment identity, action duration in seconds,
and symmetric actuator limit in newton-metres. `environment_version` represents
the model SHA-256 when using recorded data. Simple baselines require nonempty
identity strings; the simulator adapter additionally checks exact model identity.

All adapters satisfy the structural `Predictor` interface:

```python
predict(state, applied_torque_nm, context) -> DynamicsState
```

The observation comes before the action: `(state_t, applied_torque_t)` predicts
`state_{t+1}` after `control_interval_s`. The dataset's normal interval is
`0.010 s`; these equations also support other finite positive intervals.
No adapter receives target states, raw controller output, or expected next
states. Both simple baselines deliberately ignore torque after checking that
it is finite and within the context limit. They do not simulate acceleration
or damping; the MuJoCo reference applies the torque to the known plant.

Validation occurs at prediction and metric function entry, not when raw frozen
records are constructed. `validate_prediction_inputs` is the shared adapter
validator. Position, velocity, and torque must be finite real scalars;
booleans, strings, arrays, and non-finite values raise `ValueError`. Interval
and torque limit must also be strictly positive. Applied actions outside the
limit raise `ValueError` rather than being silently clipped. Non-finite
computed predictions and residuals are rejected too.

Accepted real scalars are converted to ordinary binary64 Python floats before
arithmetic, including NumPy integers and single-precision floats. This avoids
fixed-width overflow in torque limits, subtraction, and error summaries.
Large integers may lose precision in that conversion; arbitrary-precision
integer or extended-precision dynamics are outside this float-valued contract.

| Adapter ID | Predicted position | Predicted velocity |
| --- | --- | --- |
| `no_change` | `q` | `v` |
| `constant_velocity` | `q + v * dt` | `v` |

The `mujoco` adapter supports only the checksummed self-contained Experiment 01
XML (`8612091d73babb1a0122cb5163ea744183486cd8cc6fbae2813f45818be57263`) on
MuJoCo 3.16.0. Each call constructs fresh simulator data, initializes the
supplied position/velocity, and applies torque for integer `0.002 s` substeps.
Calls do not retain previous hidden simulator state. A query is limited to
10,000 substeps (20 s); the recorded data uses five. New models, actuator state,
contacts or history-dependent dynamics require an expanded state/model
contract even when position and velocity dimensions happen to match.

For `q = 1 rad`, `v = 2 rad/s`, and `dt = 0.01 s`, these return `(1, 2)` and
`(1.02, 2)` respectively. Position is never wrapped or clamped: an excursion
beyond the nominal joint range must remain visible to evaluation.

## Independent analytical dynamics

`plant.py` owns the frozen `PlantParameters(environment_id, environment_version,
inertia_kg_m2, damping_nm_s_per_rad)` record. `load_plant_parameters(model_path,
expected_environment_version)` verifies SHA-256 before compiling the
self-contained model with MuJoCo 3.16.0. For the authoritative Experiment 01
XML above it extracts `J = 0.23242512112403102 kg m^2` and
`b = 0.05 N m s/rad`; `J` includes `0.01 kg m^2` of joint armature.
These are known model coefficients, with no fitting to transitions.

For unit hinge axis `a`, body-frame COM offset from the joint `r`, principal
inertias `I`, body mass `m`, and principal-to-body rotation `R`, extraction uses:

```text
J = a^T R diag(I) R^T a + m * |r cross a|^2 + armature
```

Thus the principal inertia frame and parallel-axis shift both matter. Static
kinematics and composite rigid-body inertia independently cross-check this
calculation against the compiled generalized mass matrix at five angles,
including outside the joint limits. Extraction supports one rigid moving body,
one hinge and one unit-gain direct torque motor, zero gravity, no fluid forces,
spring, friction loss, tendons, equalities, plugins or actuator activation.
Attached bodies, external model assets and altered simulator flags are refused.
This is a narrow damped-rotor extraction contract, not a general MJCF loader.

`analytical.py` owns `AnalyticalRK4Predictor(parameters, integration_timestep_s)`:

```text
dq/dt = v
dv/dt = (u - b*v) / J
```

It uses fixed-step classical fourth-order Runge–Kutta with the applied torque
held constant throughout the control interval. The frozen experiment steps are
`0.002`, `0.001` and `0.0005 s`, yielding IDs `analytical_rk4_0.002`,
`analytical_rk4_0.001` and `analytical_rk4_0.0005`. An interval must contain a
positive integer number of integration steps (at most 10,000); the predictor
checks the context's identity against the parameter record. Inertia is finite
and positive, damping finite and nonnegative. Arithmetic uses ordinary Python
floats, and non-finite results fail evaluation.

Neither importing the parameter record nor importing/running the analytical
predictor imports MuJoCo. Only explicit parameter extraction needs the simulator;
prediction queries no simulator forces and retains no state between calls.

Tests isolate numerical integration error with the exact constant-torque rotor:
for `b > 0`, let `v_inf = u/b` and `d = exp(-b*t/J)`. Then
`v(t) = v_inf + (v0-v_inf)*d` and
`q(t) = q0 + v_inf*t + (v0-v_inf)*(J/b)*(1-d)`.
For `b = 0`, `v(t) = v0 + (u/J)*t` and
`q(t) = q0 + v0*t + 0.5*(u/J)*t^2`.
Tests cover both `0.01 s` and `1 s` and check decreasing integration error
above floating-point noise; they do not tune steps using dataset errors.

The ODE omits MuJoCo soft joint-limit and contact forces. It never wraps or
clamps position, drops boundary failures, or changes the common evaluation
membership. A finite prediction beyond a limit is retained. More accurate
integration of these simplified equations cannot restore missing forces:
exact-ODE numerical error and disagreement with MuJoCo are different quantities.

## Metric definitions and ownership

`residual(predicted, expected)` returns a frozen `StateResidual` with direct
signed differences:

```text
position_error_rad = predicted.position_rad - expected.position_rad
velocity_error_rad_s = predicted.velocity_rad_s - expected.velocity_rad_s
```

This is a limited hinge, so position error is not wrapped modulo `2*pi`.
Predicted `+3 rad` versus expected `-3 rad` has error `+6 rad`. Continuous
rotational joints will require a different, explicitly documented geometry
contract.

`summarize_errors(errors: tuple[StateResidual, ...])` requires at least one
finite residual and returns `ErrorSummary` with `sample_count`, MAE, RMSE, and
maximum absolute error for each coordinate separately. Position metrics use
radians; velocity metrics use radians per second. There is no combined score
that mixes those units.

For coordinate errors `e_1, ..., e_N`:

```text
MAE = sum(abs(e_i)) / N
RMSE = sqrt(sum(e_i**2) / N)
max_abs = max(abs(e_i))
```

The implementation scales errors before summing/squaring to avoid overflow
or underflow for otherwise representable summaries. For errors `(-1, 3)`, MAE
is `2`, RMSE is `sqrt(5)`, and maximum absolute error is `3`.

Every supplied transition receives equal weight; no sample is filtered out.
Concatenating episodes therefore weights longer episodes more heavily. The
evaluator also calls these functions per episode and retains those
results, scenario/controller/split membership, counts, and terminal/failure
boundaries. These pure metric functions do not choose groups, select episodes,
perform file I/O, or evaluate
the held-out dataset.

## Recursive open-loop evaluation

`rollouts.evaluate_rollouts(cases, predictor, horizons)` starts at every eligible
recorded state and feeds each predicted state into the next call. Each call
uses that step's recorded `applied_torque_nm`; it never recomputes a controller
action or replaces an intermediate prediction with an observation. This tests
conditional prediction under a fixed action sequence. It does not measure how
a controller would respond to a different predicted trajectory.

The frozen protocol's horizons `(1, 5, 10, 25, 50, 100)` correspond to
`(0.01, 0.05, 0.10, 0.25, 0.50, 1.00) s` at the dataset's `0.010 s` interval.
The API also accepts other increasing, unique positive integer horizons;
booleans and empty horizon tuples are rejected. For example:

```python
from pathlib import Path
from forward_dynamics.baselines import ConstantVelocityPredictor
from forward_dynamics.dataset import load_prediction_cases
from forward_dynamics.rollouts import evaluate_rollouts

cases = load_prediction_cases(
    Path("experiments/02-state-transition-dataset/data/dataset-v1"),
    "experiment-02-transition-v1",
)
evaluation = evaluate_rollouts(
    cases, ConstantVelocityPredictor(), (1, 5, 10, 25, 50, 100),
)
```

`RolloutEvaluation.predictions` retains `RolloutPrediction` records with
split/episode/scenario/controller/seed identity, `start_step_index`,
`horizon_steps`, `horizon_s`, `predictor_id`, `predicted_endpoint_state`,
`expected_endpoint_state`, and the shared physical-unit `residual`. A window
starting at step `s` with horizon `h` applies the torques at steps `s` through
`s+h-1` and compares against `expected_next_state` at step `s+h-1`. Horizon-one
predictions match the corresponding observed-state one-step calls.

`RolloutEvaluation.windows` independently retains a `WindowCount` for every
episode/horizon, including horizons without predictions. For `N` supplied
transitions, `eligible_count = max(N-h+1, 0)` and
`unavailable_count = N-eligible_count`. All predictors receive the same eligible
windows. A three-transition episode at horizons `(1, 2, 5)` therefore has
eligible counts `(3, 2, 0)` and unavailable counts `(0, 1, 3)`. The dataset's
two failed 70-transition episodes remain visible at horizon 100 with zero
eligible and 70 unavailable starts each.

Cases within each episode must have contiguous unique indices, consistent
identity/context and state continuity. Continuity uses the dataset's absolute
tolerance `1e-12` separately in radians and radians per second, with zero
relative tolerance. No window crosses an episode, split, or terminal boundary;
the final terminal/truncated transition may be its endpoint. Contiguous slices
may start above step zero and end before a boundary, but their counts describe
only those supplied rows. Full-coverage reporting must pass complete episodes
from `load_prediction_cases`. Records are ordered by dataset split order,
episode ID, increasing horizon, then start step.

Finite predictions beyond nominal joint limits are retained without wrapping
or clipping. Every intermediate prediction is validated and its real scalars
are normalized to Python floats before reuse. Invalid inputs, non-finite
predictions or residuals fail explicitly with predictor, episode and step
context; difficult windows are never silently removed. Overlapping windows
share transitions and are correlated, so their counts are not independent
trials. This API does not publish a rollout evidence bundle or Stage 3 result.

## Run and audit a one-step evaluation

Commit source/config changes before a reproducible run. From the repository
root, choose a fresh output directory:

```bash
uv run python experiments/03-forward-dynamics/scripts/run_evaluation.py \
  --config experiments/03-forward-dynamics/configs/evaluation-v1.toml \
  --output artifacts/experiment-03-one-step-v1
```

`--config` and `--output` may be absolute paths when running from another
directory. Configured dataset/model paths resolve relative to the TOML file.
The config, inputs and evaluator sources must reside in this checkout.
`--allow-dirty` permits exploratory runs with uncommitted changes, labels them,
and records that permission; it is not a clean-reproduction claim.
Input bytes absent from or different from the recorded commit count as dirty
even when Git status omits an ignored config or an assume-unchanged file.

The versioned protocol declares `scope = "one_step"`, dataset identity,
environment checksum, adapters, tolerance `1e-12`, future rollout horizons
`[1, 5, 10, 25, 50, 100]`, and future analytical timesteps
`[0.002, 0.001, 0.0005] s`. The future settings are recorded but unused in this
scope. No model is fit, no normalization is learned and no settings are chosen
from test results.

The validated dataset contains 940 train, 800 validation and 800 test rows.
All 2,540 transitions, including terminal rows, are evaluated by all three
adapters, producing 7,620 records. Each call starts from the observed state,
so model error is corrected before the next recorded transition. This differs
from recursive rollout evaluation, which feeds a prediction into the next
prediction. Paired controller episodes and whole-scenario splits are preserved.

The output directory contains:

- `one_step.jsonl`: identity, current state, applied action/context, expected
  and predicted next state, residual and terminal boundary for every prediction.
- `evaluation.json`: per-episode/scenario/controller/split summaries, counts,
  adapter parameters, frozen protocol and source/runtime provenance.
- `manifest.json`: schema/scope and SHA-256 of the two saved record files.

Audit the saved data independently:

```bash
uv run python -c "from pathlib import Path; from forward_dynamics.evidence import load_evaluation_evidence; e = load_evaluation_evidence(Path('artifacts/experiment-03-one-step-v1')); print(e.evaluation['prediction_count'])"
```

Loading verifies input/output checksums, schema, exact membership and expected
states, residuals, adapter settings, all counts and recomputed aggregates.
It checks MuJoCo predictions against recorded next states at the declared
tolerance and simple predictions against their equations. It never queries
simulator dynamics. Bad records fail rather than silently disappearing.

Provenance distinguishes the dataset source commit (Experiment 01), dataset
builder commit (Experiment 02), evaluator commit, and any later publication
commit. Input hashes include the config, model, dataset manifest/partitions,
`uv.lock`, packaging and participating experiment sources. Clean provenance
requires those bytes to match the recorded Git commit. Runtime records Python,
MuJoCo, OS and architecture. Absolute checkout/output paths and timestamps are
excluded; the recorded command uses `<output-dir>` as a substitution point.
Two fresh output locations at the same clean source commit/environment should
therefore have identical bytes. Reloading requires the recorded source/input
checkout; this is not a standalone or cryptographically authenticated archive.

Existing output directories are always refused. Inputs and outputs must not
overlap. New files are staged and independently audited before a directory
rename exposes the bundle. If a process is interrupted, preserve any previous
output, inspect an abandoned `.one-step-staging-*` directory if present, and
rerun into a fresh directory; incomplete bundles fail loading. This workflow
does not promise durability across filesystem/power failures or concurrent
writers. No published Stage 3 report or physical-system validation is implied.

## File ownership

`dataset.py` preserves the independently validated Experiment 02 contract.
`mujoco_predictor.py` owns fresh simulator queries. `config.py` owns the frozen
protocol; `evaluation.py` owns observed-state calls and grouped summaries.
`rollouts.py` owns recursive recorded-action windows, endpoint predictions and
eligible/unavailable counts independently of publication.
`provenance.py` captures/checks source identity, `serialization.py` handles
strict finite JSON, and `evidence.py` audits saved derivations.
`generation.py` owns staging/publication and the CLI owns user-facing errors.
`plant.py` extracts static physical coefficients; `analytical.py` integrates
the simplified unconstrained ODE independently of simulator dynamics.

## Learning checkpoint

At `dt = 0.01 s`, an acceleration error of `10 rad/s^2` produces a velocity
error of `0.1 rad/s` but a position error of only `0.0005 rad` under constant
acceleration. Small one-step position error can therefore coexist with a
meaningful velocity error, which will affect later position predictions.

Why would wrapping a predicted joint-limit excursion make a baseline look
better? It could replace a large direct error with a smaller equivalent-angle
error even though this limited joint cannot rotate through that path. Retain
the excursion and measure each coordinate in its own units.

If a predictor is wrong on the first transition, which state does the next
one-step call receive? The next recorded observation, not the previous
prediction. This isolates local transition error. A recursive rollout instead
receives its previous prediction, so local error can affect every later state.
Why do recorded future actions still fail to demonstrate controller behavior?
Those actions were chosen for the recorded trajectory; a controller observing
the drifting prediction could choose different torques.
