# Experiment 03 — Forward dynamics

## Question and current scope

How well can simple state-transition models predict the next joint position
and velocity, and how should their errors be measured?

Issue [#12](https://github.com/Hostileoracle0606/motion-lab/issues/12) implements
two deterministic baselines and pure error metrics. Its minimal prerequisites
are the shared prediction types, validation, and package setup from
[#11](https://github.com/Hostileoracle0606/motion-lab/issues/11). The MuJoCo
reference adapter and dataset conversion remain in #11. Held-out evaluation,
recursive rollouts, analytical integration, and published comparisons remain
planned. Stage 3 is not complete and no predictive accuracy has been measured.

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
identity strings; checksum/model compatibility checks belong to the future
simulator adapter.

Both adapters satisfy the structural `Predictor` interface:

```python
predict(state, applied_torque_nm, context) -> DynamicsState
```

The observation comes before the action: `(state_t, applied_torque_t)` predicts
`state_{t+1}` after `control_interval_s`. The dataset's normal interval is
`0.010 s`; these equations also support other finite positive intervals.
Neither adapter receives target states, raw controller output, or expected
next states. Both deliberately ignore torque after checking that it is finite
and within the context limit. They do not simulate acceleration or damping.

Validation occurs at prediction and metric function entry, not when raw frozen
records are constructed. `validate_prediction_inputs` is the shared adapter
validator. Position, velocity, and torque must be finite real scalars;
booleans, strings, arrays, and non-finite values raise `ValueError`. Interval
and torque limit must also be strictly positive. Applied actions outside the
limit raise `ValueError` rather than being silently clipped. Non-finite
computed predictions and residuals are rejected too.

| Adapter ID | Predicted position | Predicted velocity |
| --- | --- | --- |
| `no_change` | `q` | `v` |
| `constant_velocity` | `q + v * dt` | `v` |

For `q = 1 rad`, `v = 2 rad/s`, and `dt = 0.01 s`, these return `(1, 2)` and
`(1.02, 2)` respectively. Position is never wrapped or clamped: an excursion
beyond the nominal joint range must remain visible to evaluation.

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
future evaluator must also call these functions per episode and retain those
results, scenario/controller/split membership, counts, and failures. These pure
functions do not choose groups, select episodes, perform file I/O, or evaluate
the held-out dataset.

## Learning checkpoint

At `dt = 0.01 s`, an acceleration error of `10 rad/s^2` produces a velocity
error of `0.1 rad/s` but a position error of only `0.0005 rad` under constant
acceleration. Small one-step position error can therefore coexist with a
meaningful velocity error, which will affect later position predictions.

Why would wrapping a predicted joint-limit excursion make a baseline look
better? It could replace a large direct error with a smaller equivalent-angle
error even though this limited joint cannot rotate through that path. Retain
the excursion and measure each coordinate in its own units.
