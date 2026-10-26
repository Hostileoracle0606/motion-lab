# Dataset Card — Experiment 02 Transition Dataset v1

## Summary

`experiment-02-transition-v1` is a small, deterministic state-transition
dataset derived from the committed Experiment 01 one-link MuJoCo evaluation.
It contains 2,540 transitions from eight episodes: two controllers, four
scenarios, and one declared seed. It exists to provide a trustworthy input for
a later forward-dynamics experiment. It contains no learned model, benchmark,
or model-performance result.

The authoritative machine-readable description is
[`data/dataset-v1/manifest.json`](data/dataset-v1/manifest.json). It records the
dataset config checksum, clean builder commit, source evaluation checksum,
source code commit, partition checksums, episode membership, and replay errors.

## Collection policy and source

The builder consumes the validated
[Experiment 01 evaluation](../01-feedback-control/results/evaluation-v1/evaluation.json)
and its eight checksummed trajectory files. It does not rerun the P or PD
controllers and does not resample, interpolate, shuffle, or normalize their
transitions. Source trajectory order is preserved within each partition.

The source commit is
`7a6d152f5d6f9358681e1b67ad08cbb1f0c3484c`. The source protocol uses MuJoCo
3.16.0, a fixed 0.002 s simulation step, a 0.010 s controller interval, no
process or measurement noise, and seed `0`. The builder independently validates
the source evaluation summary, configuration checksum, trajectory checksums,
episode plan, derived metrics, and failure outcomes before conversion.

## Coverage and partitions

Splits are assigned by complete scenario. Every seed, both controllers, and all
adjacent transitions for one scenario remain together.

| Partition | Scenarios | Episodes | Transitions |
| --- | --- | ---: | ---: |
| Train | `negative-to-positive-wide`, `positive-to-negative-wide` | 4 | 940 |
| Validation | `negative-to-positive-narrow` | 2 | 800 |
| Test | `positive-to-negative-narrow` | 2 | 800 |
| Total | 4 | 8 | 2,540 |

The validation and test partitions therefore hold out narrow-displacement
scenarios by direction. This is a declared scenario shift, not a statistically
representative sample. The dataset has only one seed.

Two P-controller wide-displacement episodes terminate after 70 transitions at
a joint-limit violation. Their terminal boundaries and failure reasons are
retained rather than discarded. The remaining six episodes contain 400
transitions each. Experiment 01's other P-controller failure outcomes are also
preserved in the source evidence; this dataset does not add a success-only
filter.

## Row schema

Each JSONL object has dataset schema version `1` and exactly these fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `schema_version` | integer | Row schema; exactly `1` |
| `dataset_version` | string | Dataset identity |
| `split` | string | `train`, `validation`, or `test` |
| `episode_id` | string | Stable source episode identity |
| `scenario_id` | string | Whole-scenario split unit |
| `controller_id` | string | Source controller, `p` or `pd` |
| `seed` | integer | Source random seed |
| `step_index` | integer | Contiguous control step from zero |
| `time_s` | number | Pre-action observation time, seconds |
| `control_interval_s` | number | Action duration, seconds |
| `state_position_rad` | number | Joint angle at time `t`, radians |
| `state_velocity_rad_s` | number | Joint angular velocity at `t`, rad/s |
| `action_applied_torque_nm` | number | Clipped torque actually sent to MuJoCo, N m |
| `next_state_position_rad` | number | Joint angle after the action, radians |
| `next_state_velocity_rad_s` | number | Joint angular velocity after the action, rad/s |
| `target_position_rad` | number | Controller target angle, radians |
| `target_velocity_rad_s` | number | Controller target angular velocity, rad/s |
| `torque_limit_nm` | number | Symmetric actuator limit magnitude, N m |
| `terminated` | boolean | Environment-ended boundary flag |
| `truncated` | boolean | Time-limit boundary flag |
| `termination_reason` | string or null | Boundary cause; null before the final row |
| `environment_id` | string | Stable one-link environment name |
| `environment_version` | string | MuJoCo model SHA-256 |
| `source_code_version` | string | Source Experiment 01 Git commit |
| `config_digest` | string | Source episode configuration SHA-256 |
| `source_trajectory_sha256` | string | Source trajectory file SHA-256 |

All numeric row values are finite scalars. This first schema is intentionally
specific to the one-position, one-velocity, one-actuator system; it does not
pretend scalar shapes generalize to multi-joint robots.

## Coordinates, units, and action timing

Position is the generalized angle of the one-link hinge about the MuJoCo
model's y axis. Positive rotation moves the arm toward positive x. Position is
in radians, angular velocity in radians per second, torque in newton-metres, and
time in seconds. The model disables gravity and constrains the hinge to
`[-pi/2, +pi/2]`.

Each row has observation-before-action semantics:

```text
(state_t, action_applied_torque_t) -> state_{t+1}
```

The controller chose the saved action from `state_t`. MuJoCo received that
applied action for five 0.002 s simulation substeps, totaling the recorded
0.010 s control interval, before `state_{t+1}` was observed. Raw unclipped
controller output is deliberately not the dynamics action.

## Validation and replay evidence

The independent loader rejects:

- unknown or missing fields, unsupported versions, and non-finite values;
- invalid action limits, indices, times, state continuity, or terminal
  boundaries;
- duplicate transition keys or episode/scenario leakage across partitions;
- rows without declared episode membership;
- partition checksum or count mismatches;
- replay errors beyond the declared tolerance.

For every episode, the builder checks the MuJoCo model SHA-256, resets its
recorded keyframe and initial state, applies the saved
`action_applied_torque_nm` sequence, and compares each replayed next state with
the saved next state. The manifest records maximum position and velocity errors
of `0.0` for all eight episodes at absolute tolerance `1e-12` on the recorded
environment.

This verifies deterministic simulator replay only. It does not validate model
fidelity to physical hardware, sensor noise, actuator dynamics, real-time
execution, or safety.

## Reproduction

From a clean checkout of the builder commit recorded in the manifest:

```bash
uv sync
uv run pytest -q
uv run python experiments/02-state-transition-dataset/scripts/build_dataset.py
```

The exact separate-worktree byte-comparison procedure is documented in the
[Experiment 02 README](README.md#reproduce-the-committed-bytes).

## Intended use

This dataset is suitable for testing loaders, forward-dynamics interfaces, and
small simulation-only prediction baselines that explicitly respect the
scenario split. Any normalization or fitted preprocessing must be derived from
the training partition only.

## Limitations and prohibited interpretations

- The data comes from one deterministic simulated link, not a physical robot.
- Coverage is four prescribed scenarios, two feedback policies, and one seed.
- Validation and test each contain one scenario, so aggregate performance would
  have high uncertainty and weak external validity.
- Controller-generated data is not broad excitation of the state-action space.
- Early joint-limit failures are retained and may require boundary-aware use.
- Deterministic replay does not establish cross-platform bitwise physics,
  real-time behavior, production readiness, or safety.
- No learned model has been trained or measured. This artifact supports no
  accuracy, generalization, control-quality, or state-of-the-art claim.
