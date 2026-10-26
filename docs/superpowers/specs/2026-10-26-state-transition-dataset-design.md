# Experiment 02 State-Transition Dataset Design

**Issue:** [#8 — Begin Experiment 02 with a versioned state-transition dataset](https://github.com/Hostileoracle0606/motion-lab/issues/8)

## Intent

Turn the validated Experiment 01 episodes into a small dataset that a later dynamics model can consume without losing the scientific context that makes each transition trustworthy. The dataset is a data-engineering artifact, not a learned-model result.

## Goals

- Define an explicit dataset schema and dataset version.
- Materialize deterministic JSONL partitions for training, validation, and test use.
- Split by whole scenario so adjacent transitions and paired P/PD episodes never cross partitions.
- Reject missing, non-finite, inconsistent, checksum-mismatched, or leaked records.
- Replay every saved applied-action sequence in MuJoCo and record the maximum state error.
- Publish a small byte-reproducible dataset and a dataset card.

## Non-goals and anti-goals

- Do not train, select, or evaluate a learned model.
- Do not fit normalization statistics; that belongs to the future training pipeline and must use training data only.
- Do not claim stochastic coverage from the single declared seed.
- Do not invent a second simulation or trajectory contract; Experiment 01 remains the source of truth.
- Do not shuffle individual transition rows into splits.

## Approaches considered

### Selected: materialized JSONL plus a manifest

Write one line per transition into `train.jsonl`, `validation.jsonl`, and `test.jsonl`, plus a checksummed `manifest.json`. JSONL is directly consumable, diffable, dependency-free, and small enough for this first dataset. The manifest retains episode, scenario, source checksum, environment, builder provenance, replay evidence, and partition counts.

### Rejected: manifest-only wrapper around Experiment 01 trajectories

This would avoid duplicated numbers, but every future consumer would need to understand the Experiment 01 control schema before it could train on `(state, action, next_state)`. It is useful as provenance, not sufficient as the reusable dataset itself.

### Rejected: Parquet, HDF5, or SQLite

Those formats scale better, but this dataset is only a few thousand scalar transitions. They would add dependencies and hide the first schema from human inspection before scale requires them.

## Dataset contract

Dataset schema version `1` writes finite scalar fields with explicit units:

- identity: `episode_id`, `scenario_id`, `split`, `controller_id`, `seed`, `step_index`;
- timing: `time_s`, `control_interval_s`;
- state: `state_position_rad`, `state_velocity_rad_s`;
- action: `action_applied_torque_nm`;
- next state: `next_state_position_rad`, `next_state_velocity_rad_s`;
- boundary: `terminated`, `truncated`, `termination_reason`;
- context: target position/velocity, torque limit, environment ID/version, source code version, configuration digest, and source trajectory SHA-256.

Each row preserves the Experiment 01 temporal meaning:

```text
(state_t, applied_torque_t) -> state_{t+1}
```

The action was selected from `state_t` and applied for `control_interval_s` before `state_{t+1}` was observed.

## Split policy

The versioned configuration assigns complete scenarios:

- train: `negative-to-positive-wide`, `positive-to-negative-wide`;
- validation: `negative-to-positive-narrow`;
- test: `positive-to-negative-narrow`.

Both controllers and every seed for one scenario stay together. Validation and test therefore hold out narrow-displacement scenarios while training retains both motion directions. This is a declared distribution shift, not a performance experiment.

## Validation

The loader validates the manifest and every JSONL row. It checks exact field sets, schema and dataset versions, finite scalar values, action limits, monotonic and contiguous episode indices/times, state continuity, one final terminal boundary per episode, declared partition membership, partition checksums/counts, unique `(episode_id, step_index)` keys, and absence of episode or scenario leakage across splits.

## Deterministic replay

For every episode, reset the declared MuJoCo model to the recorded initial state, apply each saved `action_applied_torque_nm` for the recorded number of simulator substeps, and compare the simulated next state with the saved next state. The manifest records maximum absolute position and velocity error and the configured absolute tolerance. Replay is evidence for this deterministic simulator configuration only; it is not physical-system validation.

## Published artifacts

`experiments/02-state-transition-dataset/data/dataset-v1/` contains:

- `manifest.json`;
- `train.jsonl`;
- `validation.jsonl`;
- `test.jsonl`.

`DATASET_CARD.md` documents collection policy, schema, units, coordinate frame, timing, coverage, splits, provenance, replay, failures, limitations, and exact reproduction commands.

## Completion boundary

Experiment 02 is complete when a clean checkout at the manifest's builder commit passes all tests, regenerates these four files byte-for-byte, and the repository status pages link to the dataset card. No claim extends to model accuracy or physical hardware.

