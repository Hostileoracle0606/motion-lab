# Engineering Standards

These standards define when work in Motion Lab is reproducible and when a claim is ready to publish. They will evolve with the codebase, but experiments should not silently relax them.

## 1. Evidence levels

Use precise language:

| Label | Required evidence |
| --- | --- |
| Planned | A documented proposal or acceptance criteria |
| Implemented | Source exists and relevant automated checks pass |
| Reproduced | A documented command succeeds from a clean checkout |
| Measured | Raw observations and metric-generation code are retained |
| Compared | Candidates share a declared protocol and constraints |

Do not use “complete,” “faster,” “more stable,” “real time,” or “state of the art” unless the corresponding evidence and definition are available.

## 2. Reproducible environments

Every experiment should declare:

- supported operating system and architecture;
- language and runtime version;
- exact direct dependencies through a lock file or pinned environment;
- MuJoCo and model versions;
- hardware details when timing is reported;
- one setup command and one experiment command, where practical.

A clean-checkout verification should not depend on untracked local files or globally installed packages that are absent from the setup instructions.

## 3. Configuration over hidden constants

Experiment-defining values belong in versioned configuration or a clearly named source module. This includes:

- simulation and control time steps;
- episode length;
- controller gains;
- actuator and safety limits;
- target and initial-state distributions;
- seeds;
- optimizer settings;
- model architecture;
- evaluation tolerances.

Every result should identify the exact configuration used.

## 4. Determinism and seeds

Determinism is a property to test, not assume.

- Record seeds for the environment, data sampler, model initialization, and optimizer when they can differ.
- Use a declared list of seeds for comparisons.
- Report nondeterministic hardware or library behavior that remains.
- Never choose the best seed after inspecting the test results.

A deterministic smoke scenario and a multi-seed evaluation serve different purposes and both may be useful.

## 5. Data contract

Trajectory formats must document:

- field names, data types, and shapes;
- physical units and coordinate frames;
- action semantics and limits;
- time stamps or fixed sampling interval;
- observation timing relative to action application;
- episode identifiers and step indices;
- termination versus time-limit truncation;
- environment, code, and schema versions;
- collection policy and seed.

### Minimum transition schema

The eventual on-disk format may differ, but it should preserve information equivalent to:

```text
episode_id
step_index
time_s
state
action
next_state
terminated
truncated
termination_reason
environment_id
environment_version
controller_id
config_digest
seed
```

### Dataset validation

Automated validation should check:

- required fields and compatible schema version;
- finite numeric values;
- shape and dtype consistency;
- monotonic step and time indices;
- continuity between `next_state[t]` and `state[t+1]` when expected;
- action limits;
- exactly one terminal boundary per completed episode;
- declared split membership.

Large generated data and model checkpoints should not be committed casually. Prefer a documented artifact store and checksums once those systems exist.

## 6. Train, validation, and test separation

- Split at the episode, scenario, or environment level—not by adjacent transition rows.
- Fit normalization only on training data.
- Use validation data for model and hyperparameter choices.
- Keep test scenarios fixed and evaluate them only for final comparisons.
- Declare any distribution shift, such as unseen targets, masses, friction, or camera views.

## 7. Experiment structure

Each experiment should eventually contain or link to:

```text
README.md           question, protocol, status, and commands
configs/            versioned experiment configurations
src/                experiment-specific implementation
tests/              unit and integration checks
scripts/            data collection, evaluation, and plotting entry points
results/             small machine-readable summaries, when appropriate
```

Generated plots must be derivable from machine-readable results. Do not treat a screenshot or manually edited table as the source of truth.

## 8. Controller evaluation

P-versus-PD comparisons must hold constant:

- MuJoCo model and solver settings;
- simulation and controller time steps;
- initial states and targets;
- actuator limits;
- episode duration;
- measurement noise and delay settings;
- metric definitions.

Report at least:

- number of scenarios and seeds;
- gain-selection procedure;
- rise and settling tolerances;
- overshoot and steady-state-error definitions;
- control-effort definition;
- failures, saturation, and constraint violations.

If gains are tuned separately, describe the tuning budget so the comparison is not biased by unequal search effort.

## 9. Dynamics-model evaluation

Measure prediction independently from closed-loop control.

### Prediction metrics

- one-step error on held-out transitions;
- per-component error in physical units;
- wrapped or geometry-aware error for rotations;
- multi-step open-loop error versus horizon;
- calibration or likelihood for probabilistic models;
- inference time under a declared batch size and hardware setup.

### Control metrics

- success or failure rate;
- task cost or return;
- constraint violations;
- planning time and action frequency;
- performance across declared scenario shifts.

A model can have lower average prediction error and still produce worse control. Report both layers.

## 10. Testing expectations

Before an experiment is called implemented, relevant checks should cover:

- controller equations and action clipping;
- time-step and observation/action ordering;
- reset and termination behavior;
- metric calculations on small known signals;
- trajectory serialization round trips;
- schema validation failures;
- model input/output shapes;
- deterministic smoke execution where supported.

Integration tests should be short enough for routine use. Longer evaluation suites can run separately but must remain scripted.

## 11. Reporting results

Every published result should include:

- commit identifier;
- configuration or digest;
- dependency/environment identifier;
- dataset version or checksum;
- hardware when relevant;
- sample count and seed policy;
- aggregate and dispersion, not only the best trial;
- failure cases and excluded runs with reasons;
- command used to regenerate the result.

Keep observed facts separate from interpretation. For example:

```text
Observed: Under protocol X, controller A had a lower median settling time.
Interpretation: The derivative term may have improved damping in these scenarios.
Unknown: Whether this conclusion transfers to different models or noisy sensors.
```

## 12. Change review checklist

Before merging experiment work:

- [ ] The status label matches the available evidence.
- [ ] New interfaces document units, shapes, timing, and ownership.
- [ ] Tests cover important failure boundaries.
- [ ] Commands work from a clean checkout.
- [ ] Generated artifacts are reproducible from versioned inputs.
- [ ] No credentials, tokens, private datasets, or machine-specific paths are committed.
- [ ] Documentation distinguishes results from hypotheses and future work.
