# Experiment 03 Forward-Dynamics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish simulator, simple, and independent numerical forward-dynamics baselines with auditable one-step and open-loop comparisons.

**Architecture:** Experiment 03 consumes validated Experiment 02 rows at a small predictor seam. Adapters predict state, evaluation owns expected outcomes and timing, evidence modules retain provenance, and reporting derives views from saved records. T1-T3 deliver the first vertical slice; T4 and T5 are parallel-ready; T6 integrates; T7 reproduces and closes.

**Tech Stack:** Python 3.12, MuJoCo 3.16.0, existing NumPy/Matplotlib, standard-library JSON/TOML/hashlib, uv, pytest.

**Spec:** [Forward-Dynamics Design](../specs/2026-11-03-forward-dynamics-design.md)

## Global Constraints

- Preserve `(state_t, applied_torque_t) -> state_{t+1}` exactly.
- Use dataset schema version `1` and dataset version `experiment-02-transition-v1`.
- Preserve complete scenario splits; never fit or choose settings using test results.
- Use Python `>=3.12,<3.13`, MuJoCo `3.16.0`, the committed uv environment, and no new third-party dependency.
- Keep Experiment 03 implementation under `experiments/03-forward-dynamics/` and package it as `forward_dynamics`.
- Report position errors in radians and velocity errors in radians per second, separately.
- Refuse dirty-tree publication by default; record full clean evaluator commit and input checksums.
- Keep Stage 3 planned until its committed artifacts reproduce from a clean checkout.

## Review Focus

- A predictor that accidentally uses expected next state makes an invalid oracle; T1/T3 spy tests enforce the interface and observed-state inputs.
- Reused simulator data can make predictions depend on call order; T1 checks fresh/reset data and A-B-A/reversed calls.
- Bool horizons and partial terminal windows can create false samples; T3 config tests and T4 eligibility tests reject them.
- More accurate integration cannot repair omitted soft joint-limit forces; T5 validates against the exact simplified ODE and retains boundary mismatch.
- Rehashed but internally contradictory saved results must fail report loading; T3/T6 semantic-tampering tests recompute outer hashes.

## Ownership and dependency graph

The design's file table is authoritative. Existing Experiment 01/02 modules are read-only inputs unless a separate defect is demonstrated. New types belong to `forward_dynamics`, not a generic multi-robot framework. Planned file paths below are new until implementation occurs.

```text
T1 -> T2 -> T3 -> T4 --+
              \-> T5 --+-> T6 -> T7
```

T4 and T5 can use separate branches/worktrees once T3 lands. Both add their own modules; T6 owns shared orchestration/registry integration. Do not start implementation as part of ticket creation.

## Ticket execution rule

For each implementation ticket, create its focused failing tests, observe the expected failure, implement the listed interface/objectives, run focused verification and ordinary execution where listed, then commit only its scoped files. Tests should assert public behavior and scientific contracts, not implementation details. Run the full repository suite before integration/publication. A ticket is independently reviewable, not necessarily a public release.

---
### T1: Define the Experiment 03 prediction contract and MuJoCo reference

**Depends on:** none (Experiment 02 is complete)

**Background:** Experiment 02 validates complete recorded trajectories, but consumers do not yet have an independent one-step predictor. Establish the reusable state/action interface before measuring approximation or learning. A fresh-data planning diagnostic reproduced all 2,540 rows exactly on the current environment; the ticket must turn that observation into automated evidence.

**Files:**

- Create experiments/03-forward-dynamics/src/forward_dynamics/{__init__,types,dataset,mujoco_predictor}.py
- Create experiments/03-forward-dynamics/tests/{test_types,test_dataset,test_mujoco_predictor}.py
- Create experiments/03-forward-dynamics/README.md
- Modify pyproject.toml to package forward_dynamics and discover its tests
- Read experiments/02-state-transition-dataset/src/state_transition_dataset/{schema,validation,replay}.py and experiments/01-feedback-control/models/one_link.xml; preserve their contracts

**Interfaces:** Produce frozen DynamicsState(position_rad: float, velocity_rad_s: float), DynamicsContext(environment_id: str, environment_version: str, control_interval_s: float, torque_limit_nm: float), Predictor protocol with predictor_id and predict(state: DynamicsState, applied_torque_nm: float, context: DynamicsContext) -> DynamicsState, and PredictionCase with the identity/state/action/context/boundary fields in the spec. Produce load_prediction_cases(dataset_dir: Path, expected_dataset_version: str) -> tuple[PredictionCase, ...] and MujocoPredictor(model_path: Path), ID mujoco.

**Implementation objectives:**

- Call load_and_validate_dataset(dataset_dir, expected_dataset_version); map all 940/800/800 rows without normalization or repartitioning. Keep expected_next_state in evaluation data only.
- Validate finite non-boolean real inputs, positive interval/torque limit, and already-applied torque within the declared limit; never silently clip. Permit position outside nominal joint limits so terminal states remain representable.
- Verify model SHA-256, supported environment/model contract, one position/velocity/actuator, limits, timestep, and integer control substeps. Fresh/reset simulator data on every call prevents state leakage.
- Document observation-before-action timing, input ownership, units, supported plant, and simulator-only interpretation. Package the module so ordinary imports work outside pytest.

**Acceptance tests:**

- Parameterize NaN, infinity, booleans, nonpositive context values, torque outside limits, wrong checksum/environment, and nonintegral substep intervals; assert ValueError.
- Load committed data and assert len(cases) == 2540 and split counts == {'train': 940, 'validation': 800, 'test': 800}; preserve episode/terminal fields and order.
- Compare fresh MuJoCo predictions to every committed expected next state at absolute tolerance 1e-12, including terminal rows. Predict A, B, A and reverse case order; A must repeat.
- Use a spy at the predictor seam to assert only current state, applied torque, and context are supplied; target/controller output/expected next state cannot enter the prediction call.

**Execution:**

- [ ] Write the acceptance tests above; run the focused command and confirm a missing implementation or incorrect behavior fails.
- [ ] Implement this ticket's interfaces and objectives.
- [ ] Run `uv run pytest -q experiments/03-forward-dynamics/tests/test_types.py experiments/03-forward-dynamics/tests/test_dataset.py experiments/03-forward-dynamics/tests/test_mujoco_predictor.py`; expect success.
- [ ] Run `uv run python -c "import forward_dynamics, feedback_control, state_transition_dataset"`; expect ordinary package/CLI execution to succeed.
- [ ] Review the acceptance criteria and commit scoped files with `feat: define simulator-backed forward dynamics`.

**Learning objective:** Understand a transition model as a function of state, applied action, and timing; distinguish independent one-step prediction from replaying a whole episode.

---

### T2: Add simple dynamics baselines and physical-unit error metrics

**Depends on:** T1

**Background:** A simulator reference establishes consistency, but its error alone says little about predictive usefulness. No-change and constant-velocity baselines give interpretable comparisons and make the predictor seam real. Metrics must expose position and velocity separately.

**Files:**

- Create experiments/03-forward-dynamics/src/forward_dynamics/{baselines,metrics}.py
- Create experiments/03-forward-dynamics/tests/{test_baselines,test_metrics}.py
- Update experiments/03-forward-dynamics/README.md with baseline equations and metric definitions

**Interfaces:** Consume T1 types. Produce NoChangePredictor (no_change), ConstantVelocityPredictor (constant_velocity), StateResidual(position_error_rad: float, velocity_error_rad_s: float), ErrorSummary with sample_count plus per-component MAE/RMSE/max absolute fields from the spec, residual(predicted: DynamicsState, expected: DynamicsState) -> StateResidual, and summarize_errors(errors: tuple[StateResidual, ...]) -> ErrorSummary.

**Implementation objectives:**

- Implement no-change as (q, v) and constant velocity as (q + v*dt, v), applying shared input validation and intentionally ignoring validated torque.
- Use direct signed hinge-angle error predicted minus expected; do not wrap the limited joint or clamp baseline predictions.
- Implement finite, nonempty, per-component MAE/RMSE/max-absolute summaries; keep metrics independent of MuJoCo and file I/O.
- Define aggregate weighting explicitly; keep every episode visible rather than comparing only a best trajectory.

**Acceptance tests:**

- For state (1.0, 2.0), torque 0.0, interval 0.01, assert no-change == (1.0, 2.0), constant velocity == (1.02, 2.0); use pytest.approx for floating arithmetic.
- For position residuals (-1, 3), assert MAE == 2, RMSE == sqrt(5), maximum == 3, count == 2; independently exercise a different velocity residual signal.
- Assert residual from position 3 to expected -3 is +6 radians, not a wrapped value. Reject empty residuals and non-finite values.
- Run each adapter through the shared invalid-input cases; importing metrics must not import mujoco.

**Execution:**

- [ ] Write the acceptance tests above; run the focused command and confirm a missing implementation or incorrect behavior fails.
- [ ] Implement this ticket's interfaces and objectives.
- [ ] Run `uv run pytest -q experiments/03-forward-dynamics/tests/test_baselines.py experiments/03-forward-dynamics/tests/test_metrics.py`; expect success.
- [ ] Review the acceptance criteria and commit scoped files with `feat: add dynamics baselines and component metrics`.

**Learning objective:** Explain why tiny one-step position error can coexist with bad velocity prediction, and why a limited joint needs a different angle-error convention from a continuous joint.

---

### T3: Evaluate and persist held-out one-step dynamics predictions

**Depends on:** T1, T2

**Background:** The first two tickets provide predictors and metrics. This ticket is the first complete experiment command: independently validate Experiment 02, evaluate observed-state one-step predictions, and preserve sufficient evidence to audit every aggregate. Training, validation, and test membership already exists and must remain intact.

**Files:**

- Create experiments/03-forward-dynamics/src/forward_dynamics/{config,evaluation,generation,evidence}.py
- Create experiments/03-forward-dynamics/configs/evaluation-v1.toml
- Create experiments/03-forward-dynamics/scripts/run_evaluation.py
- Create experiments/03-forward-dynamics/tests/{test_config,test_evaluation,test_evidence,test_cli}.py
- Update experiments/03-forward-dynamics/README.md

**Interfaces:** Produce EvaluationConfig and load_evaluation_config(path: Path) -> EvaluationConfig; OneStepPrediction(case: PredictionCase, predictor_id: str, predicted_next_state: DynamicsState, residual: StateResidual); evaluate_one_step(cases: tuple[PredictionCase, ...], predictor: Predictor) -> tuple[OneStepPrediction, ...]; run_evaluation(config_path: Path, output_dir: Path, *, allow_dirty: bool = False) -> Path returning manifest.json; EvaluationEvidence and load_evaluation_evidence(input_dir: Path) -> EvaluationEvidence for independently validated saved records.

**Implementation objectives:**

- Freeze evaluation-v1 config before test comparison: dataset/model identity and relative paths, base adapter IDs, tolerance 1e-12, horizons [1,5,10,25,50,100], and future analytical timesteps [0.002,0.001,0.0005]. Record scope one_step until T6 integration; require this scope in evidence.
- Run each predictor once from each observed state, retaining row identity, expected/predicted next state, and residual. Aggregate per episode/scenario/controller/split with declared transition weighting.
- Write schema-v1 evaluation.json, one_step.jsonl, manifest.json using deterministic order and finite JSON. Preserve input manifest/partition/model/config/uv.lock hashes, dataset source and builder commits, evaluator SHA/dirty flag, adapter settings, runtime/platform, scope, and command.
- Validate counts, expected states, membership, checksums, residuals and aggregates on reload. Reject unsupported schema, conflicting provenance, tampering and non-finite predictor outputs with predictor/episode/step context.
- Reject input/output path overlap before writing. Refuse dirty-tree publication unless --allow-dirty is explicit and recorded. Ensure failed validation does not replace existing evidence; document recovery for interrupted writes.

**Acceptance tests:**

- Use a spy that returns a deliberately wrong state to prove each next call still receives the next recorded state, not the previous prediction.
- Assert one row per (predictor, episode, step), expected 2540 rows per predictor, unchanged split counts, correct per-component aggregate values on a hand-computable fixture, and all terminal rows present.
- Tamper a prediction, its residual, an aggregate, membership and provenance independently; repair outer checksums where necessary so semantic checks must detect the tampering.
- Test config-relative paths from another working directory, bool/float horizons, duplicate IDs/horizons, checksum mismatch, dirty refusal, allowed dirty labeling and output overlap.
- Run CLI in a subprocess with ordinary package imports; repeated same-commit runs produce identical evidence bytes.

**Execution:**

- [ ] Write the acceptance tests above; run the focused command and confirm a missing implementation or incorrect behavior fails.
- [ ] Implement this ticket's interfaces and objectives.
- [ ] Run `uv run pytest -q experiments/03-forward-dynamics/tests/test_config.py experiments/03-forward-dynamics/tests/test_evaluation.py experiments/03-forward-dynamics/tests/test_evidence.py experiments/03-forward-dynamics/tests/test_cli.py`; expect success.
- [ ] Run `uv run python experiments/03-forward-dynamics/scripts/run_evaluation.py --help`; expect ordinary package/CLI execution to succeed.
- [ ] Review the acceptance criteria and commit scoped files with `feat: evaluate held-out one-step dynamics`.

**Learning objective:** Distinguish a held-out protocol from fitting a model, and distinguish dataset source, dataset builder, evaluator and publication provenance.

---

### T4: Measure open-loop prediction drift over recorded action sequences

**Depends on:** T3

**Background:** One-step evaluation corrects the model with an observation before every prediction. Planning needs predictions to feed themselves forward. This ticket measures accumulated error while holding future actions equal to recorded applied torques, so it tests conditional prediction rather than controller performance.

**Files:**

- Create experiments/03-forward-dynamics/src/forward_dynamics/rollouts.py
- Create experiments/03-forward-dynamics/tests/test_rollouts.py
- Update experiments/03-forward-dynamics/README.md with rollout semantics
- Consume config.py/evaluation.py/types.py without editing orchestration; T6 integrates output

**Interfaces:** Produce RolloutPrediction with identity, start_step_index, horizon_steps, horizon_s, predictor_id, predicted/expected endpoint states and residual; RolloutEvaluation(predictions: tuple[RolloutPrediction, ...], windows: tuple[WindowCount, ...]); WindowCount(split, episode_id, horizon_steps, eligible_count, unavailable_count); evaluate_rollouts(cases: tuple[PredictionCase, ...], predictor: Predictor, horizons: tuple[int, ...]) -> RolloutEvaluation. Unavailable-window counts survive independently of prediction records.

**Implementation objectives:**

- For every episode and eligible start, recursively apply recorded torques at horizons 1,5,10,25,50,100. Feed predicted state into subsequent calls; never teacher-force or rerun controllers.
- Require complete contiguous windows with consistent context. Terminal transition may be an endpoint; never continue through a boundary or join episodes/splits.
- Eligibility is predictor-independent. For episode length N and horizon h, eligible windows are max(N-h+1,0), unavailable starts are N-eligible. Record both counts, including horizons with zero eligible windows.
- Retain short failed episodes and all finite predictions, including baseline states beyond joint limits. Fail explicitly on non-finite predictions without dropping difficult windows.
- Return endpoint residuals using shared metrics; label overlapping windows as correlated in documentation.

**Acceptance tests:**

- Use a deterministic accumulating-error predictor; assert horizon-2 input equals horizon-1 prediction and differs from the second observed state.
- For N=3 and horizons (1,2,5), assert eligible counts (3,2,0) and unavailable counts (0,1,3). Verify endpoint alignment and time h*dt.
- Assert horizon-1 results equal corresponding one-step predictions, terminal endpoints are allowed, no window crosses episode/split boundaries, and all adapters receive identical eligible windows.
- Reject duplicate/nonpositive/noninteger/bool horizons, inconsistent context/continuity and non-finite output. Exercise the 70-transition failed episodes at horizon 100.

**Execution:**

- [ ] Write the acceptance tests above; run the focused command and confirm a missing implementation or incorrect behavior fails.
- [ ] Implement this ticket's interfaces and objectives.
- [ ] Run `uv run pytest -q experiments/03-forward-dynamics/tests/test_rollouts.py`; expect success.
- [ ] Review the acceptance criteria and commit scoped files with `feat: evaluate open-loop dynamics drift`.

**Learning objective:** Explain teacher forcing versus recursive prediction, and why recorded future actions do not demonstrate how a controller would behave under model error.

---

### T5: Add an independent analytical RK4 dynamics baseline

**Depends on:** T3

**Background:** MuJoCo consistency cannot tell us how an independent model behaves. A zero-gravity damped rotor provides a useful known-parameter baseline and an exact solution for isolating integration error. Its omission of soft joint-limit/contact forces must remain visible in comparison.

**Files:**

- Create experiments/03-forward-dynamics/src/forward_dynamics/{plant,analytical}.py
- Create experiments/03-forward-dynamics/tests/{test_plant,test_analytical}.py
- Update experiments/03-forward-dynamics/README.md with equations, parameter provenance and omitted physics
- Read experiments/01-feedback-control/models/one_link.xml; preserve the original model
- Consume T1 types and T3 config; T6 owns registry/orchestration integration

**Interfaces:** Produce PlantParameters(environment_id: str, environment_version: str, inertia_kg_m2: float, damping_nm_s_per_rad: float); load_plant_parameters(model_path: Path, expected_environment_version: str) -> PlantParameters; AnalyticalRK4Predictor(parameters: PlantParameters, integration_timestep_s: float) implementing Predictor. Stable IDs are analytical_rk4_0.002, analytical_rk4_0.001, analytical_rk4_0.0005.

**Implementation objectives:**

- Extract the effective generalized hinge inertia, including armature, and damping from the checksummed compiled static plant. Account for inertia frames and parallel-axis shifts; cross-check against the compiled mass matrix at several angles.
- Implement pure fixed-step classical RK4 for qdot=v and vdot=(u-b*v)/J with constant applied action over each control interval. Require an integer positive substep count and finite positive inertia/nonnegative damping.
- Use timesteps 0.002, 0.001 and 0.0005 seconds, frozen before held-out comparison. The prediction module imports no MuJoCo and makes no simulator force/step calls.
- Validate integration against the exact damped constant-torque rotor and its zero-damping limit; report numerical error separately from mismatch against MuJoCo.
- Keep all common evaluation rows/windows, including joint-limit failures. Document missing soft-limit/contact forces; never wrap or clamp predictions to manufacture agreement.

**Acceptance tests:**

- For J=2, b=0, u=4, q0=1, v0=3, dt=0.1, assert q_next approx 1.31 and v_next approx 3.2.
- Use a nonzero damping fixture and an exact exponential solution at 0.01 and 1.0 seconds; show smaller RK4 steps reduce numerical error above floating-point noise.
- Check extracted damping 0.05 and armature contribution 0.01, finite positive inertia, checksum mismatch and unsupported plant features; cross-check generalized inertia at multiple configurations.
- Reject invalid parameters/substeps/context and test torque sign; confirm repeated calls are order independent and no simulator step/forward-force function is invoked by predict.
- Show a joint-limit case remains in evaluation with finite unconstrained prediction and declared approximation, without asserting that finer integration removes model mismatch.

**Execution:**

- [ ] Write the acceptance tests above; run the focused command and confirm a missing implementation or incorrect behavior fails.
- [ ] Implement this ticket's interfaces and objectives.
- [ ] Run `uv run pytest -q experiments/03-forward-dynamics/tests/test_plant.py experiments/03-forward-dynamics/tests/test_analytical.py`; expect success.
- [ ] Review the acceptance criteria and commit scoped files with `feat: add independent analytical dynamics baseline`.

**Learning objective:** Distinguish numerical integration error from structural model error; explain why accurate integration of incomplete equations can still predict the wrong world.

---

### T6: Publish an auditable one-step and rollout baseline comparison

**Depends on:** T4, T5

**Background:** Prediction modules now support reference, simple and analytical adapters. Bring them under one frozen protocol and generate a report from saved records, so future learned models can be assessed against the same evidence rather than a separate evaluation script.

**Files:**

- Modify experiments/03-forward-dynamics/src/forward_dynamics/{config,generation,evidence}.py
- Modify experiments/03-forward-dynamics/scripts/run_evaluation.py
- Create experiments/03-forward-dynamics/src/forward_dynamics/{report_tables,report_plots,report_markdown,reporting}.py
- Create experiments/03-forward-dynamics/scripts/generate_report.py
- Extend tests/{test_config,test_evidence,test_cli}.py and create tests/test_reporting.py under experiments/03-forward-dynamics
- Update experiments/03-forward-dynamics/README.md

**Interfaces:** Extend run_evaluation to save full scope with all six adapters and both prediction modes; preserve its Path return. Extend EvaluationEvidence with validated rollout records/counts and physical parameters. Produce generate_report(input_dir: Path, output_dir: Path) -> Path returning report.md, behind independent evidence loading.

**Implementation objectives:**

- Integrate mujoco, no_change, constant_velocity and all three analytical RK4 step sizes using one config, shared cases, matching window membership and physical-unit metrics.
- Save deterministic evaluation.json, one_step.jsonl, rollouts.jsonl and manifest.json with full scope, input hashes, all adapter parameters, episode outcomes, and eligible/unavailable counts.
- Independently validate expected endpoints/actions/context against the original validated dataset, checksum provenance, residual derivations, all grouping aggregates, and counts before reporting. Do not treat matching outer checksums as semantic validity.
- Generate report/report.md, one_step.csv, rollouts.csv, and four plots: one_step_position.png, one_step_velocity.png, rollout_position.png, rollout_velocity.png. Tables retain per-episode/per-split counts and metrics, plus controller/scenario summaries.
- Explain error conventions, timestep, known-parameter access, simplified analytical physics, terminal episodes, overlapping windows, one-seed/narrow-coverage limits, and the difference between simulator consistency and generalization.
- Report generation consumes saved predictions without simulator calls; zero-valued error plots must render correctly without log-scale artifacts. Avoid learned/control-quality/physical-hardware claims.

**Acceptance tests:**

- Assert six adapters and identical per-adapter case/window identities, exact provenance, parameter values and full-scope declaration in the generated evidence.
- Corrupt rollout endpoints, counts, source IDs, aggregate values and analytical parameters; repair outer hashes to exercise independent semantic validation.
- Assert all report tables match recomputed evidence, all plot data series/counts are correct including zero oracle error, and no simulator query occurs during reporting.
- Run evaluation twice from one clean commit and compare the four evidence files byte-for-byte; generate reports from each and compare CSV bytes and plotted numeric data.

**Execution:**

- [ ] Write the acceptance tests above; run the focused command and confirm a missing implementation or incorrect behavior fails.
- [ ] Implement this ticket's interfaces and objectives.
- [ ] Run `uv run pytest -q experiments/03-forward-dynamics/tests`; expect success.
- [ ] Run `uv run python experiments/03-forward-dynamics/scripts/generate_report.py --help`; expect ordinary package/CLI execution to succeed.
- [ ] Review the acceptance criteria and commit scoped files with `feat: report reproducible forward dynamics comparisons`.

**Learning objective:** Understand a comparison protocol as a reusable research instrument; learned models should enter the same evaluation seam rather than receive favorable private metrics.

---

### T7: Reproduce Experiment 03 from a clean checkout and close Stage 3

**Depends on:** T6

**Background:** Implemented code and locally generated output do not establish reproducibility. Experiment 01 and 02 closed only after independent clean-checkout reproduction. Apply the same evidence standard before making Stage 3 complete or treating it as the Stage 4 baseline.

**Files:**

- Publish experiments/03-forward-dynamics/results/evaluation-v1/{evaluation.json,one_step.jsonl,rollouts.jsonl,manifest.json}
- Publish experiments/03-forward-dynamics/results/evaluation-v1/report/{report.md,one_step.csv,rollouts.csv,one_step_position.png,one_step_velocity.png,rollout_position.png,rollout_velocity.png}
- Update experiments/03-forward-dynamics/README.md
- Update root README.md experiment index and docs/roadmap.md Stage 3 status after verification
- Fix docs/superpowers/specs/2026-10-26-state-transition-dataset-design.md and docs/superpowers/plans/2026-10-26-state-transition-dataset.md only if the previously reported extra EOF blank line is still present

**Interfaces:** Consume T6's committed evaluator, manifest and report entry points. Produce clean-reproduction commands, source/publication commit distinction, committed evidence and an evidence-backed Stage 3 completion gate. No new prediction interface.

**Implementation objectives:**

- Freeze and commit source/config first; run setup, ordinary imports/CLI and all repository tests. Build from that clean evaluator SHA, then publish unchanged generated artifacts in a later commit.
- Use a separate clean checkout/worktree pinned to the evaluator SHA, an isolated uv environment, and documented runtime/platform. Regenerate all four evidence files and compare bytes against publication.
- Regenerate reports from validated saved outputs; compare exact CSV/table values and plotted data. Document rendering/environment limits separately from physics accuracy.
- Verify source/evaluator/publication provenance is noncontradictory, every adapter is covered, terminal/short windows remain represented, and all local documentation links resolve.
- Review standards and ticket/spec coverage, fix findings, rerun affected checks, and update Stage 3 status only after its completion gate is satisfied.
- Keep learned dynamics, normalization fitting, MPC, visual/latent encoding and broader data collection as subsequent work. Stage 4 may reuse this protocol but must address dataset coverage before broad generalization claims.

**Acceptance tests:**

- Run uv sync and uv run pytest -q in the clean pinned checkout; run ordinary package imports and both experiment entry points.
- Compare the four machine-readable evidence files byte-for-byte; independently load the regenerated evidence and derive the report without recomputing predictions.
- Run git diff --check and a relative documentation-link check; verify manifest source SHA resolves and working_tree_dirty is false.
- If EOF cleanup is needed, scope it to the two reported documents; preserve all content and verify one trailing newline.

**Execution:**

- [ ] Write the acceptance tests above; run the focused command and confirm a missing implementation or incorrect behavior fails.
- [ ] Implement this ticket's interfaces and objectives.
- [ ] Run `git diff --check`; expect success.
- [ ] Review the acceptance criteria and commit scoped files with `docs: reproduce and close Experiment 03`.

**Learning objective:** Explain why evaluator commit and publication commit differ, and which claims deterministic simulator reproduction can and cannot support.

## Planning verification

Scoped existing inputs were checked against local and remote commit `6eff203`. The fresh builder check found no substantive blockers. Every design requirement has a ticket owner: prediction/data/model validation T1; simple baselines and residuals T2; configuration, one-step evidence and provenance T3; rollout eligibility T4; independent physics T5; full comparison/report validation T6; clean reproduction/status T7. The dependency graph is acyclic. Proposed files and interfaces remain planned work.

## GitHub tickets

| Ticket | Deliverable | Dependencies |
| --- | --- | --- |
| [#11](https://github.com/Hostileoracle0606/motion-lab/issues/11) | Define the Experiment 03 prediction contract and MuJoCo reference | Experiment 02 complete |
| [#12](https://github.com/Hostileoracle0606/motion-lab/issues/12) | Add simple dynamics baselines and physical-unit error metrics | #11 |
| [#13](https://github.com/Hostileoracle0606/motion-lab/issues/13) | Evaluate and persist held-out one-step dynamics predictions | #11, #12 |
| [#14](https://github.com/Hostileoracle0606/motion-lab/issues/14) | Measure open-loop prediction drift over recorded action sequences | #13 |
| [#15](https://github.com/Hostileoracle0606/motion-lab/issues/15) | Add an independent analytical RK4 dynamics baseline | #13 |
| [#16](https://github.com/Hostileoracle0606/motion-lab/issues/16) | Publish an auditable one-step and rollout baseline comparison | #14, #15 |
| [#17](https://github.com/Hostileoracle0606/motion-lab/issues/17) | Reproduce Experiment 03 from a clean checkout and close Stage 3 | #16 |

Start with #11. After #13, #14 and #15 can proceed in parallel; #16 integrates them and #17 closes the stage.
