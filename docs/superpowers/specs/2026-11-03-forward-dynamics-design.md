# Experiment 03 Forward-Dynamics Design

**Status:** Planned. This document records the simulator-first approach accepted in the conversation on 2026-11-03; it does not establish implementation or measured model performance.

## Intent and world-model connection

Build a small, replaceable action-conditioned predictor and a trustworthy way to evaluate it. Experiment 01 supplies simulated control episodes; Experiment 02 supplies validated transitions. Experiment 03 establishes the prediction contract, physical-unit metrics, and open-loop evaluation that Stage 4 learned dynamics and Stage 5 MPC can reuse. Stage 6 will need an observation encoder and a different representation contract; the scalar state type here is not a promise that latent models share its exact types.

## Existing evidence and ownership

- Baseline source is `6eff20320c4d2fca8a43335828061ab5073dd081`.
- `experiments/02-state-transition-dataset/src/state_transition_dataset/validation.py` owns `load_and_validate_dataset(output_dir: Path, expected_dataset_version: str) -> ValidatedDataset`.
- Dataset `experiment-02-transition-v1` contains 2,540 transitions: train 940, validation 800, test 800. Split membership remains unchanged.
- `feedback_control/simulation.py` owns controller-driven simulation. `state_transition_dataset/replay.py` verifies complete saved action sequences. Neither becomes the Experiment 03 predictor interface.
- The authoritative plant is `experiments/01-feedback-control/models/one_link.xml`: one hinge, one direct torque actuator, zero gravity, RK4, simulation timestep `0.002 s`, control interval `0.010 s`, damping `0.05`, armature `0.01`, actuator limits `[-3, 3] N m`, and joint limits approximately `[-pi/2, pi/2]`.
- Planning diagnostic on 2026-11-03: fresh MuJoCo data initialized from each row's position/velocity, followed by five saved-torque substeps, matched all 2,540 saved next states with maximum position and velocity errors `0.0` in the current environment. This is a feasibility check, not published Experiment 03 evidence or cross-platform validation.

## Approaches

**Selected: simulator first, independent physics later.** Establish one prediction interface and a MuJoCo adapter, then simple baselines and one-step evaluation. Add open-loop evaluation and a separate numerical physics adapter before comparison and clean reproduction. This exposes timing and data errors before introducing physics approximation.

**Analytical first:** educational but mixes parameter extraction, integration, and simplified joint-limit behavior into the first deliverable.

**Learned dynamics now:** premature because there is no independent prediction protocol yet. Learning remains Stage 4.

## Global constraints

- Preserve `(state_t, applied_torque_t) -> state_{t+1}` exactly.
- Use dataset schema version `1` and dataset version `experiment-02-transition-v1`.
- Preserve complete scenario splits; never fit or choose settings using test results.
- Use Python `>=3.12,<3.13`, MuJoCo `3.16.0`, the committed uv environment, and no new third-party dependency.
- Keep Experiment 03 implementation under `experiments/03-forward-dynamics/` and package it as `forward_dynamics`.
- Report position errors in radians and velocity errors in radians per second, separately.
- Refuse dirty-tree publication by default; record full clean evaluator commit and input checksums.
- Keep Stage 3 planned until its committed artifacts reproduce from a clean checkout.

## Prediction interface

`types.py` defines frozen `DynamicsState(position_rad: float, velocity_rad_s: float)` and `DynamicsContext(environment_id: str, environment_version: str, control_interval_s: float, torque_limit_nm: float)`.

`Predictor` is a structural protocol with a stable `predictor_id: str` and:

```python
predict(
    state: DynamicsState,
    applied_torque_nm: float,
    context: DynamicsContext,
) -> DynamicsState
```

State and torque values must be finite real scalars; reject booleans. Context interval and torque limit must be finite and strictly positive. Applied torque outside the declared limit raises `ValueError`; the predictor never silently clips it. State position may lie beyond the nominal joint range because boundary transitions and unconstrained baseline predictions must remain inspectable. Input validation is shared across adapters.

`dataset.py` defines `PredictionCase(split, episode_id, scenario_id, controller_id, seed, step_index, state, applied_torque_nm, expected_next_state, context, terminated, truncated, termination_reason)` and `load_prediction_cases(dataset_dir: Path, expected_dataset_version: str) -> tuple[PredictionCase, ...]`. It uses the existing independent dataset validator, preserves every transition and its partition, and orders cases by split, episode ID, then step index. No target, raw torque, or expected next state is passed to `predict`.

## Adapters

### MuJoCo reference

`MujocoPredictor(model_path: Path)` verifies model bytes against `context.environment_version`, the environment ID, dimensions, torque range, timestep, and supported memoryless one-link model contract. It uses fresh/reset simulator data on every prediction, initializes the supplied position and velocity, calls forward dynamics initialization, applies the saved torque for an integer number of substeps, and returns finite position/velocity. Calls must be order independent. A generic MuJoCo model is not supported merely because its dimensions match; models with additional actuator state, external forces, contacts/history-dependent behavior, or different dynamics need an expanded contract.

### Simple baselines

`NoChangePredictor`, ID `no_change`: return the current state.

`ConstantVelocityPredictor`, ID `constant_velocity`: return `(q + v * dt, v)`. Both intentionally ignore action after validating it; neither clamps position.

### Independent numerical physics baseline

`AnalyticalRK4Predictor(parameters: PlantParameters, integration_timestep_s: float)`, with IDs `analytical_rk4_0.002`, `analytical_rk4_0.001`, and `analytical_rk4_0.0005`, integrates:

```text
dq/dt = v
dv/dt = (applied_torque_nm - damping_nm_s_per_rad * v) / inertia_kg_m2
```

`PlantParameters(environment_id, environment_version, inertia_kg_m2, damping_nm_s_per_rad)` contains effective hinge inertia including armature. `load_plant_parameters(model_path: Path, expected_environment_version: str) -> PlantParameters` may use MuJoCo's compiled static model for mass, inertia, frame transforms, and damping. It records those values and their source checksum. The numerical predictor itself performs no simulator stepping or simulator force query and imports no MuJoCo. Parameter extraction is cross-checked against the compiled generalized inertia at several configurations; the integration is independently checked against the damped rotor's exact solution, including zero damping.

This baseline omits MuJoCo soft joint-limit and contact forces. It evaluates the same rows/windows as other adapters and retains boundary cases. Reports identify that structural limitation and distinguish integration error against the exact simplified ODE from model mismatch against MuJoCo. A finer timestep cannot be claimed to fix omitted physics. These are known-parameter baselines, with no learned or fitted coefficients.

## Metrics

The limited hinge uses direct signed position residual `predicted - expected`; velocity uses the same convention. MAE, RMSE, and maximum absolute error are calculated separately per state component using finite, nonempty residuals. No wrapped angle metric is used for this limited joint: wrapping would hide excursions. Any future continuous-joint predictor must explicitly introduce a different geometry contract.

`StateResidual(position_error_rad, velocity_error_rad_s)` and `ErrorSummary(sample_count, position_mae_rad, position_rmse_rad, position_max_abs_rad, velocity_mae_rad_s, velocity_rmse_rad_s, velocity_max_abs_rad_s)` are the shared metric records.

Retain per-transition predictions and residuals; aggregate per episode, scenario, controller, and split. Split totals are transition-weighted, explicitly labeled; episode-level dispersion remains visible. The MuJoCo `1e-12` absolute tolerance is a reference check in the recorded environment, not a required accuracy for simple or analytical baselines.

## One-step and rollout evaluation

`evaluate_one_step(cases: tuple[PredictionCase, ...], predictor: Predictor) -> tuple[OneStepPrediction, ...]` resets conceptually at every observed state. `OneStepPrediction(case: PredictionCase, predictor_id: str, predicted_next_state: DynamicsState, residual: StateResidual)` retains identity and expected state without exposing them to the predictor.

`evaluate_rollouts(cases: tuple[PredictionCase, ...], predictor: Predictor, horizons: tuple[int, ...]) -> RolloutEvaluation` starts at every eligible recorded state, feeds predictions back as future inputs, and uses the recorded applied-action sequence. `RolloutEvaluation(predictions: tuple[RolloutPrediction, ...], windows: tuple[WindowCount, ...])` retains both predictions and unavailable-window counts. `WindowCount` records split, episode ID, horizon steps, eligible count, and unavailable count. `RolloutPrediction` records split, episode/scenario/controller/seed, start step, horizon in steps and seconds, predictor ID, predicted and expected endpoint states, and residual. Intermediate observations must not reset the model. This is conditional open-loop prediction under recorded actions, not closed-loop controller evaluation.

Use horizons `(1, 5, 10, 25, 50, 100)` control steps, corresponding to `(0.01, 0.05, 0.10, 0.25, 0.50, 1.00)` seconds. Only use complete windows within one episode; the final terminal transition may be the endpoint, but no window crosses it. Eligibility depends on the recorded data, equally for every predictor. Record eligible and unavailable window counts per episode/horizon; preserve short failed episodes. Overlapping windows are correlated and are not independent trials.

A non-finite prediction fails the run with predictor, episode, and step context; it must never create a silently filtered comparison. Horizon values must be unique positive integers, excluding booleans, in increasing order. Context and state continuity must be consistent within an episode.

## Configuration, evidence, and reporting

`configs/evaluation-v1.toml` declares dataset/model paths (resolved relative to that file), dataset version, environment identity, adapter IDs, simulator tolerance `1e-12`, horizons above, and analytical substeps `(0.002, 0.001, 0.0005)` seconds. Choices are frozen before test comparison. No normalization, fitting, or test-driven tuning occurs.

One-step evaluation initially exposes `scripts/run_evaluation.py --config ... --output ... [--allow-dirty]`. Final integration adds rollouts and all adapters using the same entry point. It saves `evaluation.json`, `one_step.jsonl`, `rollouts.jsonl`, and `manifest.json` under `results/evaluation-v1/`. An intermediate one-step run declares its scope and cannot masquerade as a complete Stage 3 run.

Result schema version `1` retains exact input manifest/partition/model/config/uv.lock checksums, dataset builder/source commits, clean evaluator commit, Python/MuJoCo/OS/architecture, adapter parameters, scope, counts, eligible/unavailable windows, and command. Distinguish dataset source, dataset builder, evaluator, and later publication commits. Use sorted deterministic serialization, finite JSON numbers, stable ordering, and paths relative to the repository; omit timestamps, absolute worktree paths, and measured execution times from reproducible evidence. Input and output trees must not overlap, and existing outputs need a documented safe rebuild procedure.

Reporting independently validates all checksums, record membership, expected values against the validated dataset, counts, window eligibility, prediction/residual consistency, and regenerated aggregates before generating `report/report.md`, CSV tables, and per-component one-step/error-versus-horizon plots. It consumes saved predictions without querying MuJoCo. Error-vs-horizon comparisons show sample counts and distinguish the simplified analytical plant from the simulator reference. No timing benchmark is required in this phase; latency and calibration remain future extensions when relevant.

Stage 3 closes after a clean evaluator commit reproduces the four evidence files byte-for-byte in the recorded environment, reports regenerate from those bytes with exact tables and numerically identical plotted data, documentation links resolve, ordinary package/CLI execution works, and the full repository tests pass. Image/PDF byte identity is not the scientific comparison criterion. Root README stays global; experiment-specific details belong in the local README. Cross-platform and physical-system fidelity remain unmeasured.

## File responsibilities and delivery slices

All new source is under `experiments/03-forward-dynamics/src/forward_dynamics/`:

| File | Responsibility |
| --- | --- |
| `types.py`, `dataset.py` | Small prediction interface, validation, dataset conversion |
| `mujoco_predictor.py` | Simulator reference adapter |
| `baselines.py`, `metrics.py` | Simple adapters and pure physical-unit metrics |
| `config.py`, `evaluation.py` | Frozen protocol and one-step prediction records |
| `rollouts.py` | Recursive open-loop windows and endpoint records |
| `plant.py`, `analytical.py` | Static physical-parameter extraction and pure RK4 integration |
| `generation.py`, `evidence.py` | Orchestration, provenance, serialization, independent evidence loading |
| `report_tables.py`, `report_plots.py`, `report_markdown.py`, `reporting.py` | Derived views and report orchestration |

Seven tickets: prediction contract and MuJoCo adapter; simple baselines/metrics; one-step evaluator; rollout evaluation; analytical adapter; integrated comparison/report; clean reproduction/closure. Rollouts and analytical physics can proceed independently after the one-step evaluator. Each ticket has a learning objective and a verification gate. No product implementation is authorized by this planning document alone.

## Planning review

A fresh builder check found no substantive blocking questions about state sufficiency, the analytical approximation, rollout windows, or reproduction. The plan retains unavailable-window counts separately from prediction records so short episodes remain visible at horizons with no complete windows. Ticket creation is the authorized deliverable; implementation remains subsequent work.
