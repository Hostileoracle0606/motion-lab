# Learned Dynamics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Train a small residual dynamics model and compare its prediction, rollout, and CPU/MPS behavior using independently auditable evidence.

**Architecture:** Experiment 04 composes the existing validated dataset and pure dynamics types/metrics through new data, model, training, evidence, benchmark, and report modules. Its evidence schema is separate from Experiment 03's fixed six-adapter schema. Training, saved-evidence auditing, checkpoint replay, and reporting have distinct entry points.

**Tech Stack:** Python 3.12, uv, NumPy, PyTorch (candidate pin 2.15.0), Matplotlib, pytest; macOS arm64 CPU and MPS.

**Spec:** [approved design](../specs/2026-11-10-learned-dynamics-design.md), issue #24. Approval recorded in chat on 2026-11-11. This plan completes #24; product implementation belongs to #25 onward.

## Global Constraints

- Preserve dataset v1 and its 940/800/800 scenario-safe splits and all terminal rows. Train-only normalization uses float64 population statistics, a 1e-12 constant-feature threshold, and scale 1.0 for constant features.
- Canonical inputs are position rad, velocity rad/s, applied torque N m; outputs are delta position/velocity. The control interval is 0.010 s with five 0.002 s physics steps. Finite states are never clamped or angle-wrapped.
- MPS is primary; CPU is a measured comparison. Use float32 model computation and identical CPU initialization/permutations per device/seed. No silent device fallback.
- Canonical architecture 3->32->32->2 with tanh hidden layers; seeds [0,1,2,3,4]; Adam lr 0.001, betas (0.9,0.999), eps 1e-8, weight_decay 0; batches 64; 300 epochs unless numerical failure; no early stopping/scheduler.
- Initialization seed is run seed; sampler seed is 100000+run seed. Select earliest minimum validation normalized one-step MSE checkpoint. Test rows never enter fitting or selection.
- One-step/rollout horizons [1,5,10,25,50,100], physical-unit component MAE/RMSE/max; report all failures/unavailable windows and all seeds. Overlapping windows are correlated.
- Preserve all earlier dependencies' exact direct pins and Experiment 03 source contracts/artifact bytes. Candidate torch compatibility is unverified until Task 3.
- Checkpoints are inference-only JSON+NPZ with allow_pickle=False; no resume. Total publication budget 500 MiB; source/config commit precedes publication; source SHA remains reachable.
- Repeated same-device inference tolerances: 1e-6 rad, 1e-5 rad/s. Same-device retraining validation MAE tolerances: 1e-4 rad, 1e-3 rad/s. Do not relax tolerances after test inspection.
- Follow CONTRIBUTING.md and docs/engineering-standards.md. Only independently reproduced evidence supports Stage 4 closure; MPC and new data collection are later scopes.

## Review Focus

1. An integrity loader reads all files: training/selection APIs must still receive no test rows (Tasks 1, 5).
2. A history or recursive loss window touches an episode boundary: only complete, continuous windows are eligible; both study arms share starts (Tasks 2, 6).
3. Requested MPS is missing or an operation is unsupported: produce a contextual failure without CPU substitution (Tasks 3, 8).
4. Predictions diverge or a checkpoint/manifest is tampered with: retain denominators and reject repaired-checksum semantic corruption (Tasks 4, 7).
5. A GPU call is asynchronous or reporting can execute inference: synchronize timing and prove report-only operation with runtime traps (Tasks 8, 9).

## File ownership and dependency order

All future paths below are relative to `experiments/04-learned-dynamics/`.
The source package is `src/learned_dynamics/`; tests are under `tests/`.

| Ticket/tasks | Files and responsibility | Dependencies |
| --- | --- | --- |
| #25 / 1–2 | types.py, data.py, normalization.py, windows.py: checked training views and transforms | approved #24 |
| #26 / 3–4 | model.py, predictor.py, devices.py, checkpoint.py: inference and bundles | approved #24 contracts; Task 1 types |
| #27 / 5–6 | config.py, training.py, losses.py, training_evidence.py: fitting/selection | #25, #26 |
| #28 / 7 | evaluation.py, evidence.py, evaluation_config.py: saved prediction audit | #25, #26; use roster fixtures until #27 |
| #29 / 8–9 | benchmarking.py, studies.py, report_tables.py, report_plots.py, report_markdown.py, reporting.py | #27, #28 |
| #30 / 10 | reproduction instructions and selected results | #29 |
| #31 / follow-up | CASE_STUDY.md: reader-facing evidence and decisions | #29, #30 |

#25 and #26 can be isolated in parallel once the shared Task 1 types land.
#25 owns package/test registration; #26 owns torch pin/lock resolution.
Serialize edits to root pyproject.toml/uv.lock. #27 and #28 can then proceed
in parallel. Do not execute the entire stage merely because this plan exists.

## Shared contracts and schemas

Use dataclasses frozen where possible. Every schema rejects extra/missing
fields, booleans as integers, non-finite numbers, duplicate JSON keys and
unsupported versions. Reuse `forward_dynamics.serialization` JSON/hash helpers,
`DynamicsState`, `DynamicsContext`, `PredictionCase` and metric functions.

- `SplitView(split: Literal["train","validation","test"], cases: tuple[PredictionCase,...], dataset_version: str, manifest_sha256: str, train_sha256: str)`.
- `TrainingView` is a separately checked wrapper around a train SplitView;
  constructor rejects any non-train case. `DataViews(train: TrainingView, validation: SplitView, test: SplitView)`.
- `Normalizer(schema_version: int, feature_order: tuple[str,...], feature_units: tuple[str,...], input_mean: tuple[float,...], input_scale: tuple[float,...], input_constant: tuple[bool,...], delta_mean: tuple[float,float], delta_scale: tuple[float,float], delta_constant: tuple[bool,bool], fit_row_count: int, dataset_version: str, manifest_sha256: str, train_sha256: str)`; canonical input fields have length 3. History reuses the same three scales twice.
- `Window(cases: tuple[PredictionCase,...], previous: PredictionCase|None)` and `WindowCoverage(split: str, episode_id: str, horizon: int, eligible: int, unavailable: int, missing_history: int)`.
- `ModelSpec(hidden_width: int=32, history_steps: int=0)`; input width 3 or 6, output width 2.
- `RunSpec(variant_id: str, model: ModelSpec, loss: Literal["one_step","rollout_5"], membership: Literal["all","history_common","loss_common"], seed: int, device: Literal["cpu","mps"], repeat: int)`.
- `TrainingConfig` contains strict schema_version, dataset_dir/version,
  output_dir, the RunSpec roster, epoch/batch/optimizer settings, sampler seed
  offset and normalization settings with the exact global values. Paths
  resolve relative to the config file; output must not overlap any input.
- `RunResult(run_id: str, status: Literal["success","failed"], checkpoint_dir: Path|None, selected_epoch: int|None, selected_validation_mse: float|None, failure_reason: str|None)`.
- Checkpoint JSON: schema_version, model_spec, normalizer_sha256, run_spec,
  selected_epoch, selection_metric, selected_validation_mse, dataset_version,
  dataset_manifest_sha256, train_sha256, training_config_sha256, source_sha,
  working_tree_dirty, runtime (python/torch/numpy/os/architecture/device),
  tensor_file and tensor_sha256. Bundle manifest hashes metadata, normalizer,
  and weights files. Validate each semantic relationship before constructing
  a model. Checkpoint path is an external input, not a config value to import.
- Evaluation JSONL: schema_version, mode, split, episode_id, scenario_id,
  controller_id, collection_seed, start_step_index, horizon_steps, predictor_id,
  training_seed (null for baselines), repeat, checkpoint_sha256 (null for
  baselines), status, predicted_state, expected_state, residual, failure_step,
  failure_reason. Success has finite predicted/residual fields and null
  failure fields; failure has null prediction/residual and a contextual reason.
  Unavailable windows are coverage rows, not pretend predictions.
- Evaluation summary: schema_version, protocol, provenance, checkpoint_roster,
  coverage, episode_outcomes, one_step_aggregates, rollout_aggregates. Manifest
  hashes every evidence file. No NaN/sentinel error metrics.

Membership is explicit for training, validation selection and evaluation:
all retains every row; history_common requires a preceding row in the same
episode; loss_common requires five future transitions from the start.
For rollout horizon H, intersect that membership with complete horizon-H
availability. Validation selection scores only the first one-step prediction
at these starts: all 800 validation rows for all, 798 for history_common,
792 for loss_common. Canonical training has 940 items; history_common 936;
loss_common 924. Both arms of each diagnostic share these counts and
normalization fitted from the original 940 train rows.

### Task 1 (#25): Data boundary and installed package

**Files:** Create `src/learned_dynamics/{__init__,types,data}.py`,
`tests/{test_data,test_types}.py`, `README.md`; modify root pyproject.toml
Hatch packages and pytest testpaths.

**Interfaces:** Consume `load_prediction_cases(dataset_dir: Path, expected_dataset_version: str) -> tuple[PredictionCase,...]`.
Produce `load_views(dataset_dir: Path, expected_dataset_version: str) -> DataViews`,
`training_arrays(view: TrainingView) -> tuple[np.ndarray,np.ndarray]`
(float64 [N,3] physical inputs, [N,2] physical deltas), and the shared types.

- [ ] **Red:** Write `test_views_preserve_scenario_membership_and_failures`
  asserting train/validation/test lengths 940/800/800 and both 70-row failures;
  `test_training_view_rejects_validation_and_test`; and
  `test_features_exclude_targets_and_raw_torque` using a small synthetic case
  with q=1, v=2, u=3, next=(1.02,2.1), expecting x=[1,2,3], delta=[.02,.1].
- [ ] **Run:** `uv run pytest -q experiments/04-learned-dynamics/tests/test_data.py`;
  expect missing package/API or explicit failing assertions before implementation.
- [ ] **Implement:** Register package/tests, strict types and data boundary;
  validate v1 through the existing loader, sort by episode/step and retain
  immutable provenance. Integrity validation may inspect every partition,
  but downstream fitting receives only TrainingView.
- [ ] **Green:** Run data/type tests and
  `uv run python -c "import learned_dynamics,forward_dynamics,state_transition_dataset"`.
  Corrupt manifest and split fixtures must fail; package import must succeed.
- [ ] **Commit:** Stage only these files; commit `feat: add checked learned-dynamics data views`.

### Task 2 (#25): Normalization and episode-local windows

**Files:** Create `src/learned_dynamics/{normalization,windows}.py`,
`tests/{test_normalization,test_windows}.py`.

**Interfaces:** Produce `fit_normalizer(view: TrainingView) -> Normalizer`,
`transform_inputs(values: np.ndarray, normalizer: Normalizer) -> np.ndarray`,
`inverse_delta(values: np.ndarray, normalizer: Normalizer) -> np.ndarray`,
`write_normalizer(normalizer: Normalizer, path: Path) -> None`,
`load_normalizer(path: Path) -> Normalizer`, and
`make_windows(view: SplitView|TrainingView, horizon: int, history_steps: int=0) -> tuple[tuple[Window,...],tuple[WindowCoverage,...]]`.

- [ ] **Red:** `test_train_only_statistics_and_round_trip` uses inputs
  [[0,2,3],[2,2,3]] to assert means [1,2,3], scales [1,1,1], constant flags
  [False,True,True], and exact transform/inverse behavior within 1e-12.
  `test_heldout_mutation_cannot_change_fitted_statistics` modifies valid
  held-out fixtures and expects identical serialized fitted parameters.
  `test_terminal_ending_window_allowed_but_crossing_rejected` and
  `test_history_removes_only_starts_without_previous_row` verify membership.
- [ ] **Run:** Run the two test modules; expect missing APIs or failing assertions.
- [ ] **Implement:** Use training float64 population statistics with the
  declared constant-feature rule. Accept only verified TrainingView in fit.
  Windows are contiguous, same-episode/context/split, can end at a boundary
  and never cross it; count missing-history versus missing-future separately.
- [ ] **Green:** Run both modules plus Task 1 tests; test invalid schema,
  infinite/negative scales, wrong feature order, booleans and repeated writes.
  Assert train memberships 940/936/924 and validation 800/798/792 for
  all/history_common/loss_common.
- [ ] **Commit:** `feat: add train-only scaling and boundary-aware windows`.

### Task 3 (#26): PyTorch model, device gate and predictor

**Files:** Create `src/learned_dynamics/{model,devices,predictor}.py`,
`tests/{test_model,test_devices,test_predictor}.py`; modify root
pyproject.toml and uv.lock for the exact torch pin.

**Interfaces:** Consume ModelSpec, Normalizer and dynamics types.
Produce `ResidualMLP(spec: ModelSpec)` as torch.nn.Module with
`forward(inputs: torch.Tensor) -> torch.Tensor`;
`require_device(name: str) -> torch.device`;
`LearnedPredictor(model: ResidualMLP, normalizer: Normalizer, context: DynamicsContext, predictor_id: str, device: str)`
with canonical `predict(state, applied_torque_nm, context) -> DynamicsState`
and `predict_batch(states: np.ndarray, actions: np.ndarray, context: DynamicsContext) -> np.ndarray`.
History has separate `predict_with_history(state, applied_torque_nm, previous_state, previous_applied_torque_nm, context) -> DynamicsState`; canonical predict rejects history models.

- [ ] **Red:** `test_zero_normalized_output_means_saved_mean_delta` uses
  zero weights and a nonzero saved delta mean; expected next state is current
  plus that mean, not blindly unchanged. `test_predict_a_b_a_is_stateless`,
  `test_wrong_context_rejected`, and `test_mps_unavailable_does_not_fallback`
  pin order independence, environment checksum and explicit device failures.
- [ ] **Run:** Run those modules; missing torch/model APIs are the expected
  initial failures. Resolve `torch==2.15.0` with uv without changing existing
  exact direct pins; if resolution fails, investigate the reported conflict
  before selecting a different pin. Record actual lock and installed version.
- [ ] **Implement:** Three Linear layers with tanh, float32 and explicit CPU
  initialization seed, then transfer. Preprocess/restore physical units on
  CPU using saved statistics; inference has no gradients. Validate all
  inputs/outputs and never clamp predictions. Explicit MPS selection requires
  availability and supported operations; environment CPU fallback is disabled.
- [ ] **Green:** Run CPU tests, ordinary installed imports, and a guarded MPS
  forward/backward+Adam smoke test using the approved operations. Record real
  availability; a skip proves no GPU result. Check equal batch/single outputs
  within 1e-6 rad and 1e-5 rad/s; no simulator symbol may be used by inference.
- [ ] **Commit:** `feat: add residual dynamics model with explicit CPU and MPS devices`.

### Task 4 (#26): Checkpoint integrity

**Files:** Create `src/learned_dynamics/checkpoint.py`,
`tests/test_checkpoint.py`.

**Interfaces:** `CheckpointBundle(model: ResidualMLP, normalizer: Normalizer, metadata: dict, bundle_sha256: str)`;
`save_checkpoint(output_dir: Path, model: ResidualMLP, normalizer: Normalizer, metadata: dict) -> Path`;
`load_checkpoint(input_dir: Path, device: str="cpu") -> CheckpointBundle`.
Metadata fields and hashes are defined in Shared contracts.

- [ ] **Red:** Write round-trip prediction, unknown/missing tensor names,
  wrong shape/dtype/context/seed, object-dtype NPZ and non-finite weight tests.
  In `test_repaired_outer_hash_does_not_hide_normalizer_mismatch`, change
  the normalizer's fitted train identity and repair file hashes; require rejection.
- [ ] **Run:** Run test_checkpoint.py; expect missing API or failing checks.
- [ ] **Implement:** Save CPU tensors under deterministic parameter names
  (`network.0/2/4.{weight,bias}`), strict finite JSON metadata and NPZ loaded
  with allow_pickle=False. Validate complete tensor/schema/normalizer links,
  not only file hashes. Refuse existing outputs or source overlap; stage,
  audit and publish. Resume requests are rejected in v1.
- [ ] **Green:** Run checkpoint/model tests. All corruption tests must fail
  safely before arbitrary model construction; accepted bundles reproduce
  predictions at the declared same-device tolerance.
- [ ] **Commit:** `feat: persist validated dynamics checkpoint bundles`.

### Task 5 (#27): Training configuration and evidence

**Files:** Create `src/learned_dynamics/{config,training_evidence}.py`,
`configs/training-v1.toml`, `tests/{test_config,test_training_evidence}.py`.

**Interfaces:** `load_training_config(path: Path) -> TrainingConfig`;
`write_run_manifest(output_dir: Path, config: TrainingConfig, results: tuple[RunResult,...], provenance: dict) -> Path`;
`load_run_manifest(input_dir: Path) -> dict`.
An accepted selection roster includes every planned variant/seed/device/
repeat, checkpoint hash, chosen epoch/metric, and explicit failed entries.

- [ ] **Red:** Assert config defaults equal all Global Constraints, file-
  relative paths work from a different cwd, bool seed/count is rejected,
  output cannot overlap dataset/config/source, and missing/duplicate seeds
  cannot pass as a complete run manifest.
- [ ] **Run:** Run both modules; expect missing APIs or failed contract checks.
- [ ] **Implement:** Strict TOML parsing and versioned manifests; store source
  SHA/dirty state, config/data/normalizer/environment hashes and actual command.
  No path expansion or arbitrary code loading. Distinguish exploratory dirty
  runs from clean published runs. Guard the 500 MiB budget and fresh outputs.
- [ ] **Green:** Test tampered selected epoch, best-seed-only roster,
  duplicate JSON keys, invalid hashes and unsupported schema.
- [ ] **Commit:** `feat: define reproducible training configurations and run manifests`.

### Task 6 (#27): Training, explicit histories and loss studies

**Files:** Create `src/learned_dynamics/{losses,training}.py`,
`scripts/train.py`, `tests/{test_losses,test_training,test_cli}.py`;
update README with actual setup and smoke commands.

**Interfaces:** `train_run(config: TrainingConfig, run: RunSpec, train: TrainingView, validation: SplitView, output_dir: Path) -> RunResult`;
`train(config_path: Path, output_dir: Path) -> Path`;
`one_step_loss(model, batch, normalizer) -> torch.Tensor`;
`rollout_loss(model, windows, normalizer, horizon: int=5) -> torch.Tensor`.
Losses use normalized physical residuals; inverse normalization in training
uses differentiable float32 device tensors so gradients pass through rollout.

- [ ] **Red:** Write short synthetic fixtures for falling training loss,
  earliest validation tie, fixed CPU seed/permutation repeatability,
  `test_training_api_cannot_access_test_view`, failed/non-finite loss retention,
  `test_rollout_loss_does_not_teacher_force`, and common-start/update-count tests.
  Both diagnostic arms must select checkpoints on identical validation
  starts; the first repeat (repeat=0) is the predeclared publication run,
  never whichever repeated run happens to score best.
- [ ] **Run:** Run losses/training/CLI modules; expect missing API assertions.
- [ ] **Implement:** Apply exact Adam/seed/epoch/batch policy, choose epoch
  using validation one-step normalized MSE, and persist curves including the
  smaller last batch. Inverse-transform predicted delta at each recursive
  step; future observed states are labels only. History shifts predicted
  states and recorded actions. Numerical failure writes a failed RunResult;
  never omit a seed or silently resume. Tests use explicit short configs;
  published config remains 300 epochs.
- [ ] **Green:** Run modules and
  `uv run python experiments/04-learned-dynamics/scripts/train.py --help`;
  run a short synthetic CPU training command and MPS smoke where available.
  Real training happens only after source/config is frozen in Task 9.
- [ ] **Commit:** `feat: train residual dynamics with validation-only checkpoint selection`.

### Task 7 (#28): Independent Stage 4 evaluation

**Files:** Create `src/learned_dynamics/{evaluation_config,evaluation,evidence}.py`,
`configs/evaluation-v1.toml`, `scripts/{run_evaluation,replay_checkpoints}.py`,
`tests/{test_evaluation,test_evidence,test_replay}.py`.

**Interfaces:** `evaluate(cases: tuple[PredictionCase,...], predictors: tuple, horizons: tuple[int,...], membership: str) -> tuple[dict,...]`;
`run_evaluation(config_path: Path, roster_path: Path, output_dir: Path) -> Path`;
`load_evidence(input_dir: Path) -> dict`;
`replay_checkpoints(input_dir: Path, device: str) -> dict`.
Use explicit history adapters where required and synthetic roster fixtures
until real #27 checkpoints exist. Baselines have null training seed and are
computed once; model identity is variant+device+seed+repeat+checkpoint hash.

- [ ] **Red:** Use a biased synthetic predictor to prove recursive drift
  without observed-state resets. A horizon-100 window in a 70-row episode
  is unavailable; numerical overflow is failed. Assert counts sum correctly.
  Tamper expected states/actions/seed identities/residuals and aggregates,
  repair outer hashes, and expect independent rejection.
- [ ] **Run:** Run evaluation/evidence/replay tests; expect missing APIs.
- [ ] **Implement:** Build Stage 4 v1 records and manifests from the shared
  case loader and pure residual/summary functions. Stage 3's evaluator
  aborts on prediction exceptions, so the Stage 4 orchestration must capture
  failures into its own schema; do not relax Stage 3's semantics.
  Recompute membership, expected states, counts, residuals and all group
  derivations independently. Separate checkpoint replay from saved-record
  audit; replay validates the model produced the predictions.
- [ ] **Green:** Run modules with simulator access blocked during audit;
  load the committed Stage 3 full and legacy outputs unchanged. Exercise
  both CLI helps and a complete synthetic fixture including a failed model.
- [ ] **Commit:** `feat: audit learned dynamics one-step and recursive evidence`.

### Task 8 (#29): Synchronized device benchmarking

**Files:** Create `src/learned_dynamics/benchmarking.py`,
`scripts/benchmark_devices.py`, `tests/test_benchmarking.py`.

**Interfaces:** `benchmark_inference(checkpoint_dir: Path, validation: SplitView, output_dir: Path) -> Path`;
`summarize_timings(samples: tuple[float,...]) -> dict`.
Training timing comes from each train_run; benchmark_inference loads the
canonical MPS seed-0 repeat-0 checkpoint on both devices.

- [ ] **Red:** Use a fake asynchronous device/clock to assert synchronization
  occurs immediately before and after timed GPU work, plus end-to-end
  versus model-only distinction. Verify warmups do not enter summary counts
  and MPS failure cannot become a mislabeled CPU timing.
- [ ] **Run:** Run test_benchmarking.py; expect missing functions or assertions.
- [ ] **Implement:** Validation rows in stable order: first row for batch 1,
  first 256 for batch 256. Use 20 warmups, 100 measured calls, three rounds
  ordered CPU/MPS, MPS/CPU, CPU/MPS; CPU threads 1/1 and float32.
  End-to-end inputs/outputs are physical CPU arrays including normalization
  and transfer; model-only inputs stay device-resident. Record raw timings,
  median, p95 using NumPy linear percentile interpolation, batch latency and
  items/s; disclose hardware/OS/build/power/competing load. No speed thresholds
  decide whether MPS remains primary.
- [ ] **Green:** Fake timing tests and CLI help pass; actual benchmark later
  records raw samples and outputs, with no unsupported-device claims.
- [ ] **Commit:** `feat: measure synchronized CPU and MPS dynamics workloads`.

### Task 9 (#29): Frozen study matrix and saved-record reports

**Files:** Create `src/learned_dynamics/{studies,report_tables,report_plots,report_markdown,reporting}.py`,
`configs/studies-v1.toml`, `scripts/{run_studies,generate_report}.py`,
`tests/{test_studies,test_reporting}.py`.

**Interfaces:** `run_studies(config_path: Path, output_dir: Path, phase: Literal["validation","final_test"]) -> Path`;
`generate_report(input_dir: Path, output_dir: Path) -> Path` returns report.md.

- [ ] **Red:** Assert this exact six-variant matrix: canonical32/all,
  capacity16/all, history0/history_common, history1/history_common,
  loss1/loss_common, loss5/loss_common. Every variant uses seeds 0..4 on MPS
  repeat 0. Canonical32 also has CPU repeats 0/1 and MPS repeat 1, yielding
  **45 total training runs** (30 base MPS + 5 repeat MPS + 10 CPU).
  Assert history pairs and loss pairs share membership and update budgets.
  Block final_test before a frozen validation selection roster exists.
- [ ] **Run:** Run study/report tests; expect missing APIs or failing matrix.
- [ ] **Implement:** Validation phase trains the roster and device diagnostics,
  records failures, and freezes all selections/config hashes before test.
  Final_test evaluates all six MPS repeat-0 variants across five seeds and
  six baselines once; CPU and repeat-1 runs remain validation diagnostics.
  Canonical all-row comparison includes every original case; diagnostic
  study arms additionally report common memberships. No adaptive test tuning.
  Generate one-step/rollout CSVs, failure/coverage CSV, learning curves,
  component-error/horizon figures and timing summaries from audited records.
  Aggregate seed means and sample standard deviations (ddof=1), retain
  per-seed/episode rows and failed/missing counts; missing seeds prevent
  complete-matrix claims. Explain known-physics access, joint-limit cases,
  correlations and dataset limitations. Report generation executes no
  training, inference or simulator calls.
- [ ] **Green:** Hand-computed metrics and seed statistics, zero-error plots,
  tampered aggregates, matching plot arrays and report runtime traps pass.
  Commit source/config before running real experiments; record clean SHA.
  Run the frozen validation phase, benchmark and final_test once; retain
  negative results, raw files and size accounting. If resource/storage gates
  fail, report the gap rather than quietly reducing the matrix.
- [ ] **Commit:** Source commit `feat: compare controlled learned dynamics studies`;
  publish approved-size generated evidence in a later artifact commit, after
  validation and independent reproduction. NPZ is globally ignored: choose
  a documented durable store or explicitly stage only audited checkpoint
  files with git add -f; never weaken the global ignore rule.

### Task 10 (#30): Independent reproduction and scientific closure

**Files:** Update experiment README, root README experiment index and
docs/roadmap.md; publish selected results with source/publication identities.

**Interfaces:** Consume frozen source commit, complete config/checkpoint
roster, raw evidence, saved report and declared tolerances. No new predictor.

- [ ] **Red/checklist:** Before reproduction, list all expected runs, bundles,
  files, schema/count/hash assertions and numeric tolerances. Any missing
  run/evidence, unavailable retrieval, or unresolved review finding means the
  completion gate is unsatisfied; this is a verification task, not a new test
  suite that mirrors the implementation.
- [ ] **Execute:** Create a separate clean checkout pinned to source SHA;
  use its isolated `uv sync --locked` environment. Run ordinary imports/CLI
  and `uv run pytest -q`. Retrain all 45 approved runs and replay published
  bundles; compare same-device inference/retraining tolerances, selected
  epochs and all declared counts. Report differences rather than adjusting
  tolerances. Compare report CSV and plot values; exact bytes only where
  deterministic. Retain environment and actual commands.
- [ ] **Audit:** Verify source SHA is reachable, clean source provenance,
  config/data/checkpoint hashes, every seed and failure, reports without
  simulator/inference/training, and unchanged earlier artifact hashes.
  Verify relative links and artifact retrieval. Perform formal Standards
  and Spec review and resolve findings before integration.
- [ ] **Close:** Mark Stage 4 complete only after every gate passes. Publish
  regenerated artifacts with reproducibility notes and preserve the source
  commit in merge history. Keep #31 as a downstream explanatory case study.
- [ ] **Commit:** `docs: document reproduced learned dynamics results`;
  provenance-preserving integration and local/remote agreement are required
  before claiming the work merged.

## Final self-review and execution boundary

This plan covers design sections 1–11 through Tasks 1–10. Its shared types,
normalization, checkpoint/roster schemas and matched study membership are the
contract between isolated implementations. Plan verification is document and
interface review; it is not evidence that dependencies or training work.

The immediate implementation slice is **#25 / Tasks 1–2**. #26 can follow
the shared-contract commit in parallel when explicitly dispatched. Each
task follows red/green verification and commits only its owned files; each
ticket receives review before integration. A later execution request
authorizes the selected slice, not automatically the whole stage.
