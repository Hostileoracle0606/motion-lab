# Experiment 05 model predictive control implementation plan

**Goal:** Compare controllers and dynamics predictors under matched one-link
task conditions and explicit compute budgets.

**Architecture:** A pure planner uses the Experiment 03 predictor protocol.
One runner owns truth and applied actions. Metrics read immutable episode
records. The spec is [Experiment 05 design](../specs/2026-11-22-model-predictive-control-design.md).

**Environment:** Python 3.12, uv, NumPy, and MuJoCo from the repository lockfile.

## Global constraints

- Use `experiments/05-model-predictive-control/src/model_predictive_control`
  for source and `experiments/05-model-predictive-control/tests` for tests.
- Use the public records and constants in `contracts.py`. Reuse the existing
  predictor and PD equation. Keep earlier experiment contracts intact.
- Use original nominal MuJoCo truth, 2 ms physics, 10 ms controller period,
  torque within ±3 N·m, and hinge positions within ±1.570796 rad.
- Tests must exercise actual entry points. State evidence describes
  simulation, and timing evidence describes the measured host.

### Task 1: Random-shooting planner

**Files:** Create `planner.py` and `test_planner.py` in the source and test
directories. Own only those files.

- [ ] Write failing tests for a fixed candidate matrix with hand-scored
  candidates, recurrence, exact ties, out-of-range preview, and query counts.
- [ ] Implement immutable `CostWeights(position, velocity, effort,
  terminal_position)` with finite nonnegative values. Implement
  `score_candidates(predictor, context, state, target_position_rad, weights,
  candidates_nm) -> PlanResult`. Require a nonempty two-dimensional finite
  matrix with bounded torques. Use every row in order. At each step add
  `position * (next_q - target)**2 + velocity * next_v**2 +
  effort * torque**2`. Add `terminal_position * (last_q - target)**2` once.
  Reject only the affected candidate on a predictor `ValueError` or invalid
  next state/cost. Other unexpected exceptions propagate. Count the attempt
  before calling the predictor. Treat an out-of-range initial state or target
  as invalid input. Preserve the first feasible row on an exact tie.
- [ ] Implement `plan(predictor, context, state, target_position_rad, weights,
  *, horizon, candidate_count, rng) -> PlanResult`. Validate strictly positive
  integer dimensions. Draw a uniform candidate matrix from the supplied
  `numpy.random.Generator` in the context's bounded torque range. No extra
  candidates or access to future observations.
- [ ] Run focused tests. Confirm the infeasible result uses `None` for actions
  and cost and retains accounting. Commit and report evidence.

### Task 2: Truth plant, closed-loop runner, and PD

**Files:** Create `plant.py`, `closed_loop.py`, `test_plant.py`, and
`test_closed_loop.py`. Own only those files.

- [ ] Test observation/action order and fresh state using a fake plant and
  fake clock. Cover clamping, no-plan/invalid/deadline fallback, time-limit
  truncation and an early truth violation. Test one direct MuJoCo episode
  interval against the original checked XML.
- [ ] Implement `MujocoPlant(model_path: Path)`, `reset(state) ->
  DynamicsState`, and `advance(applied_torque_nm) -> PlantStep`. Use one
  persistent MjData per run, reset fully per episode, validate supported
  checksum and MuJoCo version using existing constants, then apply the held
  torque for up to five substeps. Inspect finite state and joint range after
  each substep. Report completed physics steps and reason; nonfinite state
  is represented as `next_state=None`. Enforce plant torque input as well as
  runner clipping.
- [ ] Implement `PDController(kp, kd)` using the existing PD function and
  return its raw torque through `ControlDecision`. Implement
  `run_episode(plant, scenario, controller, *, controller_id, seed,
  fallback, deadline_ns=None, clock_ns=time.perf_counter_ns) -> EpisodeRecord`.
  Use the returned reset state for the first request. One clock interval per
  primary decision, ending before fallback or plant advance. Deadline is
  exceeded only at elapsed > deadline_ns and takes precedence over no-plan
  or invalid-command. Use the same current state for fallback. A missing or
  nonfinite fallback is a configuration error. Keep finite primary requests
  in the record even when discarded, normalize invalid/missing requests to
  None, and count all query work from the primary and any fallback. Saturated
  means the command actually selected for application needed clipping.
  A truth termination takes precedence over a time limit on the same step.
- [ ] Verify real and fake plant tests and normal imports. Keep runner
  independent of planner and metrics. Commit and report evidence.

### Task 3: Episode and paired scenario metrics

**Files:** Create `metrics.py` and `test_metrics.py`. Own only those files.

- [ ] Test a hand-calculated trace, early termination, a zero-step record,
  nonfinite/absent terminal observation, quantile interpolation and repeated
  scenario with independent seeds.
- [ ] Implement immutable episode metrics and
  `summarize_episode(episode, *, position_tolerance_rad,
  velocity_tolerance_rad_s, hold_steps)`. Tracking RMSE in rad uses the
  initial observed state and finite next states, or None for zero steps.
  Effort in N²·m²·s integrates each applied torque squared using actual
  2 ms substeps. Peak directional overshoot in rad is relative to movement
  from initial position to target and never negative. For zero-length target
  motion, use peak absolute displacement from the target. Report None where
  required samples are absent. Success requires time-limit completion and
  the final `hold_steps` post-action samples all within both tolerances.
  Settling time is the timestamp of the first post-action sample in the
  uninterrupted final in-tolerance suffix when it meets the same success
  rule. Record final error, termination, counts of completed decisions,
  saturation, fallback, deadline misses and predictor calls. Compute p50 and
  p99 over recorded decisions using linear interpolation, in milliseconds.
- [ ] Provide `summarize_cohort(episodes, *, position_tolerance_rad,
  velocity_tolerance_rad_s, hold_steps)` for a single controller with
  unique scenario/seed keys, an episode count, successes and success rate,
  breakdown of terminations and an ordered collection of per-episode
  summaries with identifiers. Keep early terminations in the denominator.
  Document empty input behavior and do not combine controllers implicitly.
- [ ] Verify focused tests, commit and report evidence.

### Task 4: Connect and run the development pilot

- [ ] Integrate three foundation slices and add a versioned config,
  CLI entry point, run manifest, raw records, and metrics. Verify two
  deterministic paired runs with timing fields excluded from the
  equivalence check. Report decision latency as an observed quantity.

### Task 5: Compare optimizers with fixed planning budgets

- [ ] Add CEM behind the same result and scoring interface. Verify total
  candidate slots across iterations. Compare to shooting on development
  episodes before fixing the optimizer.

### Task 6: Freeze predictors, scenarios, and evaluation

- [ ] Pin the analytical, simulator, and learned predictors, then freeze the
  held-out suite and decision thresholds. Run and publish the frozen study,
  regenerate evidence independently, and document the measured result.
