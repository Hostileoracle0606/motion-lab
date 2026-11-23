# Frozen MPC predictor comparison plan

**Goal:** Isolate the effect of dynamics-model choice on the same one-link
control task. Lock the protocol before executing held-out cases.

**Spec:** Issue #47 and the
[Experiment 05 design](../specs/2026-11-22-model-predictive-control-design.md).

## Design decisions

- Keep seeded random shooting, the stage-46 objective, 16 sequences of 30
  actions, the 10 ms interval, and the independent PD fallback. Include
  PD alone as a task reference. The development results set a low
  expectation for settling under these settings, which the benchmark
  will be allowed to confirm.
- Pin the published canonical width-32, history-free, MPS-trained seed-0
  repeat-0 checkpoint by manifest digest and metadata. Use its archived
  validation-selected epoch. The rule identifies the canonical run before
  reading any Phase 5 held-out outcome. Run all predictors on CPU through
  the existing scalar `Predictor.predict` interface.
- Use nominal RK4, nominal MuJoCo and the learned model as frozen
  predictors. MuJoCo has the same simulator dynamics as nominal truth.
  When truth mass or hinge friction changes, all predictors still receive
  nominal model context and the same exact angle and angular velocity
  measurements. Keep truth XML digests and runtime flags separate.
- Lock four physical conditions, twenty paired base seeds and one
  deterministic generation seed. Use a disjoint two-seed smoke roster.
  Share the same generated starting state and target across plant shifts
  within each seed and across controllers. Use separate random streams
  for case generation and planning. Preserve the expanded identities in
  the preflight manifest before execution.
- Require within five percentage points of PD's hold rate per condition,
  zero nominal truth safety/numerical terminations, and p99 action-ready
  latency at or below 10 ms per episode in both fixed-work runs. To
  claim practical advantage, additionally require at least a 10% median
  paired reduction in settling time or applied-torque effort versus the
  relevant analytical or simulator comparator on at least ten jointly
  successful pairs. Report each criterion and paired denominator even if
  no configuration qualifies.
- Fixed work makes the outcome comparison independent of scheduling.
  A separate 10 ms deadline-discard mode records timing and fallback
  decisions. Use the #50 strict attempt audit before publication. Missing
  or interrupted attempts remain in the roster and invalidate the run as
  final evidence. Open-loop accuracy will be reported separately under
  #48.

## Implementation

1. Extend the truth plant with a checksummed, bounded nominal XML
   transformation for mass/friction shifts. Verify original compatibility,
   deterministic digests and distinct truth motion.
2. Compose the existing RK4, simulator and learned predictors through their
   scalar API and require the exact checkpoint metadata, device and bundle
   ID. Build the expanded roster and include source/config/model IDs and
   the #50 policy in the preflight manifest. Reuse the audited driver.
3. Verify the normal CLI with the development smoke roster, and generate
   the immutable held-out preflight manifest without advancing any held-out
   plant. Require that manifest for held-out execution.
