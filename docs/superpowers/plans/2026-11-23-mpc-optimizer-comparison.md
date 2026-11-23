# Matched MPC optimizer comparison implementation plan

**Goal:** Test whether spending the same number of candidate slots on an
adaptive search improves control in the one-link development task.

**Spec:** Issue #46 and the
[Experiment 05 design](../specs/2026-11-22-model-predictive-control-design.md).

## Comparison choices

- Compare the existing uniform random shooting and seeded cross-entropy
  method (CEM) at 16 candidate slots, horizon 30, and identical objective
  and feasibility checks. CEM uses 8 initial uniform candidates and 8
  follow-up candidates, fitted to up to 3 best feasible initial candidates.
  A 0.25 N·m standard-deviation floor avoids a zero-variance search.
- Use the two existing development initial states with three seeds each.
  Reset each controller on the same scenario and use separate generators
  with the same seed. Keep PD as a contextual reference.
- Run fixed-work first and deadline-discard at 10 ms separately. Compare
  optimizer outcomes on fixed-work records. Deadline mode measures how
  late decisions change control and fallback usage.
- Require the same candidate-slot budget at every MPC decision. Actual
  predictor calls can vary with early rejection. Report slots, infeasible
  candidates, actual calls, outcome and timing, retaining every episode.
- Decide which optimizer advances from fixed-work success first, then
  lower paired effort if both succeed. Publish failures and consider
  deadline mode when judging whether a configuration is practical. The
  chosen optimizer still needs a frozen predictor comparison.

## Implementation

1. Add `plan_cem` in `cem.py` using the existing planner rollout score in
   `planner.py` and returning `PlanResult`. Verify stable tie order,
   bounds, RNG replay, unequally sized batches, all-infeasible behavior and
   actual predictor calls. The scorer continues to propagate unexpected
   exceptions.
2. Add candidate and rejection counts to decisions, step records and
   metrics. Extend the pilot driver to include CEM only when configured,
   reusing journaling, fixed-work and deadline modes. Verify the normal
   paired path and raw accounting.
3. Run both development modes, repeat fixed-work, examine paired results
   and record the optimizer choice with the constraints and exact source.
