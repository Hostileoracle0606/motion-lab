# Batched learned planner implementation plan

**Goal** Compare end-to-end planning time and closed-loop control with the
published checkpoint on the existing settling development roster.

**Design** Roll the surviving candidates forward together at each horizon
step. The feedback candidate calculates torque from its own current
predicted state. The benchmark opts in to batching through the planner
entry point. Keep the checkpoint, seeds, scoring, and candidate budget
the same across fixed-work arms. If a batch raises a prediction error,
retry its lanes independently and count the extra attempted predictions.

- [x] Add failing planner tests for recurrence, independent lane rejection,
  and batch-error recovery. Implement and run the focused suite.
- [x] Record paired scalar and batch fixed-work runs on all 12 prior
  development scenarios. For the same 12 scenarios, enforce the 10 ms
  discard threshold in the batched arm and report fallback counts. Run
  standalone feedback as a reference and rotate the arm order by case.
- [x] Record elapsed p50 and p99, decisions over 10 ms, successful holds,
  actual prediction counts, and fixed-work action agreement. Include
  source, checkpoint, run order and artifact hashes.
- [x] Verify the ordinary command and review code and evidence before
  reporting the development result.

The candidate is useful on this roster if it preserves the fixed-work hold
count, and its deadline arm holds in all cases with at least 99% of plans
used on time. Report the actual misses and any work differences alongside
that rule. The deadline runner discards late proposals after they return.
