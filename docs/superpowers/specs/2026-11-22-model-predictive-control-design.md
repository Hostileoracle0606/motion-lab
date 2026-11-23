# Experiment 05 one-link model predictive control

Stage 4 evaluated predictions of recorded trajectories. Experiment 05 asks
whether a predictor helps choose torque commands that reach and hold a target
under action and computation constraints.

## Development pilot

Compare proportional-derivative control with random-shooting MPC using the
existing Experiment 03 analytical RK4 predictor. Both use the same fixed
target, initial state, simulation settings, and safety checks on each pair.
Use nominal Experiment 01 MuJoCo truth and the existing model checksum. Run
one persistent simulator instance per episode. Control uses a 10 ms interval
of five 2 ms physics steps and torque in [-3, 3] N·m. Observe the state,
request one action, enforce the torque limit, apply the command, then observe
again. The joint uses a linear limited-hinge coordinate in radians.

For each candidate sequence, recursively apply the predictor. For each
predicted successor add wq times squared position error, wv times squared
velocity, and wu times squared applied torque. Add a separate weighted
squared terminal position error on the last successor. Use the same objective
for shooting and CEM. Every proposed sequence consumes one candidate slot.
Count every attempted predictor call, including calls from rejected
sequences. Reject previews outside the declared joint range.

The runner inspects truth after each physics substep. It clips any finite
out-of-range request and records the requested and applied torque. Missing,
nonfinite, or late decisions use a declared PD fallback. A late successful
decision is discarded after it returns. The timing budget is a decision
deadline and does not interrupt synchronous Python execution. A zero
fallback or silent success is never inferred from a missing record.

Use fixed-work mode for reproducible trajectories. Use a separately
identified deadline mode for performance-sensitive decisions. Record
monotonic decision time, actual action interval, fallback reasons, truth
violations, and predictor calls. Timing evidence describes its measured
host. Record the predictor context and truth provenance separately.

## Frozen study

Choose gains, planning settings, and acceptance thresholds on development
episodes. Match horizon, objective, constraints and candidate-slot ceiling
when comparing optimizers. Match the optimizer when comparing predictors.
Select a fresh, independently simulated held-out scenario set and lock its
identities, seeds, conditions, and acceptance thresholds before final runs.
Pin one frozen learned checkpoint with its predeclared selection rule and
record all runtime devices. Preserve anomalous and failed episodes.

Use paired condition-level results for success, tracking, settling, effort,
saturation, constraint violations, fallback, and deadlines. Report actual
predictor calls and decision latency alongside nominal budgets. Maintain
the distinction between open-loop prediction error and closed-loop task
quality. The study scope is one-link simulation.
