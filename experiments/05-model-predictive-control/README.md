# Experiment 05 closed-loop MPC development pilot

Can random-shooting MPC using analytical RK4 dynamics reach and hold a fixed
position on the same one-link MuJoCo plant as the established PD controller?
This pilot checks the control and evidence path on two development scenarios
before selecting settings for a frozen evaluation.

From a clean checkout of a commit containing the pilot, run:

```sh
uv run --locked python experiments/05-model-predictive-control/scripts/run_pilot.py \
  --output /tmp/motion-lab-mpc-pilot-v1
```

Use a fresh output directory. The default configuration is
`configs/pilot-v1.toml`. Both controllers share the target, initial states,
3 s simulated episode length, 10 ms action interval, and ±3 N·m actuator
limit. The PD baseline uses Experiment 01's documented gains without
retuning. The MPC settings are a single provisional choice without a sweep.
Each MPC decision proposes 16 sequences of 30 actions, for at most 480
predictor calls. Rejected sequences can reduce the actual work. MPC applies
the first action of the best feasible sequence and plans again from the next
observation. If it cannot produce an action, the runner applies PD and
records the fallback. MuJoCo automatic numerical reset is disabled so an
unstable trajectory terminates visibly.

The default fixed-work mode allows repeatable decisions irrespective of
process scheduling. The optional `--deadline-ns 10000000` runs a separate
development experiment with a 10 ms deadline on the primary controller.
It discards late decisions after they return and records the fallback.
The report includes primary decision latency and action-ready latency,
which includes fallback computation. Timing describes the run's host and load.

The output includes a source and environment manifest, raw episode and
transition records, computed metrics, and a plot of both controllers for
each scenario. Returned failure and truncation records are retained. The
driver writes each completed episode to `completed-episodes.jsonl` as it
finishes. For a software error or interrupt it keeps completed episodes,
records the aborted run, and propagates the error. The interrupted episode
has no completion record. Use `run-status.json` to verify that a run finished
before interpreting its metrics. Success
requires a normal time-limit episode that ends with 25 post-action samples
within 0.05 rad and 0.10 rad/s of the stationary target. Tracking RMSE
includes the first observation. Effort integrates squared applied torque
over simulated time. Failure episodes remain in the denominator.

The two cases are for development. Their tolerances and outcomes can guide
the held-out benchmark, where targets, tolerances, compute budgets and the
threshold for a useful improvement will be locked before evaluation.
The [pilot report](results/pilot-v1.md) records the observed outcomes and
the proposed gates for the next comparison.

## Matched optimizer development comparison

Run both optimizers with the analytical RK4 predictor using the same
configuration, candidate count, horizon, objective, and initial conditions:

```sh
uv run --locked python experiments/05-model-predictive-control/scripts/run_pilot.py \
  --config experiments/05-model-predictive-control/configs/optimizer-comparison-v1.toml \
  --output /tmp/motion-lab-mpc-optimizers
```

Use a second fresh output with `--deadline-ns 10000000` for the
deadline-discard comparison. The fixed-work mode measures behavior at a
matched search budget. The deadline-discard mode records how often an
overdue plan gives way to PD fallback.

Random shooting evaluates all 16 sequences sampled uniformly. CEM starts
with eight uniform sequences, then samples eight bounded sequences from a
distribution fit to the best feasible candidates from the initial batch.
Each optimizer begins with a separate generator seeded with the paired
case's seed. Both use 30 model steps per valid candidate. The raw evidence
records candidate slots, rejected sequences and actual predictor calls for
each decision. A rejected sequence can consume fewer than 30 calls.
See the [development comparison](results/optimizer-comparison-v1.md) for
the paired outcomes and optimizer decision.

## Attempt accounting and recovery

Before simulation, `planned-attempts.json` records the ordered scenario,
controller and seed roster. `started-attempts.jsonl` records each attempt
immediately before its episode begins; `completed-episodes.jsonl` contains
only episodes returned by the runner. An unexpected error discards transitions
from the current episode, preserves its started identity, and propagates the
original error. All planned attempts remain in the denominator; an interrupted
or unstarted attempt is not a successful episode or a completed trajectory.

Audit final evidence with:

```sh
uv run --locked python experiments/05-model-predictive-control/scripts/audit_attempts.py /path/to/run
```

The command exits unsuccessfully unless status, counts, ordered identities and
journals all establish completion. Always require a passing strict audit before
using result files: an abrupt stop while files are being published can leave
some files in an incomplete run. Add `--forensic` to inspect incomplete
runs, including interrupted and missing identities and validation errors. A
last journal row without a newline is omitted from forensic counts and flagged
as truncated, even if its JSON parses; it cannot support final evidence. Keep
the original folder for investigation and run again in a fresh output folder;
do not append to or silently repair a partial run. Completed records remain
available for inspection, and incomplete runs cannot be accepted as final evidence.

## Frozen predictor comparison protocol

The comparison uses one random-shooting optimizer with the same objective,
16 candidate slots, and 30-step horizon for analytical RK4, nominal MuJoCo,
and the published learned checkpoint. Standalone PD is the task reference
and the fallback. Each of 20 preregistered seeds provides the same starting
state and target for the nominal, heavier, friction, and combined conditions.
Every controller receives exact angle and angular velocity from its truth
episode. RK4, MuJoCo, and the learned model all receive the nominal predictor
context across conditions. Nominal MuJoCo shares the simulator and model
parameters of nominal truth. The shifted truth model's XML digest and
runtime reset policy are recorded separately, and the predictors receive
the same nominal information under shift.

Use the checkpoint bundle
`training-v1/canonical32-mps-seed0-repeat0/checkpoint` from the original
[Experiment 04 study release](https://github.com/Hostileoracle0606/motion-lab/releases/tag/experiment-04-study-v1).
Follow the [study README's archive verification instructions](../04-learned-dynamics/README.md#download-and-audit-the-published-study).
The comparison pins that canonical width-32, history-free run by bundle
hash and validation-selected epoch 75. All three predictors execute the
scalar inference interface on CPU. There is no optimizer, checkpoint, or
parameter tuning against the held-out cases.

Use a disjoint development-only roster to verify the normal entry point:

```sh
uv run --locked python experiments/05-model-predictive-control/scripts/run_predictor_comparison.py \
  --config experiments/05-model-predictive-control/configs/predictor-comparison-smoke-v1.toml \
  --checkpoint /path/to/canonical32-mps-seed0-repeat0/checkpoint \
  --output /tmp/motion-lab-predictor-smoke
```

The [frozen protocol](configs/predictor-comparison-v1.toml) declares two
fixed-work repetitions and a 10 ms deadline-discard run. Hold success,
truth safety, and 99th-percentile action-ready latency each have a separate
gate. A practical improvement also requires a paired reduction in settling
time or torque effort on at least ten jointly successful pairs. The
[comparison plan](../../docs/superpowers/plans/2026-11-23-mpc-frozen-predictor-protocol.md)
states the rule and scope. Use the strict [attempt audit](#attempt-accounting-and-recovery)
before evaluating any held-out output. An interrupted or missing attempt
prevents a complete-evidence claim.

To generate a new preflight for inspection, use the held-out config
with `--preflight` and a fresh JSON output path. This validates the
checkpoint and expands the roster without simulating any cases.
Actual held-out execution requires `--frozen-manifest` pointing to the
reviewed manifest and verifies its config, code objects, checkpoint and
case roster before any episode begins.

The frozen manifest lists the required runs `fixed-work-01`,
`fixed-work-02`, and `deadline-discard-01`. Use `--mode` and
`--repetition` to identify each run. Before computing the final comparison,
pass all three output folders to `scripts/audit_predictor_comparison.py`
with the same `--frozen-manifest`. The audit verifies that every declared
run completed exactly once. Issue #48 owns the per-condition and paired
report and the decisions against the locked gates.
The held-out protocol must cite this policy before the study starts.
