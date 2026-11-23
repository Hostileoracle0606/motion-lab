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
