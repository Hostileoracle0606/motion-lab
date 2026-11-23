# Nominal fixed-work pilot

The PD baseline reached and held the target in two of two development
scenarios. Random-shooting RK4 MPC reached the terminal position tolerance in
both scenarios but met the velocity-and-position hold criterion in zero of
two. MPC's terminal joint speeds were 0.181 and 0.128 rad/s against a
0.10 rad/s tolerance. Each episode ran its complete 3 s simulated duration
with no truth-plant termination or fallback.

| Scenario | Controller | Tracking RMSE (rad) | Settling time (s) | Effort (N²·m²·s) | Predictor calls |
| --- | --- | ---: | ---: | ---: | ---: |
| Rest | PD | 0.451 | 1.76 | 0.517 | 0 |
| Rest | RK4 MPC | 0.448 | — | 8.886 | 144,000 |
| Moving | PD | 0.308 | 1.56 | 0.100 | 0 |
| Moving | RK4 MPC | 0.257 | — | 8.947 | 144,000 |

Position RMSE alone would favor MPC in both scenarios. MPC spent more effort
and never demonstrated a sustained hold. The result motivates comparing
optimizers at equal work while inspecting the predicted terminal velocity,
applied torque near the target and the first action from consecutive plans.
The pilot did not isolate the contribution of the objective, search
strategy, or horizon to the observed behavior.

Source: `73c29ad1374978ecbe9e86d46981a6e07fb6f931`. Run the command in the
[experiment README](../README.md) from that source with the committed
default config. The two fixed-work runs matched across all four raw
episode records after removing the primary and action-ready timing fields.
The matching plot preserves both unsuccessful trajectories. On the Apple
M4 development host with verification work running concurrently, MPC's
per-episode p99 decision latency ranged from 6.37 to 26.56 ms across the two
runs. The pilot therefore supplies host-specific timing observations.
Measure the timing gate in a controlled-load run.

The [evidence archive](pilot-v1/evidence.zip) contains the original
manifests, trajectory records with raw timing, metrics, plots, and SHA-256
index for both runs. The [paired plot](pilot-v1/paired.png) and
[first](pilot-v1/metrics-first.json) and [repeat](pilot-v1/metrics-repeat.json)
metrics are also directly viewable. These artifacts came from the exact
source above. Later driver changes add interrupted-run status and
completed-episode preservation without replacing the original measurements.

## Proposed gates before freezing the comparison

- Require success within five percentage points of the paired PD baseline
  per nominal or shifted condition, retaining all attempted episodes.
- Require zero truth joint-limit violations or numerical terminations on
  the nominal suite. Publish all such outcomes on the shifted suite.
- Require at least a 10% median paired improvement in settling time or
  applied-torque effort on successful pairs, with both metrics reported.
- Require p99 action-ready latency at or below the 10 ms control interval
  on the declared host and workload. Measure deadline misses and fallback
  separately in deadline-discard mode.

These proposed thresholds are development decisions. Lock them with
scenario counts, eligibility and timing repetitions before the held-out
comparison. Preserve pilot evidence if a gate changes.
