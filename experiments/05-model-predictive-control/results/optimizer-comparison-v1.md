# Matched MPC optimizer development comparison

Both optimizers failed the hold criterion in all six development cases.
CEM used less applied-torque effort in every fixed-work pair, with a
23.2% median relative reduction versus random shooting (range 6.9–34.6%).
PD held the target in all six cases.

The suite repeats two initial states with three RNG seeds each. The six
pairs therefore vary planner sampling and cover two physical starts.
All 18 episodes per mode ran for 300 control decisions without truth
joint-limit or numerical termination. Both optimizers used 16 sequences
of 30 prediction steps per decision with the same RK4 model, objective,
feasible set and plant. CEM used 8 uniform proposals and 8 adaptive
proposals. Refer to the
[versioned configuration](../configs/optimizer-comparison-v1.toml) for
the exact gains, weights, limits, and hold rule.

## Fixed work

| Controller | Holds | Median episode position RMSE (rad) | Median effort (N²·m²·s) | Candidate slots | Rejected candidates | Predictor calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PD | 6/6 | 0.379 | 0.309 | 0 | 0 | 0 |
| Random shooting RK4 | 0/6 | 0.359 | 8.565 | 28,800 | 0 | 864,000 |
| CEM RK4 | 0/6 | 0.364 | 6.492 | 28,800 | 0 | 864,000 |

The second fixed-work run matched the first on all 18 records after
excluding primary and action-ready timings. Across both runs, the
per-episode p99 action-ready latency ranged from 5.92 to 24.42 ms for
shooting and from 6.18 to 19.88 ms for CEM. These are host-specific
measurements from an Apple M4 with ordinary system load.

## Deadline discard

| Controller | Holds | Misses per decision | PD fallbacks | Per-episode p99 action-ready range (ms) |
| --- | ---: | ---: | ---: | ---: |
| PD | 6/6 | 0/1,800 | 0 | 0.004–0.008 |
| Random shooting RK4 | 0/6 | 0/1,800 | 0 | 6.06–8.03 |
| CEM RK4 | 0/6 | 1/1,800 | 1 | 6.17–6.91 |

The deadline is checked after the primary computation. This mode
measures discards and fallback behavior. A p99 below 10 ms can coexist
with an individual miss, as the CEM run shows. Candidate slots, rejects,
and attempted predictor calls were the same as in fixed-work mode.

## Decision

Keep seeded random shooting as the reference optimizer for the next
predictor comparison. The decision rule gives priority to successful
holds and compares effort on successful pairs. Neither optimizer
produced a successful pair under this objective and horizon. CEM's effort
reduction is a reason to keep it available for development, with further
closed-loop tuning evaluated before any final protocol lock. Record any
shared objective or horizon change using development cases before
running the held-out suite.

## Evidence and reproduction

Source commit: `f3f8aef327b6a65cf5ecc0de0388f7bab515050b`.
Use the [commands in the experiment README](../README.md#matched-optimizer-development-comparison)
and select fresh output directories. The bundled [fixed-work metrics](optimizer-comparison-v1/fixed-metrics.json),
[repeat metrics](optimizer-comparison-v1/repeat-metrics.json), and
[deadline metrics](optimizer-comparison-v1/deadline-metrics.json) preserve
individual cases and their measured timing. Each run has a machine-readable
source manifest. Full trajectories, the per-episode journal, plots and run
statuses are retained in the development evidence directory. The
[SHA-256 index](optimizer-comparison-v1/SHA256SUMS) binds all original
files, including the timing-bearing raw records.
