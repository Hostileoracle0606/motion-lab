# One hundred starts against friction-aware physics

At 0.12 N·m simulated dry friction, the learned planner held 32 of 100
targets. Standalone PD, guided nominal RK4, and guided friction-aware
physics held zero. The learned and friction-aware planners produced the
same counts with fixed work and with the 10 ms discard policy. In the
deadline comparison there were 32 paired wins and no paired losses.
The hold-rate lift was 32 percentage points with a paired bootstrap 95%
interval of 23 to 41 points. The learned hold rate had a Wilson 95%
interval of 23.7% to 41.7%. Zero of 100 puts the corresponding baseline
interval at approximately 0% to 3.7%.

## Matched benchmark

The [locked plan](../../../docs/superpowers/plans/2026-11-24-friction-aware-baseline.md)
generated new starts from seeds 74400–74499. The six arms shared each
start and target. All 600 episodes completed 300 decisions. Every planner
made 480 predictions per decision. The learned and dry-friction physics
arms ran batched on the same Mac CPU with the same cost, torque limits,
random candidates, and PD rule for the guided candidate. The friction-aware predictor
was given the simulator's friction torque and used ideal stick/slip
dynamics. On the separate 49-transition grid, its position and velocity
forecast errors were about 60% lower than nominal RK4's.

| Arm | Holds | Median decision (ms) | Action-ready P99 (ms) | Median final error (rad) | Median torque effort (N²·m²·s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Standalone PD | 0/100 | 0.002 | 0.005 | 0.06134 | 0.497 |
| Nominal RK4, fixed work | 0/100 | 5.779 | 9.838 | 0.06065 | 1.186 |
| Friction-aware, fixed work | 0/100 | 2.422 | 3.585 | 0.06056 | 1.235 |
| Learned, fixed work | 32/100 | 3.133 | 7.277 | 0.05968 | 1.612 |
| Friction-aware, deadline | 0/100 | 2.425 | 4.088 | 0.06056 | 1.235 |
| Learned, deadline | 32/100 | 3.134 | 4.837 | 0.05968 | 1.612 |

The deadline arms used 29,980 of 30,000 learned plans and 29,954 of
30,000 friction-aware plans on time. The others were discarded when
the planner returned, and PD supplied fallback after the cutoff. The
learned arm passed all predeclared checks. Its median torque effort
was about 31% higher than the friction-aware controller's in the
deadline comparison.

## What the result supports

The improvement occurred with the tested planner and scoring function at
one simulator friction setting. The learned model had higher one-step
forecast error than the friction-aware model on the preflight grid. That
grid and the control experiment measure different outcomes. The experiment
shows a control advantage and leaves its mechanism open.

A post hoc sensitivity check kept the 25-decision and velocity criteria
fixed and varied only the allowable target-position error. The primary
comparison remains the predeclared 0.05 rad column.

| Arm with deadline | 0.03 rad | 0.04 rad | 0.05 rad | 0.06 rad | 0.07 rad |
| --- | ---: | ---: | ---: | ---: | ---: |
| Friction-aware | 0/100 | 0/100 | 0/100 | 0/100 | 100/100 |
| Learned | 8/100 | 18/100 | 32/100 | 54/100 | 100/100 |

## Evidence

Source commit `a2d9261bdaaf3a7e0485ed2b45c14dec0b522ea5` used the published
checkpoint bundle `a30d771e770e9aabcb1d2f5fa7014bb9125d5487e0034dc901f4f7ded411960c`.
An independent audit recomputed holds, timing and paired results from the
ordered raw records and verified all four artifact hashes. The installed
predictor imported successfully, and all 399 tests across the affected
forward-dynamics, learned predictor, and control suites passed.

| File | SHA-256 |
| --- | --- |
| Manifest | `31aed28efc3e803ba4cf738196c8ee121b5892ecdb04ed58a698f64797df23a8` |
| Preflight | `0816b73172c9360a762def1031219cbdc4169de4f3b815eeeaec53b20c08335d` |
| Episodes | `c4c38ec788a0876c876ceeb218134bbd6372202d9ffd36afb1a9ba59ca96afff` |
| Results | `dd74fdf2057aaca2ed315ecade53d23422fccdeadce903396d43ade37ae76cb0` |
| Derived tolerance check | `e7a75824a481980f787e4eec6824cce7d4b4860670613838707bc460fba76a06` |

Run from a clean checkout of the recorded source with the published
checkpoint and a fresh output directory.

```sh
uv run --locked python experiments/05-model-predictive-control/scripts/run_friction_aware_comparison.py \
  --learned-checkpoint CHECKPOINT --output FRESH_DIRECTORY
```
