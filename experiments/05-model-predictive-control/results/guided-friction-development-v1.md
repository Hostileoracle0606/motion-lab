# Guided planner friction development results

The learned planner held 5 of 12 targets at 0.12 N·m friction on the primary
roster and 6 of 12 on the separately generated confirmation roster.
Standalone PD and the guided nominal analytical planner held zero on both
friction rosters. The learned gain appeared both at fixed work and when
plans returning after 10 ms were discarded.

The [predeclared plan](../../../docs/superpowers/plans/2026-11-23-guided-friction-development.md)
used the same starts and 16 candidates × 30 predictions for both planners
with nominal predictor context, matched cost, and one PD-guided candidate.
The published checkpoint ran batched on the Mac CPU. RK4 used its scalar
predictor. Target hold required the last 25 decisions to be within 0.05 rad
of the target and at or below 0.10 rad/s in absolute velocity.

## Holds by condition

| Condition and mode | Roster | Standalone PD | Guided analytical | Guided learned |
| --- | --- | ---: | ---: | ---: |
| Nominal, fixed work | Primary | 12/12 | 12/12 | 12/12 |
| Nominal, fixed work | Confirmation | 12/12 | 12/12 | 12/12 |
| Friction, fixed work | Primary | 0/12 | 0/12 | 5/12 |
| Friction, fixed work | Confirmation | 0/12 | 0/12 | 6/12 |
| Friction, deadline discard | Primary | 0/12 | 0/12 | 5/12 |
| Friction, deadline discard | Confirmation | 0/12 | 0/12 | 6/12 |

PD is the same reference run in both friction comparisons. Each of the
four friction comparisons had five matched wins and zero losses in the
primary and six wins and zero losses in confirmation. All 192 episodes
completed 300 decisions, with zero abnormal simulator terminations. Both planners
made exactly 480 predictions per decision and had zero rejected candidates.
All predeclared acceptance checks passed in both rosters.

## Timing and effort under friction

| Roster | Learned plans used on time | Learned action-ready P99 (ms) | Analytical action-ready P99 (ms) |
| --- | ---: | ---: | ---: |
| Primary | 3,577/3,600 | 8.445 | 12.949 |
| Confirmation | 3,597/3,600 | 5.777 | 12.407 |

The deadline runner used PD after 23 late learned plans in the primary
roster and three in confirmation. It checks the time after the proposal
returns, so fallback can occur after the threshold. In fixed-work
friction runs, median final position error was 0.0545 and 0.0491 rad for
the learned planner, versus 0.0607 and 0.0606 rad for the guided analytical
planner. The learned median squared-torque effort was roughly four times
PD's in both rosters. This identifies a measured tradeoff against the
tested baselines at one friction setting in simulation. An analytical
model calibrated with friction would be a stronger comparator for
isolating the practical value of learning.

## Evidence

Both runs used source `3a3fd49693ed2855ea7b1d7abbfa248615760200` and checkpoint
bundle SHA-256 `a30d771e770e9aabcb1d2f5fa7014bb9125d5487e0034dc901f4f7ded411960c`.
An independent raw-record audit verified 96 episodes per roster, the
hold and deadline counts, and disjoint seeds and starts. All six output
checksums and 112 focused tests passed.

| SHA-256 file | Primary | Confirmation |
| --- | --- | --- |
| Manifest | `065447c461060669cec0f804cdaf41d87cdbc2a5d6e995877904f6babf8e8cab` | `ded585f9b419e7d6f6fc7e5c2b4136bcdb3faa8aef8d476530a38f4054d74c16` |
| Episodes | `7ef596917115a21e2f63f87b753a55dcc5467b1f9ebf45e5258cf9ac1492c658` | `1e02f9b80b6c3475fb7d12a4b59e87255f4fe45ab877b2300d908e88b44fbc13` |
| Results | `04acfb2cf6a17607453058c139a3fd06f4696f34a7a08111df1819753af1d11a` | `2f82c50adf8c56182a33bd41b112c3723d812ce941ae509ab2a8c95d92c3aa0a` |

Run the same command at the recorded source with `primary` or
`confirmation` as the phase and a fresh output directory for each.

```sh
uv run --locked python experiments/05-model-predictive-control/scripts/run_guided_friction_development.py \
  --learned-checkpoint CHECKPOINT --phase PHASE --output FRESH_DIRECTORY
```
