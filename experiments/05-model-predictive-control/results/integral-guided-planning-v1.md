# Integral-guided fixed and moving target comparisons

Replacing the learned planner's plain PD candidate with an integral
candidate raised fixed-target holds from 13/40 to 40/40. Bounded learned
corrections preserved those holds and gained speed. On moving targets,
both guided planners improved tracking over standalone integral feedback.
The physics planner had the lowest median error in each tested condition.

## Fixed-target candidate ablation

The [locked protocol](../../../docs/superpowers/plans/2026-11-23-integral-guided-planning.md)
used 40 fresh starts at 0.12 N·m. The two candidate ablation arms
used the same learned checkpoint, random proposals, cost, 16 by 30
budget and integral fallback. Only the guided candidate's feedback
rule differed in their planning search.

| Arm | Holds | Median settling (s) | Median torque effort (N²·m²·s) | Proposals used on time |
| --- | ---: | ---: | ---: | ---: |
| Standalone integral | 40/40 | 1.900 | 0.538 | 12,000/12,000 |
| Learned, one PD candidate | 13/40 | 1.330 | 2.014 | 11,635/12,000 |
| Learned, one integral candidate | 40/40 | 1.630 | 1.702 | 11,575/12,000 |
| Physics, bounded corrections | 40/40 | 1.865 | 0.544 | 11,946/12,000 |
| Learned, bounded corrections | 40/40 | 1.720 | 0.582 | 11,834/12,000 |

The single integral candidate added 27 matched holds with zero losses.
The bounded learned version settled 0.185 s sooner than standalone
integral in the median paired comparison over all 40 starts. It used
8.3% more median torque effort. Settling times in the table describe
successful episodes within each arm.

## Moving-target tracking

The 120 fresh oscillating targets covered 40 starts in each condition.
Both planners used the same integral baseline, 0.25 N·m maximum
adjustment, 2% predicted-cost threshold, 30-step horizon and 16 candidates.
Each preview began from the live accumulated torque with an independent
copy for each candidate. The real controller updated once per decision.
Both predictors used nominal inertia. The physics model received the
known friction bound. Late proposals were discarded after return.

| Condition | Integral median tracking RMSE (rad) | Physics planner | Learned planner | Learned proposals used on time |
| --- | ---: | ---: | ---: | ---: |
| Nominal | 0.10334 | 0.07070 | 0.07088 | 15,835/16,000 |
| Payload | 0.13232 | 0.09393 | 0.09755 | 15,973/16,000 |
| Payload and friction | 0.13352 | 0.09797 | 0.09916 | 15,956/16,000 |

In the primary payload-and-friction condition, the learned planner
improved error 25.7% versus standalone feedback and used 13.8% more
median torque effort. The physics planner improved error 26.6% versus
standalone feedback. Learned error was 1.2% higher than physics error
and its median effort was 3.2% higher. All episodes completed. The
learned planner cleared the effort and timing limits against physics
in the primary condition and fell short of the 10% tracking-error
improvement. The conditional confirmation was therefore skipped.

## Recorded-action forecast diagnostic

Both predictors replayed identical recorded integral-controller states
and applied torques. The diagnostic measured position at the end of
30-step forecast windows. Under combined payload and friction, learned
and analytical errors were 0.075 and 0.027 rad over 106 windows centered
on direction reversals. On 111 windows centered on high-speed target
motion, those errors were 0.202 and 0.020 rad. Both groups span 40 starts.
The larger learned error during fast motion under load identifies a
specific forecast regime for further model work. Replay windows within
one episode share its trajectory.

## Evidence

Source `fcef1e4f8956917818d67052f053f0573a4c5214` used the published checkpoint
and the prior locked integral gains. An independent audit checked 560
episodes, six raw hashes, the applied torque bounds and fallback, and
tracking errors computed from the scheduled reference at the end of each
step. All 120 focused tests passed.

| Artifact | Fixed SHA-256 | Moving SHA-256 |
| --- | --- | --- |
| Manifest | `b0a04d543ec830380b0e6ed7f19930559d262faa620f9e0196efe7ba266218b4` | `0d8cfb5ff000e9b8596b0f43155ac353df792bb14fc447f6f71b76b2fca8c937` |
| Episodes | `caa41a94629f02f06d903091465ea5f594a1ab4a0f610f31b2fa9d5d64344291` | `462870fdc072f876b444852d8040c3aa1f4e8e0fd2e13bd497c52378c685cd2c` |
| Results | `e0f87930502431c1e7f3475dad48b8f13e1248bcd761e2b9274183fa61c724ce` | `180c82f298bdd0ee191a8fd675550737acf00e67963408bb086a0d6753ecf437` |

The derived forecast replay SHA-256 is
`a2c9dd77f990e3a3ab2906a641b3a7a8a4d3f6c402625f90fac36c5a292f90e8`.

Run from the recorded source with a fresh output for each phase.

```sh
uv run --locked python experiments/05-model-predictive-control/scripts/run_integral_guided_comparison.py \
  --phase PHASE --learned-checkpoint CHECKPOINT --output FRESH_DIRECTORY
```
