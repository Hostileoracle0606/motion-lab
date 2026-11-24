# Integral feedback versus learned planning

On the primary 100 fresh starts at 0.12 N·m friction, standalone integral
feedback held 100 targets, learned planning held 33, and plain PD held
zero. Integral feedback used about one-third of learned planning's median
torque effort and completed all 30,000 decisions on time in that condition.
It also held more targets at both additional friction levels.

## What the prior traces showed

In the previous 100-start fixed-work comparison, the 32 successful
learned runs first entered the target position and velocity band at a
median of 1.27 seconds. Their median absolute motion during the final
half-second was only 0.00018 rad. The median run in each group used no
target-directed torque over 0.12 N·m in that last half-second. Plain
PD and the friction-aware planner finished at median errors of 0.0613
and 0.0606 rad. Those offsets are consistent with 0.12 N·m of resistance
and the 2 N·m/rad proportional gain. The trace comparison describes
where the controllers differed in time and leaves the cause of the
learned planner's earlier success open.

## Calibration and fresh evaluation

The [locked plan](../../../docs/superpowers/plans/2026-11-23-integral-feedback-baseline.md)
kept the original PD gains. Integral feedback adds torque in proportion
to accumulated position error and caps the added torque. The selected
integral gain was 0.1 N·m/(rad·s), with a 0.12 N·m bound. It held all 48
calibration cases and won the predeclared selection rule over 15 settings.
The gain, bound, and calibration hash were committed before evaluation.

All three controllers shared starts within each condition. The learned
planner used the published checkpoint and 480 predicted states per
decision. The two feedback controllers used no model predictions. The
same 10 ms cutoff and plain PD fallback applied to all arms.

| Friction (N·m) | Plain PD holds | Integral holds | Learned holds | Integral paired lift, 95% interval (percentage points) |
| --- | ---: | ---: | ---: | --- |
| 0.10 | 0/40 | 40/40 | 21/40 | +47.5 (32.5 to 62.5) |
| 0.12 | 0/100 | 100/100 | 33/100 | +67 (58 to 76) |
| 0.16 | 0/40 | 35/40 | 4/40 | +77.5 (65 to 90) |

| Friction (N·m) | Integral / learned median torque effort (N²·m²·s) | Integral / learned action-ready P99 (ms) | Integral / learned plans or actions used on time |
| --- | ---: | ---: | ---: |
| 0.10 | 0.512 / 1.766 | 0.009 / 4.787 | 12,000 / 11,988 of 12,000 |
| 0.12 | 0.525 / 1.580 | 0.007 / 4.865 | 30,000 / 29,975 of 30,000 |
| 0.16 | 0.515 / 1.624 | 0.008 / 4.515 | 12,000 / 11,999 of 12,000 |

There were no abnormal simulator terminations. Integral feedback passed
the predeclared practical decision rule at every friction level. The
learned planner settled sooner on jointly successful starts. At
0.12 N·m there were 33 such pairs, and the integral controller took a
median 0.62 s longer. The measured preference for integral feedback
therefore depends on valuing reliable holds and lower effort over a
faster approach in the cases that both controllers solve.

## Evidence

Calibration source `58793fee4d231da5b2e0f09a73763878ea8c3c2b` and evaluation
source `9ea121f164e4302d6eb9e954f14b330e14788a68` bind the locked selection
and run commands. The evaluation used the same checkpoint and simulator
model at the 0.12 N·m anchor as the previous 100-start comparison.
An independent audit verified all 540 episode records, disjoint starts,
the feedback torques, timing and hold totals, the calibration lock, and
the three evaluation checksums. All 115 focused tests passed.

| File | SHA-256 |
| --- | --- |
| Calibration | `422f84d76ac3d7805f452adce8d84ec57e64234715d7ea2a37e964797dcc69d6` |
| Prior trace diagnostic | `5748c7b91526b4fcaf880666058af7132c6c4eb9082cd0a560820045eebf405e` |
| Evaluation manifest | `9939252fffee86de25fdc8488aeb997383b4d1476f8f671f52d52bbfac3b2d2b` |
| Evaluation episodes | `bc2a6a53150be102bf399aa10f06de502f169d4d18dc617b2cf04b6d27501816` |
| Evaluation results | `4d45c9e2ae3a5c678868f759d0bb742fee58c7aeddd7dc652b56b28742631d15` |

Run calibration from its recorded source. Use the resulting calibration
file and the published learned checkpoint for evaluation from its
recorded source. Both commands require fresh output directories.

```sh
uv run --locked python experiments/05-model-predictive-control/scripts/run_integral_feedback_comparison.py \
  calibrate --output TUNING_DIRECTORY
uv run --locked python experiments/05-model-predictive-control/scripts/run_integral_feedback_comparison.py \
  evaluate --calibration TUNING_DIRECTORY/calibration.json \
  --learned-checkpoint CHECKPOINT --output EVALUATION_DIRECTORY
```
