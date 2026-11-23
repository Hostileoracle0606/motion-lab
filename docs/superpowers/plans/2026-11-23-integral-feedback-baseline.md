# Integral feedback baseline plan

Inspect the previous 100-start traces to see when the learned planner
entered the target band and whether it kept moving or applying
target-directed torque above 0.12 N·m near the end. Compare successful
and unsuccessful runs. Treat these summaries as observational.

Add a bounded integral torque to the existing PD rule. Use the same
proportional/derivative gains, 10 ms update interval, and torque limit.
Integrate once per observed control step. Block accumulation when it
would drive an already limited request farther into saturation. Construct
a fresh integral controller for each episode.

Tune the integral gain and torque bound on 12 development starts crossed
with nominal friction, 0.09, 0.12, and 0.15 N·m. Reject settings with
abnormal terminations. Rank by the worst condition's hold count, total
holds, lower median squared-torque effort, then lower gain and bound.
Commit the selected settings and calibration hash before opening the
evaluation roster.

Use fresh, distinct starts for the evaluation at 0.10 N·m (40), the
original 0.12 N·m setting (100), and 0.16 N·m (40). Run standalone PD,
standalone integral feedback, and the existing batched learned planner.
Share starts within each condition, rotate arm order, and apply the
same 10 ms cutoff and PD fallback. Keep the learned checkpoint, PD
guidance, 16 × 30 predictions, episode length and hold tolerance fixed.
Report per-condition holds, paired wins and losses, timing and fallback,
settling and torque effort. The extra 40-start groups probe friction
shift. The 100-start anchor remains the primary comparison.

Prefer integral feedback on this benchmark if its observed hold rate
is at most five percentage points lower in every condition, its median
torque effort is no higher in each condition, it has no abnormal exits,
and it uses at least 99% of actions on time. This is a practical decision
rule. Report paired uncertainty alongside each result.

- [x] Inspect archived traces and verify integral update behavior.
- [ ] Tune on calibration starts and commit the selected settings.
- [ ] Run, audit, and report the fresh comparison.
