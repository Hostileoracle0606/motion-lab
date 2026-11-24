# Integral-guided planning comparison

Use the previously locked integral controller as the baseline. Advance
its real accumulated error once per observed step. Each simulated
candidate starts from its own copy of the accumulated torque. The
learned checkpoint and the 16-candidate, 30-step budget stay fixed.

## Fixed-target ablation

On 40 new starts at 0.12 N·m, compare standalone integral feedback,
the original learned planner with a PD candidate, the same learned
planner with one integral-feedback candidate substituted, and both
integral-guided residual planners. This tests the candidate substitution
under identical cost and random sampling. Compare paired holds, settling
on jointly successful starts, torque effort, and timing.

## Moving-target comparison

Use a smooth oscillating position and velocity reference over four
seconds. The shared reference schedule is available to every arm.
Compare standalone integral feedback, a batched friction-aware
analytical planner, and the same planner with learned predictions.
Candidate zero follows integral feedback throughout the horizon.
The other 15 candidates add bounded torque adjustments, held for
five simulated decisions at a time. Roll out candidate-local
integral state and select a correction only for at least a 2%
predicted cost reduction. Use the real integral action if there
is no comparable plan or a late result.

Use 40 new starts per condition for nominal dynamics, added payload,
and added payload with friction. Both predictors use nominal inertia.
The analytical predictor receives the known friction bound. Primary
comparison is the combined condition. Require at least 10% lower
median tracking RMSE than the strongest baseline, at most 5% greater
median squared-torque effort than that baseline, at least 99% of
proposals used within the 10 ms cutoff, and a complete learned roster.
Verify any gain on a separately generated 40-start combined roster.
If the learned result falls short, compare forecasts over recorded
actions near reversals to guide a targeted next modeling change.

- [x] Verify preview state isolation and the correction guard.
- [x] Lock the experiment and run the fixed and moving comparisons.
- [x] Audit records and document the outcome.
