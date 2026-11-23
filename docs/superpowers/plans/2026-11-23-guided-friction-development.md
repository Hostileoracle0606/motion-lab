# Guided planner friction development plan

The primary question is whether the published learned model adds target
holds under higher friction compared with the same guided planner using
nominal analytical dynamics and with standalone PD. The earlier frozen
study used random proposals. This study uses the improved guided planner
on a new development roster.

Use 12 new starts crossed with nominal truth and 0.12 N·m friction. Match
the existing frozen study's ranges, target direction, 300 decisions, hold
definition, PD gains, costs, seeds between arms, and 16-by-30 candidate
budget. Use nominal dynamics in both predictors. Use batched CPU inference
for the published learned checkpoint and the existing scalar RK4 predictor
for the analytical arm. Fixed-work runs apply every feasible plan. Friction
also gets separate timing runs for both planners, with the same 10 ms
post-return discard rule and PD fallback. Rotate controller order by case.

The learned result warrants a confirmation run only if all these hold.

- Friction has at least three net additional paired holds over **each**
  comparator, both at fixed work and with the deadline discard.
- Nominal fixed-work holds meet or exceed both baselines.
- The learned arm records no truth failures in any condition.
- At least 99% of the 3,600 expected friction deadline decisions use a
  learned plan on time.

Record the separate win and loss counts, terminal error, tracking RMSE,
settling, effort, trial completion, actual predictions, and deadline
misses. Interpret settling time within the episodes that hold. If the
primary roster passes, run the same test with 12 disjoint confirmation
starts defined in the config. If it fails and both planners struggle,
inspect their forecasts from identical recorded PD states and applied
torques before suggesting model changes.

- [x] Write the source and predeclare the two disjoint sets of cases.
- [ ] Run the primary roster and audit the raw records.
- [ ] Apply the result-dependent diagnostic or confirmation and record the
  result with source, checkpoint, scope, and checksums.
