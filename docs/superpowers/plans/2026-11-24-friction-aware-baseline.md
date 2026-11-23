# Friction-aware baseline comparison plan

Test whether the published learned model retains its hold-rate advantage
against an analytical planner given the simulator's dry-friction torque.
Use 100 new independently generated starts at 0.12 N·m under the same
one-link dynamics, three-second episodes, and hold tolerances. Freeze
the roster, checkpoint, scoring, PD guidance, and 16 × 30 prediction
budget before recording any new case.

The analytical predictor integrates viscous damping and a bounded
dry-friction torque. At rest, friction cancels applied torque up to
its bound. During slip it opposes velocity, with an explicit stop
and restart if velocity reaches zero within a predicted step. Use
the known 0.12 N·m bound without fitting against benchmark outcomes.
The predictor approximates the simulator's dry-friction constraint.
Compare its one-step forecasts on a predeclared velocity and torque
grid against independent MuJoCo transitions before benchmarking.
Require lower position and velocity RMSE than nominal RK4 on that
grid. MuJoCo models dry friction as a bounded, soft constraint.
[MuJoCo dry-friction reference](https://mujoco.readthedocs.io/en/latest/computation.html#friction-loss)

Run standalone PD, guided nominal RK4, guided dry-friction physics,
and guided learned dynamics at fixed work. Run the friction-aware and
learned planners with the same 10 ms cutoff and PD fallback. Batch
both of those predictors across candidates. Rotate the six arms by
case. Retain every attempt and the actual model-work counts.

Primary outcome is the deadline paired difference in target-hold
rate between learned and friction-aware physics. Report Wilson 95%
intervals for each hold rate and a paired bootstrap interval for the
difference, resampling starts. Interpret a learned advantage as
practically useful when there are at least 10 net extra holds, the
interval for the lift excludes zero, the learned planner finishes
every episode, and at least 99% of its expected plans are used on time.
Report the fixed-work comparison separately and show the median
torque effort, terminal error, and timing for all arms. The scope is
one friction setting in one simulator.

- [x] Verify the analytical predictor's transitions and batch behavior.
- [ ] Record the preflight grid and the 100-case run from committed source.
- [ ] Audit raw episodes and compare matched-case performance.
