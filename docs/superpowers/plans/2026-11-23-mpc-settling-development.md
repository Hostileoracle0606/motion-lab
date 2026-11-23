# MPC settling development experiment

The frozen comparison showed that predictions could end near the
target while the physical arm failed to hold it. Test whether making
the optimizer penalize predicted speed at the end of its lookahead
improves sustained holds.

- Use the analytical predictor, random shooting, the existing PD baseline,
  16 proposals with a 30-step horizon, and the existing 3 s episode and
  25-sample hold criterion. Use fixed work for the task comparison and
  report timing as context.
- Compare terminal-velocity weights 0, 1, 4 and 16 on the six existing
  optimizer development cases. Select one nonzero weight by hold count,
  then by median consecutive in-band samples at the end. If tied, choose
  the smaller weight. Test the selected weight once against zero and PD
  on six new development starts with nominal and mild-friction truth.
  Keep original controls, shared seeds, and the action sequence sampler
  matched. Rejection counts and attempted prediction work are outcomes.
- Count every episode. This first pass supports a useful improvement if
  the selected variant gains at least two of the 12 holds and introduces
  zero additional truth failures. Report the paired outcomes per
  condition, the best eligible setting even if none passes, and the
  commanded effort and latency alongside holds.
- Keep the original Phase 5 protocol and evidence frozen. This follows
  the owner's decision to defer the clean-checkout reproduction and
  prioritize development. The results support development decisions.

## Follow-up after the terminal speed experiment

The terminal speed penalty gained zero holds on the paired development
confirmation. Test proposal quality with the original objective. Draw the
same 16 random sequences and replace the last with one rollout that applies
the existing PD equation to each successive predicted state. Evaluate it
in the same scoring loop. This reserves one slot for a plausible feedback
trajectory and holds the prediction ceiling at 480 calls per decision.
The planner selects a first action from the lowest-cost feasible trajectory.
Standalone PD remains the comparator.

Use the six separate `feedback_confirmation` starts across nominal and
mild-friction truth. Compare uniform shooting, shooting with one feedback
proposal, and standalone PD. Use the same +2-hold and no new truth failure
criterion. Record the tracking, effort, and per-decision timing tradeoffs.
