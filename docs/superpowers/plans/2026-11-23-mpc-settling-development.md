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
