# Experiment 04: Learned Dynamics — Proposed Design

Status: **Approved for implementation planning**, issue #24. Approval was
given in chat on 2026-11-11 after the executive summary and decisions review.
Baseline: main commit
`84d28e227e6757c0128480fa16d0ad96ae04347a`. The user selected PyTorch
and asked to investigate the M4 GPU alongside CPU. The settings below form
the approved initial protocol; they are not measured performance results.

## 1. Question and scope

Can a small action-conditioned neural network predict a simulated joint's
next state and recorded-action rollouts, and where does it differ from simple
and known-physics baselines? For the same workload, when does M4 GPU training
or inference have an advantage over CPU in time, predictive quality, and
repeatability? A later planner may use these predictions, but no closed-loop
controller is built or assessed here.

Deliver a versioned data view, trained checkpoint bundles, a separate
auditable learned-model evaluation, controlled studies, and reproduction
instructions. Keep stage 3's published protocol and artifacts intact.
No physical-robot, broad generalization, safety, real-time, or world-model
control claim is supported by this experiment.

## 2. Known data and evidence limits

The validated `experiment-02-transition-v1` dataset has 2,540 transitions:
train 940 (four episodes, two wide scenarios), validation 800 (two episodes,
negative-to-positive narrow scenario), test 800 (two episodes,
positive-to-negative narrow scenario). The two P-controller training episodes
terminate at joint-limit violations after 70 steps; the other six last 400
steps. Collection has one seed and two related feedback policies. This is
scenario holdout, not a representative independent sample. A training-only
inspection found input ranges q [-1.5211, 1.5211] rad, velocity
[-5.6137, 5.6137] rad/s, and applied torque [-3, 3] N m; 76 of 940 actions
are exactly at the torque limit. These marginal ranges do not demonstrate
coverage of the joint state-action space.

Training and validation guide design and checkpoint selection. Fix the test
matrix before running learned test evaluation; do not revise the model or
protocol based on its outcomes. Retain terminal rows and failed episodes.
Independent integrity validation may read all partition bytes; only train
rows enter fitting, and only validation rows enter model selection.

## 3. Approaches and decisions

- **Residual MLP (selected):** learn delta state and add it to the current
  state. At 0.010 s per step, the observed changes are small relative to
  typical state magnitudes. This gives a useful identity starting point but
  does not guarantee better accuracy.
- **Direct MLP:** predict the next state itself; simpler output semantics but
  must learn the identity mapping as well. It is a reasonable future ablation;
  it is not in the initial capacity/history/loss study matrix.
- **Known-parameter physics or simple baseline only:** valuable comparators
  already exist, but cannot answer the learned-model question.

Use PyTorch as requested. MPS is the primary training device; CPU is a
supported comparison and diagnostic device. A PyPI metadata check on
2026-11-10 found `torch==2.15.0` has a CPython 3.12 macOS arm64 wheel.
This is a **candidate exact pin**, not a verified lock resolution or working
MPS installation. Implementation must resolve the lock without changing
existing exact direct pins, then verify imports, MPS availability, required
forward/backward operations, and a CPU/MPS smoke run. If this fails, report
the concrete incompatibility and resolve the pin before training; never
silently fall back or change existing direct dependencies.
[PyTorch distribution](https://pypi.org/project/torch/) and
[MPS backend](https://docs.pytorch.org/docs/stable/notes/mps.html).

## 4. State, preprocessing and model contract

The model consumes x = [position rad, velocity rad/s, applied torque N m]
and predicts delta = [next position - position, next velocity - velocity].
The action was applied over one 0.010 s interval, consisting of five
0.002 s MuJoCo steps. State is observed before the action. The canonical
`Predictor` is stateless and only accepts `DynamicsState`, applied torque,
and `DynamicsContext`; validate environment ID, XML SHA-256, interval,
torque limit, finite values and input types. No target, controller identity,
or future observation is input. Do not clip states or wrap this limited
hinge's angles; a finite predicted state outside its nominal limit is evidence.

Fit separate per-feature means and population standard deviations for x and
delta from all 940 training transitions in float64, in fixed episode/step
order. For a standard deviation below 1e-12 in its own physical units use
scale 1.0 and record a constant-feature flag. Transform with (value-mean)/
scale, convert input to float32, run the network in float32, inverse-transform
delta with the saved parameters, then add the original physical state.
Outputs return position rad and velocity rad/s. Persist means, scales, flags,
feature ordering, unit, algorithm version, train-row count, dataset identity,
manifest and train-partition hashes. No validation/test statistics are fit.

Canonical model: 3 -> 32 -> 32 -> 2 fully connected layers, tanh activations,
linear output; default PyTorch Linear initialization after seeding the
PyTorch CPU global generator with the run seed, before device transfer. Set
model to evaluation mode for inference. The input and output shapes are [batch,3] and [batch,2]; single
state adapter wraps one row. No dropout or batch normalization.

## 5. Training and selection

Declared roster: initialization/data-order seeds [0,1,2,3,4]; these are
training seeds, not additional data collection seeds. Every variant uses all
five seeds. Use Adam (learning rate 0.001, betas 0.9/0.999, epsilon 1e-8,
weight decay 0), batch size 64, at most 300 epochs, no learning-rate scheduler,
and no early stopping. One epoch iterates a seeded random permutation of all
eligible training items, once each, including a smaller final batch.
Precompute per-epoch permutations using a separate CPU generator seeded
with 100000 plus the run seed. CPU and MPS use the same initialization and
order for a given seed.
Use one-step mean squared error averaged over batch and two normalized delta
coordinates. Log validation after every epoch and save the checkpoint with
lowest validation mean normalized one-step MSE; earliest epoch wins exact
ties. Never select a best seed. Runs that diverge remain failed entries.

The same frozen design and equal epoch count apply to every seed and device.
Retain the entire per-epoch training/validation curve, chosen epoch and
checkpoint, timing and failure reason. No test examples reach training or
checkpoint selection. Validating whole-dataset integrity is a separate step.

## 6. Controlled studies

One-factor comparisons each use the canonical residual model above and the
same seeds, optimizer, epochs, normalization, and evaluation protocol:

1. **Capacity:** hidden widths 16/16 vs 32/32. Report parameter counts.
2. **History:** history-free versus one extra previous (state, applied action)
   pair, giving inputs [q_t,v_t,u_t,q_(t-1),v_(t-1),u_(t-1)] and 6->32->32->2.
   Previous values use training-only feature normalization. Exclude episode
   step 0 from both arms of this diagnostic, and compare on the same starts.
   History is an explicit sequence input, never mutable hidden state in
   `Predictor`. At rollout start use only the actual immediately preceding
   state/action; afterward shift in predicted states and applied recorded
   actions, never future measured states.
3. **Loss:** one-step loss vs a five-step recursive training loss on common
   length-five episode-local windows; use the same number of optimizer updates
   and batch windows for both arms, defined by one epoch as one pass over
   those common starts. One-step arm scores the first predicted delta; rollout
   arm averages normalized physical state errors at steps 1..5, using the
   training delta scales for the two component denominators. A window ending
   on a terminal row is eligible; it must never cross a boundary. History and
   loss studies are diagnostics in addition to the canonical all-row model.

Report eligibility, excluded starts, update counts, training wall time,
and failed runs. Equal updates do not imply equal FLOPs; report compute cost.
All hyperparameters and matrix entries are frozen before test inspection.

## 7. Evaluation and failure semantics

Implement a Stage 4-specific, versioned configuration/evidence format rather
than relaxing Experiment 03 v1's fixed six-adapter set. Reuse its validated
dataset case loader, state/context types, residual function and summary
semantics. Retain all six baseline identities: MuJoCo, no-change,
constant-velocity and RK4 at 0.002, 0.001 and 0.0005 s.

On matching cases use horizons [1,5,10,25,50,100] control steps. Compute
position and velocity MAE, RMSE, and maximum absolute error separately in rad
and rad/s; include per-episode, scenario, controller and split identities,
row/window counts, and training-seed variation. Raw signed residual is
predicted minus observed, without angle wrapping. One-step starts at each
observed state; multi-step starts at the observed initial state and recursively
feeds its own predictions forward under recorded applied actions. Keep
overlapping windows and label their dependence.

Record each eligible (split, episode, start, horizon, predictor, seed) with
status success or failure. Missing future rows are unavailable, not failed.
Numerical failure records the first failure step/reason; do not manufacture
an endpoint, omit the case silently, or average a sentinel. Report eligible,
success, failed, and unavailable counts and metrics on successes alongside
explicit denominator; no success-only aggregate may be interpreted without
the failure rate. Compare both arms of history/loss diagnostics on their
predeclared common eligible sets. Retain all six original baseline metrics,
and identify MuJoCo as a consistency reference with privileged simulator
access; analytical parameters are known and omit soft-limit/contact forces.

## 8. Device evaluation protocol

Predeclared question: **For this action-conditioned model and workload, when
does the M4 GPU differ from CPU in training time, single/batched inference
latency, predictive quality, and repeatability?**

Use the same CPU-initialized weights, precomputed permutations, float32,
training configuration, and five seeds for each device. Choose validation
checkpoints by the same rule, report validation metrics per device/seed, and
do not choose the best seed. Train each device/seed independently twice in
alternating device order; record per-run total training time including
validation, but excluding dependency setup and data-integrity loading.
The repeated training run is a diagnostic, not an additional random seed.
For each run retain hardware/OS, Python, PyTorch, device, dtype, thread count,
power state, and actual timing samples. Pin CPU intra-op and inter-op threads
to 1; document power mode and competing load rather than pretending to
control them.

Inference benchmark uses the canonical seed-0 selected checkpoint copied to
each device, identical validation input batches of sizes 1 and 256, 20 warm-up
calls and 100 timed calls per size/device, three alternating-order rounds.
Time the complete public prediction path from CPU-resident physical inputs to
CPU-resident physical outputs, including preprocessing and transfers; report
median and p95 per batch and per-item throughput. Also separately time model
forward-only with already-resident normalized tensors. Explicitly synchronize
MPS before and after every timed region. No empty timing or hidden fallback.
Keep raw timings and environment; no speed claim until measured. The timing
benchmark uses validation, not test, so the reported test roster stays frozen.

MPS is the user-selected primary training device, irrespective of whether it
is faster in this small workload. Validate that required operations run,
validation performance is finite, and repeated same-device predictions meet
the tolerance in section 10. Report both devices regardless of the winner.
If MPS is unsupported or fails those gates, record why and obtain approval
for changing the primary device before publication. Do not claim GPU speed
or quality advantages without measured evidence.
Final test matrix uses all five MPS seeds' repeat-0 checkpoints per declared
study variant once, and compares all declared baselines under the frozen
protocol. The implementation plan enumerates six variants (canonical,
capacity, two matched-history arms and two matched-loss arms) and 45 total
training runs, including canonical CPU/MPS repeatability diagnostics.

## 9. Checkpoints, audit and ownership

Use JSON metadata plus NPZ tensors loaded with `allow_pickle=False`; never
deserialize untrusted pickle. Each bundle validates tensor names, shapes,
dtype, finiteness, exact schema fields, normalizer hash, architecture,
dataset/config hashes, source SHA and clean/dirty state, runtime/device and
seed. Record file hashes in a manifest. This format is for inference; resume
training is unsupported in v1. Use fresh staged output directories; reject
overlap with source inputs and existing output destinations. Set an initial
500 MiB total artifact budget; if exceeded, stop and propose a checksummed,
durable storage alternative before publishing.

Persist selected checkpoints, full seed/run manifest and learning curves,
one-step/rollout raw records, timings, failure/coverage rows, and derived
tables/plots. The independent evidence loader validates provenance, case and
window identities, actions, expected states, checksums, residuals, grouping
and counts against original dataset rows. Checkpoint replay is a separate
explicit verification: audit of saved derivations alone does not prove the
model produced them. Report generation consumes audited saved records and
must not call the simulator, run inference or train.

New code lives under `experiments/04-learned-dynamics/{configs,src,tests,
scripts,results}`. #25 owns data/normalization; #26 model/checkpoints and
the PyTorch pin; #27 training and selection; #28 learned evidence/evaluation;
#29 studies, device benchmark and reporting; #30 independent reproduction;
#31 explanatory case study. Shared Experiment 03 changes require regression
proof, and its v1 loader, config and artifacts remain valid.

## 10. Reproducibility and tests

Target macOS arm64, Python 3.12 and pinned dependency lock. Record exact OS,
chip, PyTorch build, CPU threads, hardware, source commit, data and config
hashes, device and seeds. Seed all relevant generators; request deterministic
algorithms where supported and record unsupported operations. Do not promise
cross-platform or CPU/GPU bit equality. For repeated same-device checkpoint
inference, require per-output absolute tolerance 1e-6 rad and 1e-5 rad/s.
For independent same-device retraining, compare validation per-component MAE
within 1e-4 rad and 1e-3 rad/s, and report checkpoint-selection differences;
failed tolerance is a failed reproduction gate, not permission to relax it
after seeing test results. These are predeclared thresholds, not
measured reproducibility. Retain timing distributions without a numerical
timing-equality requirement.

Test train-only fitting and held-out mutation; normalizer round trips and
constant features; schema/tensor/checksum corruption; state/action validation;
synthetic known model outputs; A/B/A statelessness; correct boundary windows;
same-start history and loss studies; no teacher forcing; MPS unavailable and
operation errors; model divergence and failure denominators; metric and
aggregate tampering even after outer hashes are repaired; report generation
with simulator/inference/training blocked; and existing Stage 3 artifact
loading. Include short synthetic CPU tests and gated MPS smoke checks; actual
experiments remain scripted separately. Check ordinary package imports and
CLI entry points outside pytest.

Independent reproduction uses a separate clean checkout pinned to the source
commit, recreates its environment, retrains the approved roster and replays
published checkpoints, audits derived evidence and compares plotted numeric
data. Publish source/config first, then artifact commit; preserve the source
commit in merged history. Mark Stage 4 complete only after these gates pass.

## 11. Roadmap gate and ticket mapping

| Gate | Evidence and ownership |
| --- | --- |
| Scenario-safe partitions, train-only normalization | validated v1 dataset and #25 |
| Reproducible training, config, checkpoint provenance | #26 and #27 |
| One-step/recursive comparison, recorded failures and counts | #28 |
| Capacity, history, loss ablations; seed variation; held-out and boundary analysis | #29 |
| Independent reproduction and closure | #30 |
| Reader-facing explanation grounded in results | #31 |

The existing held-out direction/displacement scenarios provide only a narrow,
declared scenario shift. If they are insufficient to study the proposed
failure modes, document the gap and propose separately versioned data; do not
silently call another evaluation out-of-distribution or overwrite v1.
