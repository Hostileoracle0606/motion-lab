# Batched learned planning development results

Batching reduced median learned planning time from 31.679 to 3.139 ms on
an Apple M4 CPU. Both fixed-work arms applied exactly the same torques
and produced the same trajectories. The batched deadline arm used 3,599
of 3,600 plans on time and held the target in all 12 cases. One late
plan was discarded in favor of PD feedback.

## Method

The roster reuses the six starts crossed with nominal and mild-friction
truth from the [settling development probe](settling-development-v1.md).
Each episode ran for 300 decisions. The learned arms used the same
published checkpoint, prediction context, random seeds, cost, 30-step
horizon, and 16 candidates, including the feedback proposal. The scalar
arm advanced one candidate at a time. The batched arm advanced all
candidates together by one predicted step, then used those predicted
states as inputs for the next step. Both evaluated exactly 480 predicted
states per decision. Each arm ran first in three cases.

## Results

| Arm | Holds | Median decision (ms) | P99 decision (ms) | Decisions over 10 ms | Median settling (s) | Median torque effort (N²·m²·s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Scalar, fixed work | 12/12 | 31.679 | 58.387 | 3,600/3,600 | 1.450 | 1.500 |
| Batched, fixed work | 12/12 | 3.139 | 8.363 | 17/3,600 | 1.450 | 1.500 |
| Batched, deadline discard | 12/12 | 3.128 | 5.839 | 1/3,600 | 1.450 | 1.500 |
| Standalone PD | 12/12 | 0.002 | 0.010 | 0/3,600 | 1.975 | 0.484 |

All episodes reached the time limit with no truth failure. The audit
found exact agreement across every untimed step field in the fixed-work
arms. The scalar arm also matched the previous learned probe's trajectories.
Each learned arm evaluated 1,728,000 predicted states with zero rejected
candidates. The deadline run exceeded the predeclared 99% timely-plan
gate. The single late proposal returned after 10.104 ms, after which
the runner computed fallback. This runner checks elapsed time after the
proposal returns, so it can apply fallback after the threshold.

The two batched timing runs had 17 and one decision over 10 ms under the
same candidate budget. These timings characterize the observed runs on
one Mac. Compared with standalone PD on this roster, batched learned
planning cut about half a second from median settling time and used about
three times the torque effort for the same hold count. All cases were
simulated development cases from the prior probe.

## Recorded source and artifacts

- Source commit `56348a2ddfb9a7093e858fea07489ab939a05d16`
- Checkpoint bundle SHA-256 `a30d771e770e9aabcb1d2f5fa7014bb9125d5487e0034dc901f4f7ded411960c`
- Manifest SHA-256 `cf3cca59afce2bbbd5da9ebe36f748dac8c608a1863ba40da7c12b14ec88df65`
- Episode records SHA-256 `0dd67b3a1d04e26d4eaecc85b50d939b1deb1616bdaf3892d394efe850a4e8e4`
- Results SHA-256 `f3b5406e3ad21fb05c41456b8bc727e128897ab12d61e7d7a25a10ec2aba6269`

From a clean checkout of the recorded source, use the published checkpoint
described in the [predictor protocol](../configs/predictor-comparison-v1.md).
The command writes the manifest, 48 episode records, results, and hashes
to a fresh output directory.

```sh
uv run --locked python experiments/05-model-predictive-control/scripts/run_batching_development.py \
  --learned-checkpoint CHECKPOINT --output FRESH_DIRECTORY
```
