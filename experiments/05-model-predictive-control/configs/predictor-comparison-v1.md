# Predictor comparison protocol lock

The [frozen manifest](predictor-comparison-v1.manifest.json) was generated
from source commit `c6cb38f18b779bb4cb79f56852e363a45bfc18b6` before
any held-out episode was executed. Its SHA-256 is
`a968a37196dc60f15589727ea05d94de1d5bf5c9d0a1d2a9959dc4fc5d23200c`.
The [TOML configuration](predictor-comparison-v1.toml) declares the
thresholds and optimizer budget. The manifest pins that configuration,
the relevant code and attempt policy, the published checkpoint, the
expanded case and attempt rosters, and each truth-model digest.

| Run identity | Mode | Planned attempts |
| --- | --- | ---: |
| `fixed-work-01` | Fixed work | 320 |
| `fixed-work-02` | Fixed work | 320 |
| `deadline-discard-01` | 10 ms deadline with PD fallback | 320 |

Each run pairs 20 base seeds across four truth conditions and four
controllers. The three MPC controllers use the same random-shooting
objective, horizon and candidate budget. All predictors receive nominal
model context and the same scalar observations. The nominal MuJoCo
predictor uses the nominal truth engine and parameters. Under perturbation
it still uses its nominal model. The learned checkpoint was selected by
its published canonical run identity and validation-selected epoch.

The ordinary comparison command passed on the separate development
smoke configuration in both modes. It completed 16 attempts in each mode,
and the strict study audit accepted both declared smoke identities. These
runs checked the adapters, evidence recording and audit.

Use the [experiment instructions](../README.md#frozen-predictor-comparison-protocol)
to obtain the published checkpoint and regenerate preflight. Supply this
manifest to the comparison and audit commands when executing #48. Require
all three frozen run identities and the strict [attempt policy](../README.md#attempt-accounting-and-recovery)
before computing final metrics. Issue #48 owns the held-out execution,
per-condition and paired reporting, publication, and independent
reproduction.
