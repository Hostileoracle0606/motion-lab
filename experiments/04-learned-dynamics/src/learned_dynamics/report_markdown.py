"""Reader-facing limitations-first Markdown for Experiment 04."""


def render_report(seed_rows: tuple[dict, ...], benchmark: dict, source_sha: str) -> str:
    complete = sum(bool(row["complete_seed_matrix"]) for row in seed_rows)
    return f"""# Experiment 04 learned-dynamics study

Source identity: `{source_sha}`.

This report is derived from independently audited saved evaluation and timing
records. Report generation performed no training, prediction, or simulator
query. {complete} seed/horizon groups contain all five declared successful
training seeds; incomplete groups remain visible in `seed_summary.csv`.

## Interpretation boundary

The five training seeds measure sensitivity of optimization on one fixed
dataset. They do not create five independent held-out scenario tests. Test
The overlapping windows are correlated observations, not independent trials.
The held-out data contain two narrow direction/displacement episodes, including
joint-limit behavior; they do not test unseen mass, friction, noise, hardware,
or physical-robot transfer.

MuJoCo is a known-physics consistency reference with privileged access to the
same simulator family. Analytical baselines also use known plant parameters
and omit some soft-limit/contact effects. A learned model need not beat those
references for the experiment to remain informative.

## Artifacts

`one_step.csv`, `rollouts.csv`, `coverage.csv`, `per_episode.csv`,
`seed_summary.csv`, `learning_curves.csv`, and `timing.csv` retain numeric
sources for the two horizon figures. Timing describes the declared machine and
workload only; it is not evidence of real-time closed-loop control.
"""
