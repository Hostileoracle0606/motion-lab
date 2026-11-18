"""Reader-facing limitations-first Markdown for Experiment 04."""

from collections import defaultdict
import statistics

from learned_dynamics.config import PROTOCOL_VARIANTS


def _study_table() -> str:
    rows = []
    eligible = {"all": 940, "history_common": 936, "loss_common": 924}
    for variant, (width, history, loss, membership) in PROTOCOL_VARIANTS.items():
        inputs = 3 * (1 + history)
        parameters = inputs * width + width + width * width + width + width * 2 + 2
        rows.append(
            f"| {variant} | {parameters:,} | {inputs} | {membership} | "
            f"{eligible[membership]} / {940 - eligible[membership]} | {loss} |")
    return "\n".join(rows)


def _timing_table(benchmark: dict) -> str:
    grouped = defaultdict(list)
    failures = 0
    for row in benchmark["results"]:
        if row["status"] == "success":
            grouped[(row["device"], row["mode"], row["batch_size"])].append(
                float(row["summary"]["median_s"]))
        else:
            failures += 1
    rows = [
        f"| {device} | {mode} | {batch} | "
        f"{statistics.median(values) * 1000:.3f} | {len(values)} |"
        for (device, mode, batch), values in sorted(grouped.items())
    ]
    if not rows:
        rows.append("| unavailable | unavailable | - | - | 0 |")
    return "\n".join(rows), failures


def render_report(seed_rows: tuple[dict, ...], benchmark: dict, source_sha: str,
                  episode_outcomes: list[dict], provenance: dict) -> str:
    complete = sum(bool(row["complete_seed_matrix"]) for row in seed_rows)
    scenarios = sorted({row["scenario_id"] for row in episode_outcomes})
    scenario_text = ", ".join(f"`{value}`" for value in scenarios) or "not recorded"
    environment = provenance["benchmark_environment"]
    hardware = environment.get("hardware_model", "unavailable in original evidence")
    benchmark_status = provenance["benchmark_status"]
    report_source = provenance["report_generator_provenance"]["source_sha"]
    timing_table, timing_failures = _timing_table(benchmark)
    study_table = _study_table()
    return f"""# Experiment 04 learned-dynamics study

Evaluation source identity: `{source_sha}`.
Report-generator source identity: `{report_source}`.

This report is derived from independently audited saved evaluation and timing
records. Report generation performed no training, prediction, or simulator
query. {complete} seed/horizon groups contain all five declared successful
training seeds; incomplete groups remain visible in `seed_summary.csv`.

## Interpretation boundary

The five training seeds measure sensitivity of optimization on one fixed
dataset. They do not create five independent held-out scenario tests. Test
overlapping windows are correlated observations, not independent trials.
The held-out outcomes contain {len(episode_outcomes)} episodes from scenario
{scenario_text}. They do not test unseen mass, friction, noise, hardware, or
physical-robot transfer.

MuJoCo is a known-physics consistency reference with privileged access to the
same simulator family. Analytical baselines also use known plant parameters
and omit some soft-limit/contact effects. A learned model need not beat those
references for the experiment to remain informative.

## Frozen study design

| Variant | Parameters | Inputs | Membership | Train eligible / excluded | Loss |
| --- | ---: | ---: | --- | ---: | --- |
{study_table}

All variants use five final-test seeds. `study_summary.csv` expands this design
to all 45 declared runs and records update counts, training seconds, failures,
and selected validation checkpoints.

## Inference timing summary

Median latency below is the median of the per-round medians, in milliseconds.

| Device | Mode | Batch | Median ms | Successful rounds |
| --- | --- | ---: | ---: | ---: |
{timing_table}

Failed benchmark workloads retained: {timing_failures}.

## Artifacts

`one_step.csv`, `rollouts.csv`, `coverage.csv`, `per_episode.csv`,
`seed_summary.csv`, `learning_curves.csv`, `study_summary.csv`, `timing.csv`,
and `timing_summary.csv` retain numeric sources for the two horizon figures and
the declared training/benchmark comparisons. `study_summary.csv` includes
parameter counts, training-membership exclusions, update counts, elapsed
training time, failures, and checkpoint selection outcomes.

Benchmark status is `{benchmark_status}`. The recorded hardware model is
`{hardware}`; power state was `{environment.get("power_state", "not recorded")}`
and competing load was `{environment.get("competing_load", "not recorded")}`.
Timing describes this declared machine and workload only; it is not evidence of
real-time closed-loop control. `provenance.json` binds the source/config/input
hashes, runtime disclosures, and the regeneration command.
"""
