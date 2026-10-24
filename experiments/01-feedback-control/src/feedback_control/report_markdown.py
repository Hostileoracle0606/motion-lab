"""Human-readable report assembled from validated Experiment 01 evidence."""

import os
from pathlib import Path
from typing import Any

from feedback_control.report_evidence import EvaluationEvidence
from feedback_control.report_tables import controller_outcome_counts


REPORT_SCHEMA_VERSION = 1


def write_markdown_report(
    path: Path,
    *,
    evidence: EvaluationEvidence,
    command: str,
    summary_rows: list[dict[str, Any]],
    plot_paths: tuple[Path, ...],
) -> None:
    provenance = evidence.provenance
    failed = sum(episode.status == "failed" for episode in evidence.episodes)
    excluded = sum(not episode.included_in_aggregate for episode in evidence.episodes)
    cases = {
        (episode.scenario_id, episode.seed) for episode in evidence.episodes
    }
    controllers = ("p", "pd")
    seeds = sorted({episode.seed for episode in evidence.episodes})
    lines = [
        "# Experiment 01 generated results",
        "",
        f"Report schema: `{REPORT_SCHEMA_VERSION}`  ",
        f"Source evaluation SHA-256: `{evidence.evaluation_sha256}`  ",
        f"Source commit: `{provenance['code_version']}`  ",
        f"Source working tree dirty: `{provenance['working_tree_dirty']}`  ",
        f"Configuration file: `{provenance['config_file']}`  ",
        f"Configuration SHA-256: `{provenance['config_sha256']}`  ",
        f"Python: `{provenance['python_version']}`  ",
        f"MuJoCo: `{provenance['mujoco_version']}`  ",
        f"Platform: `{provenance['platform']}`  ",
        f"Evaluation command: `{provenance['command']}`  ",
        f"Report command: `{command}`",
        "",
        "## Protocol",
        "",
        *_protocol_lines(evidence),
        "",
        "## Observed",
        "",
        (
            f"The records contain {len(evidence.episodes)} episodes across "
            f"{len(cases)} scenario-seed cases. {failed} episodes are marked "
            f"failed and {excluded} are excluded from aggregates."
        ),
        f"Declared seeds present in the saved records: `{seeds}`.",
        "",
        "| Controller | Kp | Kd |",
        "| --- | ---: | ---: |",
    ]
    for controller in controllers:
        controller_episode = next(
            episode
            for episode in evidence.episodes
            if episode.controller == controller
        )
        gains = controller_episode.trajectory.controller
        lines.append(
            f"| {controller.upper()} | {gains.kp:g} | {gains.kd:g} |"
        )
    lines.extend(
        [
            "",
            "The following values are derived from the saved machine-readable "
            "records. Dispersion is the population standard deviation across "
            "included episodes.",
            "",
            (
                "| Controller | Episodes | Failed | Excluded | Unsettled | "
                "Joint-limit | Non-finite |"
            ),
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for controller in controllers:
        lines.append(_outcome_row(evidence, controller))
    lines.extend(
        [
            "",
            "| Controller | Metric | Unit | Mean | Std. dev. | Measured / included |",
            "| --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for controller in controllers:
        for row in summary_rows:
            if row["controller"] == controller:
                lines.append(_metric_row(row))
    lines.extend(
        [
            "",
            "Machine-readable tables: [episode metrics](episode-metrics.csv), "
            "[controller summary](controller-summary.csv), and "
            "[episode outcomes](episode-outcomes.csv).",
            "",
        ]
    )
    for plot_path in plot_paths:
        label = plot_path.stem.replace("-", " ").title()
        lines.extend((f"### {label}", "", f"![{label}]({plot_path.name})", ""))
    lines.extend(_interpretation_and_sources(path, evidence))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _protocol_lines(evidence: EvaluationEvidence) -> list[str]:
    metrics = evidence.config.metrics
    return [
        "### Gain selection",
        "",
        (
            "P and PD use the same proportional gain so derivative feedback is "
            "the central changed term. The PD derivative gain is the nearest "
            "declared sweep candidate to the local continuous-time critical-"
            "damping estimate `Kd = 2 * sqrt(I * Kp) - b`. This estimate assumes "
            "locally constant inertia, zero gravity, and no saturation."
        ),
        "",
        "### Metric definitions and tolerances",
        "",
        (
            f"- **Rise time:** first sampled signed-command crossings from "
            f"{metrics.rise_lower_fraction:g} to "
            f"{metrics.rise_upper_fraction:g} of the commanded displacement, "
            "without interpolation."
        ),
        (
            f"- **Settling time:** earliest sample after which absolute position "
            f"error stays within {metrics.settling_position_tolerance_rad:g} rad "
            f"and absolute velocity stays within "
            f"{metrics.settling_velocity_tolerance_rad_s:g} rad/s."
        ),
        (
            "- **Normalized maximum overshoot:** largest progress beyond the "
            "target in the signed command direction, divided by absolute commanded "
            "displacement; reported as a fraction."
        ),
        (
            f"- **Steady-state absolute error:** mean absolute position error over "
            f"the final {metrics.steady_state_window_s:g} s of action-bearing "
            "samples."
        ),
        (
            "- **Integrated absolute error:** sum of absolute position error times "
            "the control interval over action-bearing samples."
        ),
        (
            "- **Squared control effort:** sum of squared applied torque times the "
            "control interval, in `(N m)^2 s`; this is a proxy, not physical energy."
        ),
        (
            "- **Saturation fraction:** fraction of applied actions exactly at the "
            "declared symmetric torque limit."
        ),
    ]


def _outcome_row(evidence: EvaluationEvidence, controller: str) -> str:
    counts = controller_outcome_counts(evidence.episodes, controller)
    return (
        f"| {controller.upper()} | {counts.episode_count} | "
        f"{counts.failed_episode_count} | {counts.excluded_episode_count} | "
        f"{counts.unsettled_episode_count} | "
        f"{counts.joint_limit_violation_count} | "
        f"{counts.non_finite_state_count} |"
    )


def _metric_row(row: dict[str, Any]) -> str:
    return (
        "| {controller} | {metric} | {unit} | {mean} | {std} | "
        "{count} / {total} |"
    ).format(
        controller=row["controller"].upper(),
        metric=row["metric"],
        unit=row["unit"],
        mean=_format_optional(row["mean"]),
        std=_format_optional(row["population_standard_deviation"]),
        count=row["measured_episode_count"],
        total=row["included_episode_count"],
    )


def _interpretation_and_sources(
    report_path: Path,
    evidence: EvaluationEvidence,
) -> list[str]:
    lines = [
        "## Interpretation",
        "",
        "These tables and figures compare the declared P and PD controllers "
        "only under the recorded noiseless one-link protocol. Differences in "
        "the reported values are associations under that shared protocol; the "
        "generator does not infer a broader causal or performance claim.",
        "",
        "## Unknowns and limitations",
        "",
        "- Transfer to other mechanisms, gains, disturbances, sensor noise, "
        "delays, or physical hardware is not measured.",
        "- The declared seed policy may provide limited or no stochastic "
        "variation; scenario dispersion is not a substitute for multi-seed "
        "uncertainty.",
        "- Failed and unsettled episodes remain in the evidence and aggregate "
        "counts; unavailable timing or tail metrics are not imputed.",
        "",
        "## Source records",
        "",
        (
            "| Episode | Controller | Scenario | Seed | Status | "
            "Trajectory | SHA-256 |"
        ),
        "| --- | --- | --- | ---: | --- | --- | --- |",
    ]
    for episode in evidence.episodes:
        source_path = evidence.input_dir / episode.trajectory_file
        relative_source = Path(
            os.path.relpath(source_path, report_path.parent)
        ).as_posix()
        lines.append(
            f"| {episode.episode_id} | {episode.controller.upper()} | "
            f"{episode.scenario_id} | {episode.seed} | {episode.status} | "
            f"[{episode.trajectory_file}]({relative_source}) | "
            f"`{episode.trajectory_sha256}` |"
        )
    return lines


def _format_optional(value: float | None) -> str:
    return "unavailable" if value is None else f"{value:.6g}"
