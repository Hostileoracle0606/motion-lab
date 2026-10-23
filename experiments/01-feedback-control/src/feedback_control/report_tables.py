"""Machine-readable summary tables for Experiment 01 reports."""

import csv
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Callable

from feedback_control.report_evidence import EpisodeEvidence


def write_report_tables(
    output_dir: Path,
    episodes: tuple[EpisodeEvidence, ...],
) -> tuple[Path, Path, Path, list[dict[str, Any]]]:
    metrics_path = output_dir / "episode-metrics.csv"
    summary_path = output_dir / "controller-summary.csv"
    outcomes_path = output_dir / "episode-outcomes.csv"
    _write_episode_metrics(metrics_path, episodes)
    summary_rows = controller_summary_rows(episodes)
    _write_rows(summary_path, summary_rows)
    _write_episode_outcomes(outcomes_path, episodes)
    return metrics_path, summary_path, outcomes_path, summary_rows


def controller_summary_rows(
    episodes: tuple[EpisodeEvidence, ...],
) -> list[dict[str, Any]]:
    metric_specs: tuple[tuple[str, str, Callable[[EpisodeEvidence], Any]], ...] = (
        (
            "integrated_absolute_error",
            "rad s",
            lambda episode: episode.metrics["integrated_absolute_error_rad_s"],
        ),
        ("rise_time", "s", lambda episode: episode.metrics["rise_time"]["time_s"]),
        (
            "normalized_maximum_overshoot",
            "unitless",
            lambda episode: episode.metrics["normalized_maximum_overshoot"],
        ),
        (
            "settling_time",
            "s",
            lambda episode: episode.metrics["settling_time"]["time_s"],
        ),
        (
            "steady_state_absolute_error",
            "rad",
            lambda episode: episode.metrics["steady_state_absolute_error_rad"],
        ),
        (
            "squared_control_effort",
            "N^2 m^2 s",
            lambda episode: episode.metrics["squared_control_effort_nm2_s"],
        ),
        (
            "saturation_fraction",
            "fraction",
            lambda episode: episode.metrics["saturation_fraction"],
        ),
    )
    rows: list[dict[str, Any]] = []
    for controller in ("p", "pd"):
        controller_episodes = tuple(
            episode for episode in episodes if episode.controller == controller
        )
        included = tuple(
            episode
            for episode in controller_episodes
            if episode.included_in_aggregate
        )
        for metric_name, unit, get_value in metric_specs:
            values = tuple(
                float(value)
                for episode in included
                if (value := get_value(episode)) is not None
            )
            rows.append(
                {
                    "controller": controller,
                    "metric": metric_name,
                    "unit": unit,
                    "mean": fmean(values) if values else None,
                    "population_standard_deviation": (
                        pstdev(values) if values else None
                    ),
                    "measured_episode_count": len(values),
                    "included_episode_count": len(included),
                    "failed_episode_count": sum(
                        episode.status == "failed"
                        for episode in controller_episodes
                    ),
                    "excluded_episode_count": sum(
                        not episode.included_in_aggregate
                        for episode in controller_episodes
                    ),
                    "unsettled_episode_count": sum(
                        "not_settled" in episode.failure_reasons
                        for episode in controller_episodes
                    ),
                    "joint_limit_violation_count": sum(
                        "joint_limit_violation" in episode.failure_reasons
                        for episode in controller_episodes
                    ),
                    "non_finite_state_count": sum(
                        "non_finite_state" in episode.failure_reasons
                        for episode in controller_episodes
                    ),
                }
            )
    return rows


def _write_episode_metrics(
    path: Path,
    episodes: tuple[EpisodeEvidence, ...],
) -> None:
    fieldnames = (
        "episode_id",
        "controller",
        "scenario_id",
        "seed",
        "integrated_absolute_error_rad_s",
        "rise_time_outcome",
        "rise_time_s",
        "normalized_maximum_overshoot",
        "settling_time_outcome",
        "settling_time_s",
        "steady_state_error_availability",
        "steady_state_absolute_error_rad",
        "squared_control_effort_nm2_s",
        "saturation_fraction",
        "trajectory_file",
        "trajectory_sha256",
    )
    rows = []
    for episode in episodes:
        metrics = episode.metrics
        rows.append(
            {
                "episode_id": episode.episode_id,
                "controller": episode.controller,
                "scenario_id": episode.scenario_id,
                "seed": episode.seed,
                "integrated_absolute_error_rad_s": metrics[
                    "integrated_absolute_error_rad_s"
                ],
                "rise_time_outcome": metrics["rise_time"]["outcome"],
                "rise_time_s": metrics["rise_time"]["time_s"],
                "normalized_maximum_overshoot": metrics[
                    "normalized_maximum_overshoot"
                ],
                "settling_time_outcome": metrics["settling_time"]["outcome"],
                "settling_time_s": metrics["settling_time"]["time_s"],
                "steady_state_error_availability": metrics[
                    "steady_state_error_availability"
                ],
                "steady_state_absolute_error_rad": metrics[
                    "steady_state_absolute_error_rad"
                ],
                "squared_control_effort_nm2_s": metrics[
                    "squared_control_effort_nm2_s"
                ],
                "saturation_fraction": metrics["saturation_fraction"],
                "trajectory_file": episode.trajectory_file,
                "trajectory_sha256": episode.trajectory_sha256,
            }
        )
    _write_rows(path, rows, fieldnames=fieldnames)


def _write_episode_outcomes(
    path: Path,
    episodes: tuple[EpisodeEvidence, ...],
) -> None:
    rows = [
        {
            "episode_id": episode.episode_id,
            "controller": episode.controller,
            "scenario_id": episode.scenario_id,
            "seed": episode.seed,
            "status": episode.status,
            "failure_reasons": ";".join(episode.failure_reasons),
            "included_in_aggregate": episode.included_in_aggregate,
            "exclusion_reason": episode.exclusion_reason,
            "saturation_fraction": episode.metrics["saturation_fraction"],
            "trajectory_file": episode.trajectory_file,
            "trajectory_sha256": episode.trajectory_sha256,
        }
        for episode in episodes
    ]
    _write_rows(path, rows)


def _write_rows(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    fieldnames: tuple[str, ...] | None = None,
) -> None:
    if not rows:
        raise ValueError("cannot write an empty report table")
    selected_fields = fieldnames or tuple(rows[0])
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=selected_fields)
        writer.writeheader()
        writer.writerows(rows)
