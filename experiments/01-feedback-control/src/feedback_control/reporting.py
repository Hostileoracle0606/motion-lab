"""Generate Experiment 01 tables and plots from saved evaluation records."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Callable, Literal

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from feedback_control.trajectory import TrajectoryRecord, load_trajectory


REPORT_SCHEMA_VERSION = 1
ControllerName = Literal["p", "pd"]


@dataclass(frozen=True)
class EpisodeEvidence:
    episode_id: str
    controller: ControllerName
    scenario_id: str
    seed: int
    status: str
    failure_reasons: tuple[str, ...]
    included_in_aggregate: bool
    exclusion_reason: str | None
    trajectory_file: str
    trajectory_sha256: str
    config_digest: str
    metrics: dict[str, Any]
    trajectory: TrajectoryRecord


@dataclass(frozen=True)
class EvaluationEvidence:
    input_dir: Path
    evaluation_sha256: str
    provenance: dict[str, Any]
    episodes: tuple[EpisodeEvidence, ...]


def generate_report(
    *,
    input_dir: Path,
    output_dir: Path,
    command: str,
) -> tuple[Path, ...]:
    """Generate all report artifacts without running the simulator."""
    if not command.strip():
        raise ValueError("command must not be empty")
    evidence = load_evaluation_evidence(input_dir)
    resolved_output = output_dir.resolve()
    resolved_output.mkdir(parents=True, exist_ok=True)

    metrics_path = resolved_output / "episode-metrics.csv"
    summary_path = resolved_output / "controller-summary.csv"
    outcomes_path = resolved_output / "episode-outcomes.csv"
    _write_episode_metrics(metrics_path, evidence.episodes)
    summary_rows = _controller_summary_rows(evidence.episodes)
    _write_rows(summary_path, summary_rows)
    _write_episode_outcomes(outcomes_path, evidence.episodes)

    plot_paths = tuple(
        _write_signal_plot(
            resolved_output / filename,
            evidence.episodes,
            title=title,
            ylabel=ylabel,
            signal=signal,
            final_signal=final_signal,
            include_target=include_target,
        )
        for (
            filename,
            title,
            ylabel,
            signal,
            final_signal,
            include_target,
        ) in _signal_specs()
    )
    report_path = resolved_output / "report.md"
    _write_report(
        report_path,
        evidence=evidence,
        command=command,
        summary_rows=summary_rows,
        plot_paths=plot_paths,
    )
    return (report_path, metrics_path, summary_path, outcomes_path, *plot_paths)


def load_evaluation_evidence(input_dir: Path) -> EvaluationEvidence:
    """Load and cross-check an evaluation summary and all source trajectories."""
    resolved_input = input_dir.resolve()
    evaluation_path = resolved_input / "evaluation.json"
    payload = _load_json_object(evaluation_path)
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported evaluation schema version")
    provenance = _require_object(payload, "provenance")
    _validate_provenance(provenance)
    episodes_payload = payload.get("episodes")
    if not isinstance(episodes_payload, list) or not episodes_payload:
        raise ValueError("episodes must be a non-empty array")

    episodes = tuple(
        _load_episode(resolved_input, episode_payload, index)
        for index, episode_payload in enumerate(episodes_payload)
    )
    if len({episode.episode_id for episode in episodes}) != len(episodes):
        raise ValueError("episode identifiers must be unique")
    if any(
        episode.trajectory.code_version != provenance["code_version"]
        for episode in episodes
    ):
        raise ValueError("trajectory code versions do not match evaluation provenance")
    plan_by_controller = {
        controller: tuple(
            (episode.scenario_id, episode.seed)
            for episode in episodes
            if episode.controller == controller
        )
        for controller in ("p", "pd")
    }
    if not plan_by_controller["p"] or (
        plan_by_controller["p"] != plan_by_controller["pd"]
    ):
        raise ValueError("P and PD evidence must contain the same ordered cases")
    p_episodes = tuple(
        episode for episode in episodes if episode.controller == "p"
    )
    pd_episodes = tuple(
        episode for episode in episodes if episode.controller == "pd"
    )
    if any(
        p_episode.config_digest != pd_episode.config_digest
        for p_episode, pd_episode in zip(p_episodes, pd_episodes, strict=True)
    ):
        raise ValueError("paired P and PD cases must share one configuration")
    for controller in ("p", "pd"):
        gains = {
            (episode.trajectory.controller.kp, episode.trajectory.controller.kd)
            for episode in episodes
            if episode.controller == controller
        }
        if len(gains) != 1:
            raise ValueError(f"{controller.upper()} controller gains must be shared")
    return EvaluationEvidence(
        input_dir=resolved_input,
        evaluation_sha256=hashlib.sha256(evaluation_path.read_bytes()).hexdigest(),
        provenance=provenance,
        episodes=episodes,
    )


def _load_episode(
    input_dir: Path,
    payload: Any,
    index: int,
) -> EpisodeEvidence:
    if not isinstance(payload, dict):
        raise ValueError(f"episodes[{index}] must be an object")
    episode = payload
    episode_id = _require_string(episode, "episode_id", index)
    controller = _require_string(episode, "controller", index)
    if controller not in ("p", "pd"):
        raise ValueError(f"episodes[{index}].controller is invalid")
    scenario_id = _require_string(episode, "scenario_id", index)
    seed = episode.get("seed")
    if type(seed) is not int or seed < 0:
        raise ValueError(f"episodes[{index}].seed must be non-negative")
    trajectory_file = _require_string(episode, "trajectory_file", index)
    relative_trajectory = Path(trajectory_file)
    if relative_trajectory.is_absolute() or ".." in relative_trajectory.parts:
        raise ValueError(f"episodes[{index}].trajectory_file must stay in input_dir")
    trajectory_path = (input_dir / relative_trajectory).resolve()
    if not trajectory_path.is_relative_to(input_dir):
        raise ValueError(f"episodes[{index}].trajectory_file must stay in input_dir")
    expected_sha256 = _require_sha256(episode, "trajectory_sha256", index)
    actual_sha256 = hashlib.sha256(trajectory_path.read_bytes()).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(f"episodes[{index}] trajectory checksum mismatch")
    trajectory = load_trajectory(trajectory_path)
    config_digest = _require_sha256(episode, "config_digest", index)
    if (
        trajectory.episode_id != episode_id
        or trajectory.controller.name != controller
        or trajectory.seed != seed
        or trajectory.config_digest != config_digest
    ):
        raise ValueError(f"episodes[{index}] does not match its trajectory")

    status = _require_string(episode, "status", index)
    if status not in ("passed", "failed"):
        raise ValueError(f"episodes[{index}].status is invalid")
    failure_reasons_value = episode.get("failure_reasons")
    if not isinstance(failure_reasons_value, list) or any(
        not isinstance(reason, str) or not reason
        for reason in failure_reasons_value
    ):
        raise ValueError(f"episodes[{index}].failure_reasons is invalid")
    if len(set(failure_reasons_value)) != len(failure_reasons_value):
        raise ValueError(f"episodes[{index}].failure_reasons must be unique")
    included = episode.get("included_in_aggregate")
    if type(included) is not bool:
        raise ValueError(f"episodes[{index}].included_in_aggregate must be boolean")
    exclusion_reason = episode.get("exclusion_reason")
    if exclusion_reason is not None and (
        not isinstance(exclusion_reason, str) or not exclusion_reason
    ):
        raise ValueError(f"episodes[{index}].exclusion_reason is invalid")
    if (status == "failed") != bool(failure_reasons_value):
        raise ValueError(f"episodes[{index}] status does not match failure reasons")
    if included == (exclusion_reason is not None):
        raise ValueError(f"episodes[{index}] aggregate inclusion is inconsistent")
    metrics = _require_object(episode, "metrics")
    _validate_metrics(metrics, index)
    return EpisodeEvidence(
        episode_id=episode_id,
        controller=controller,
        scenario_id=scenario_id,
        seed=seed,
        status=status,
        failure_reasons=tuple(failure_reasons_value),
        included_in_aggregate=included,
        exclusion_reason=exclusion_reason,
        trajectory_file=relative_trajectory.as_posix(),
        trajectory_sha256=actual_sha256,
        config_digest=config_digest,
        metrics=metrics,
        trajectory=trajectory,
    )


def _validate_metrics(metrics: dict[str, Any], index: int) -> None:
    for name in (
        "integrated_absolute_error_rad_s",
        "normalized_maximum_overshoot",
        "squared_control_effort_nm2_s",
        "saturation_fraction",
    ):
        _require_finite(metrics.get(name), f"episodes[{index}].metrics.{name}")
        if metrics[name] < 0:
            raise ValueError(f"episodes[{index}].metrics.{name} must be non-negative")
    if metrics["saturation_fraction"] > 1:
        raise ValueError(
            f"episodes[{index}].metrics.saturation_fraction must not exceed one"
        )
    steady_state_error = metrics.get("steady_state_absolute_error_rad")
    if steady_state_error is not None:
        _require_finite(
            steady_state_error,
            f"episodes[{index}].metrics.steady_state_absolute_error_rad",
        )
    steady_state_availability = metrics.get("steady_state_error_availability")
    if steady_state_availability not in ("measured", "insufficient_duration"):
        raise ValueError(
            f"episodes[{index}].metrics.steady_state_error_availability is invalid"
        )
    if (steady_state_availability == "measured") != (
        steady_state_error is not None
    ):
        raise ValueError(
            f"episodes[{index}] steady-state availability is inconsistent"
        )
    for name in ("rise_time", "settling_time"):
        timing = _require_object(metrics, name)
        outcome = timing.get("outcome")
        if not isinstance(outcome, str) or not outcome:
            raise ValueError(f"episodes[{index}].metrics.{name}.outcome is invalid")
        if timing.get("time_s") is not None:
            _require_finite(
                timing["time_s"],
                f"episodes[{index}].metrics.{name}.time_s",
            )
            if timing["time_s"] < 0:
                raise ValueError(
                    f"episodes[{index}].metrics.{name}.time_s must be non-negative"
                )
        if (outcome == "measured") != (timing.get("time_s") is not None):
            raise ValueError(
                f"episodes[{index}].metrics.{name} outcome is inconsistent"
            )


def _write_episode_metrics(path: Path, episodes: tuple[EpisodeEvidence, ...]) -> None:
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


def _controller_summary_rows(
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


def _signal_specs() -> tuple[
    tuple[
        str,
        str,
        str,
        Callable[[Any], float],
        Callable[[TrajectoryRecord], float | None] | None,
        bool,
    ],
    ...,
]:
    return (
        (
            "position-rad.png",
            "Joint position",
            "Position (rad)",
            lambda transition: transition.state.position_rad,
            lambda record: record.transitions[-1].next_state.position_rad,
            True,
        ),
        (
            "velocity-rad-s.png",
            "Joint velocity",
            "Velocity (rad/s)",
            lambda transition: transition.state.velocity_rad_s,
            lambda record: record.transitions[-1].next_state.velocity_rad_s,
            False,
        ),
        (
            "position-error-rad.png",
            "Position error",
            "Target - position (rad)",
            lambda transition: (
                transition.target.position_rad - transition.state.position_rad
            ),
            lambda record: (
                None
                if record.transitions[-1].next_state.position_rad is None
                else record.transitions[-1].target.position_rad
                - record.transitions[-1].next_state.position_rad
            ),
            False,
        ),
        (
            "controller-output-nm.png",
            "Controller output before clipping",
            "Raw torque (N m)",
            lambda transition: transition.raw_torque_nm,
            None,
            False,
        ),
        (
            "applied-action-nm.png",
            "Applied action after clipping",
            "Applied torque (N m)",
            lambda transition: transition.applied_torque_nm,
            None,
            False,
        ),
    )


def _write_signal_plot(
    path: Path,
    episodes: tuple[EpisodeEvidence, ...],
    *,
    title: str,
    ylabel: str,
    signal: Callable[[Any], float],
    final_signal: Callable[[TrajectoryRecord], float | None] | None,
    include_target: bool,
) -> Path:
    case_keys = tuple(
        dict.fromkeys((episode.scenario_id, episode.seed) for episode in episodes)
    )
    figure, axes = plt.subplots(
        len(case_keys),
        1,
        figsize=(10, max(3.2, 2.7 * len(case_keys))),
        sharex=True,
        squeeze=False,
        constrained_layout=True,
    )
    colors = {"p": "#D97706", "pd": "#2563EB"}
    for axis, (scenario_id, seed) in zip(axes[:, 0], case_keys, strict=True):
        case_episodes = tuple(
            episode
            for episode in episodes
            if (episode.scenario_id, episode.seed) == (scenario_id, seed)
        )
        for episode in case_episodes:
            transitions = episode.trajectory.transitions
            times = [transition.time_s for transition in transitions]
            values = [signal(transition) for transition in transitions]
            if final_signal is not None:
                final_value = final_signal(episode.trajectory)
                if final_value is not None:
                    times.append(
                        transitions[-1].time_s
                        + episode.trajectory.configuration.control_interval_s
                    )
                    values.append(final_value)
            gains = episode.trajectory.controller
            controller_label = (
                f"{episode.controller.upper()} "
                f"(Kp={gains.kp:g}, Kd={gains.kd:g})"
            )
            axis.plot(
                times,
                values,
                color=colors[episode.controller],
                linewidth=1.5,
                label=controller_label,
            )
        if include_target:
            target = case_episodes[0].trajectory.transitions[0].target.position_rad
            axis.axhline(
                target,
                color="#111827",
                linestyle="--",
                linewidth=1.0,
                label=f"Target ({target:g} rad)",
            )
        elif "error" in title.lower() or "velocity" in title.lower():
            axis.axhline(0.0, color="#6B7280", linewidth=0.8, linestyle=":")
        axis.set_title(f"{scenario_id} · seed {seed}", fontsize=10)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.2)
        axis.legend(loc="best", fontsize=8)
    axes[-1, 0].set_xlabel("Time (s)")
    figure.suptitle(f"Experiment 01 — {title}", fontsize=13)
    figure.savefig(
        path,
        dpi=140,
        metadata={
            "Title": f"Experiment 01 — {title}",
            "Author": "Motion Lab",
            "Software": "Motion Lab report generator",
        },
    )
    plt.close(figure)
    return path


def _write_report(
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
        f"Source commit: `{provenance.get('code_version', 'unknown')}`  ",
        (
            "Source working tree dirty: "
            f"`{provenance.get('working_tree_dirty', 'unknown')}`  "
        ),
        f"Configuration file: `{provenance.get('config_file', 'unknown')}`  ",
        f"Configuration SHA-256: `{provenance.get('config_sha256', 'unknown')}`  ",
        f"Python: `{provenance.get('python_version', 'unknown')}`  ",
        f"MuJoCo: `{provenance.get('mujoco_version', 'unknown')}`  ",
        f"Platform: `{provenance.get('platform', 'unknown')}`  ",
        f"Evaluation command: `{provenance.get('command', 'unknown')}`  ",
        f"Report command: `{command}`",
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
        "The following values are derived from the saved machine-readable records. "
        "Dispersion is the population standard deviation across included episodes.",
        "",
        (
            "| Controller | Episodes | Failed | Excluded | Unsettled | "
            "Joint-limit | Non-finite |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for controller in controllers:
        controller_episodes = tuple(
            episode
            for episode in evidence.episodes
            if episode.controller == controller
        )
        failed_count = sum(
            episode.status == "failed" for episode in controller_episodes
        )
        excluded_count = sum(
            not episode.included_in_aggregate
            for episode in controller_episodes
        )
        unsettled_count = sum(
            "not_settled" in episode.failure_reasons
            for episode in controller_episodes
        )
        joint_limit_count = sum(
            "joint_limit_violation" in episode.failure_reasons
            for episode in controller_episodes
        )
        non_finite_count = sum(
            "non_finite_state" in episode.failure_reasons
            for episode in controller_episodes
        )
        lines.append(
            f"| {controller.upper()} | {len(controller_episodes)} | "
            f"{failed_count} | {excluded_count} | {unsettled_count} | "
            f"{joint_limit_count} | {non_finite_count} |"
        )
    lines.extend(
        [
            "",
            "| Controller | Metric | Unit | Mean | Std. dev. | Measured / included |",
            "| --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for controller in controllers:
        for row in summary_rows:
            if row["controller"] != controller:
                continue
            lines.append(
                (
                    "| {controller} | {metric} | {unit} | {mean} | {std} | "
                    "{count} / {total} |"
                ).format(
                    controller=controller.upper(),
                    metric=row["metric"],
                    unit=row["unit"],
                    mean=_format_optional(row["mean"]),
                    std=_format_optional(row["population_standard_deviation"]),
                    count=row["measured_episode_count"],
                    total=row["included_episode_count"],
                )
            )
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
    lines.extend(
        [
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
    )
    for episode in evidence.episodes:
        source_path = evidence.input_dir / episode.trajectory_file
        relative_source = Path(os.path.relpath(source_path, path.parent)).as_posix()
        lines.append(
            f"| {episode.episode_id} | {episode.controller.upper()} | "
            f"{episode.scenario_id} | {episode.seed} | {episode.status} | "
            f"[{episode.trajectory_file}]({relative_source}) | "
            f"`{episode.trajectory_sha256}` |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as input_file:
            value = json.load(input_file)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load evaluation summary: {path}") from error
    if not isinstance(value, dict):
        raise ValueError("evaluation summary must be an object")
    return value


def _require_object(payload: dict[str, Any], field_name: str) -> dict[str, Any]:
    value = payload.get(field_name)
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object")
    return value


def _validate_provenance(provenance: dict[str, Any]) -> None:
    code_version = provenance.get("code_version")
    if (
        not isinstance(code_version, str)
        or len(code_version) not in (40, 64)
        or any(character not in "0123456789abcdef" for character in code_version)
    ):
        raise ValueError("provenance.code_version must be a Git object ID")
    if type(provenance.get("working_tree_dirty")) is not bool:
        raise ValueError("provenance.working_tree_dirty must be boolean")
    config_sha256 = provenance.get("config_sha256")
    if (
        not isinstance(config_sha256, str)
        or len(config_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in config_sha256
        )
    ):
        raise ValueError("provenance.config_sha256 must be a SHA-256 digest")
    for field_name in (
        "config_file",
        "command",
        "python_version",
        "platform",
        "mujoco_version",
    ):
        value = provenance.get(field_name)
        if not isinstance(value, str) or not value:
            raise ValueError(f"provenance.{field_name} must not be empty")


def _require_string(payload: dict[str, Any], field_name: str, index: int) -> str:
    value = payload.get(field_name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"episodes[{index}].{field_name} must not be empty")
    return value


def _require_sha256(payload: dict[str, Any], field_name: str, index: int) -> str:
    value = _require_string(payload, field_name, index)
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"episodes[{index}].{field_name} must be a SHA-256 digest")
    return value


def _require_finite(value: Any, field_name: str) -> None:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be finite")
    try:
        finite = math.isfinite(value)
    except TypeError as error:
        raise ValueError(f"{field_name} must be finite") from error
    if not finite:
        raise ValueError(f"{field_name} must be finite")


def _format_optional(value: float | None) -> str:
    return "unavailable" if value is None else f"{value:.6g}"
