"""Shared-axis plots derived from validated Experiment 01 trajectories."""

from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from feedback_control.report_evidence import EpisodeEvidence
from feedback_control.trajectory import TrajectoryRecord


def write_report_plots(
    output_dir: Path,
    episodes: tuple[EpisodeEvidence, ...],
) -> tuple[Path, ...]:
    return tuple(
        _write_signal_plot(
            output_dir / filename,
            episodes,
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
