"""Replay the tested P and PD trajectories in a recordable MuJoCo window."""

import argparse
import os
from pathlib import Path
import sys

import mujoco.viewer

from feedback_control.config import ExperimentConfig, load_experiment_config
from feedback_control.playback import (
    PlaybackOptions,
    configure_demo_camera,
    load_playback_model,
    replay_comparison,
    replay_episode,
)
from feedback_control.simulation import ControllerName, EpisodeResult, run_episode


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = EXPERIMENT_DIR / "configs" / "demo.toml"
COMPARISON_MODEL = EXPERIMENT_DIR / "models" / "one_link_comparison.xml"


def main() -> None:
    arguments = _parse_arguments()
    _require_mjpython_on_macos()

    config = load_experiment_config(arguments.config)
    controllers: tuple[ControllerName, ...]
    if arguments.controller == "both":
        controllers = ("p", "pd")
    else:
        controllers = (arguments.controller,)

    # Results are computed completely before the presentation window opens.
    results = tuple(run_episode(config, controller) for controller in controllers)
    options = PlaybackOptions(
        speed=arguments.speed,
        intro_seconds=arguments.intro_seconds,
        hold_seconds=arguments.hold_seconds,
    )

    print("Visual demo ready. Close the MuJoCo window at any time to stop.")
    print("The viewer replays recorded states; it does not rerun the controller.")
    if arguments.controller == "both":
        _run_side_by_side_demo(config, results, options, arguments.repeat)
    else:
        _run_single_demo(config, results[0], options, arguments.repeat)


def _run_side_by_side_demo(
    config: ExperimentConfig,
    results: tuple[EpisodeResult, ...],
    options: PlaybackOptions,
    repeat: int,
) -> None:
    model, data = load_playback_model(COMPARISON_MODEL)
    with mujoco.viewer.launch_passive(
        model,
        data,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        configure_demo_camera(model, viewer, "comparison_camera")
        for repeat_index in range(repeat):
            if repeat > 1:
                print(f"Replay {repeat_index + 1}/{repeat}")
            print("Showing P and PD side by side")
            if not replay_comparison(
                model=model,
                data=data,
                viewer=viewer,
                config=config,
                p_result=results[0],
                pd_result=results[1],
                options=options,
            ):
                return


def _run_single_demo(
    config: ExperimentConfig,
    result: EpisodeResult,
    options: PlaybackOptions,
    repeat: int,
) -> None:
    model, data = load_playback_model(config.model.path)
    with mujoco.viewer.launch_passive(
        model,
        data,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        configure_demo_camera(model, viewer)
        for repeat_index in range(repeat):
            if repeat > 1:
                print(f"Replay {repeat_index + 1}/{repeat}")
            print(f"Showing {result.controller.upper()} controller")
            if not replay_episode(
                model=model,
                data=data,
                viewer=viewer,
                config=config,
                result=result,
                options=options,
            ):
                return


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--controller",
        choices=("p", "pd", "both"),
        default="both",
        help="trajectory to show; both uses a side-by-side view (default: both)",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="playback speed multiplier (default: 1.0)",
    )
    parser.add_argument(
        "--intro-seconds",
        type=float,
        default=2.0,
        help="hold the starting pose before each run (default: 2.0)",
    )
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=1.5,
        help="hold the final pose after each run (default: 1.5)",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="number of times to play the requested comparison (default: 1)",
    )
    arguments = parser.parse_args()
    if arguments.repeat < 1:
        parser.error("--repeat must be at least 1")
    return arguments


def _require_mjpython_on_macos() -> None:
    if sys.platform == "darwin" and "MJPYTHON_BIN" not in os.environ:
        raise SystemExit(
            "MuJoCo's passive viewer requires mjpython on macOS. Run:\n"
            "  uv run mjpython "
            "experiments/01-feedback-control/scripts/run_visual_demo.py"
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nVisual demo stopped.")
