"""Real-time visualization of recorded feedback-control episodes.

Playback is intentionally downstream of simulation: it copies recorded states
into a MuJoCo model for display but never advances physics or recomputes an
action. Closing the viewer therefore cannot change an experiment result.
"""

from dataclasses import dataclass
import math
from pathlib import Path
import time
from typing import Protocol

import mujoco

from feedback_control.config import ExperimentConfig
from feedback_control.simulation import ControllerName, EpisodeResult


@dataclass(frozen=True)
class PlaybackFrame:
    """One recorded state formatted for visual replay."""

    time_s: float
    position_rad: float
    velocity_rad_s: float
    target_position_rad: float
    applied_torque_nm: float
    saturated: bool


@dataclass(frozen=True)
class PlaybackOptions:
    """Presentation timing that does not affect simulated motion."""

    speed: float = 1.0
    intro_seconds: float = 2.0
    hold_seconds: float = 1.5


class ViewerHandle(Protocol):
    """Small portion of the passive viewer API used during playback."""

    cam: mujoco.MjvCamera

    def is_running(self) -> bool: ...

    def set_texts(self, texts: list[tuple[object, object, str, str]]) -> None: ...

    def sync(self, state_only: bool = False) -> None: ...


ARM_COLORS: dict[ControllerName, tuple[float, float, float, float]] = {
    "p": (0.98, 0.53, 0.12, 1.0),
    "pd": (0.10, 0.62, 0.95, 1.0),
}


def build_playback_frames(result: EpisodeResult) -> tuple[PlaybackFrame, ...]:
    """Convert an episode record into immutable display frames.

    The final state is appended with zero torque because no new action is
    selected after an episode ends.
    """
    if not result.samples:
        raise ValueError("cannot replay an episode with no samples")

    frames = [
        PlaybackFrame(
            time_s=sample.time_s,
            position_rad=sample.position_rad,
            velocity_rad_s=sample.velocity_rad_s,
            target_position_rad=sample.target_position_rad,
            applied_torque_nm=sample.applied_torque_nm,
            saturated=sample.saturated,
        )
        for sample in result.samples
    ]
    frames.append(
        PlaybackFrame(
            time_s=result.final_time_s,
            position_rad=result.final_position_rad,
            velocity_rad_s=result.final_velocity_rad_s,
            target_position_rad=result.samples[-1].target_position_rad,
            applied_torque_nm=0.0,
            saturated=False,
        )
    )
    return tuple(frames)


def build_comparison_frames(
    p_result: EpisodeResult,
    pd_result: EpisodeResult,
) -> tuple[tuple[PlaybackFrame, PlaybackFrame], ...]:
    """Pair synchronized P and PD frames for a fair visual comparison."""
    if p_result.controller != "p" or pd_result.controller != "pd":
        raise ValueError("comparison requires P results followed by PD results")

    p_frames = build_playback_frames(p_result)
    pd_frames = build_playback_frames(pd_result)
    if len(p_frames) != len(pd_frames):
        raise ValueError("comparison trajectories must have equal frame counts")

    pairs = tuple(zip(p_frames, pd_frames, strict=True))
    for p_frame, pd_frame in pairs:
        if not math.isclose(
            p_frame.time_s,
            pd_frame.time_s,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("comparison trajectories must share one timeline")
    return pairs


def validate_playback_options(options: PlaybackOptions) -> None:
    """Reject presentation settings that cannot produce a valid replay."""
    for name, value in (
        ("speed", options.speed),
        ("intro_seconds", options.intro_seconds),
        ("hold_seconds", options.hold_seconds),
    ):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if options.speed <= 0:
        raise ValueError("speed must be greater than zero")
    if options.intro_seconds < 0:
        raise ValueError("intro_seconds must be non-negative")
    if options.hold_seconds < 0:
        raise ValueError("hold_seconds must be non-negative")


def controller_description(
    config: ExperimentConfig,
    controller: ControllerName,
) -> tuple[str, str]:
    """Return concise controller and learning labels for the viewer."""
    if controller == "p":
        return (
            f"P CONTROLLER   Kp={config.p.kp:.2f}",
            "Torque responds to position error only",
        )
    return (
        f"PD CONTROLLER   Kp={config.pd.kp:.2f}   Kd={config.pd.kd:.2f}",
        "Derivative feedback opposes velocity",
    )


def replay_episode(
    *,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    viewer: ViewerHandle,
    config: ExperimentConfig,
    result: EpisodeResult,
    options: PlaybackOptions,
) -> bool:
    """Replay one episode at wall-clock speed while the viewer is open.

    Returns ``False`` if the user closes the viewer during this episode.
    """
    validate_playback_options(options)
    frames = build_playback_frames(result)
    _set_controller_color(model, result.controller)
    _apply_frame(model, data, frames[0])
    _set_overlay(viewer, config, result.controller, frames[0], "STARTING")
    viewer.sync()

    if not _hold_viewer(viewer, options.intro_seconds):
        return False

    wall_start = time.perf_counter()
    for frame in frames:
        if not _wait_until(
            viewer,
            wall_start + frame.time_s / options.speed,
        ):
            return False
        _apply_frame(model, data, frame)
        _set_overlay(viewer, config, result.controller, frame, "REPLAY")
        viewer.sync()

    _set_overlay(viewer, config, result.controller, frames[-1], "COMPLETE")
    viewer.sync()
    return _hold_viewer(viewer, options.hold_seconds)


def replay_comparison(
    *,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    viewer: ViewerHandle,
    config: ExperimentConfig,
    p_result: EpisodeResult,
    pd_result: EpisodeResult,
    options: PlaybackOptions,
) -> bool:
    """Replay synchronized P and PD trajectories in one two-arm scene."""
    validate_playback_options(options)
    frame_pairs = build_comparison_frames(p_result, pd_result)
    _validate_comparison_model(model)

    first_p, first_pd = frame_pairs[0]
    _apply_comparison_frame(model, data, first_p, first_pd)
    _set_comparison_overlay(viewer, config, first_p, first_pd, "STARTING")
    viewer.sync()
    if not _hold_viewer(viewer, options.intro_seconds):
        return False

    wall_start = time.perf_counter()
    for p_frame, pd_frame in frame_pairs:
        if not _wait_until(
            viewer,
            wall_start + p_frame.time_s / options.speed,
        ):
            return False
        _apply_comparison_frame(model, data, p_frame, pd_frame)
        _set_comparison_overlay(viewer, config, p_frame, pd_frame, "REPLAY")
        viewer.sync()

    final_p, final_pd = frame_pairs[-1]
    _set_comparison_overlay(viewer, config, final_p, final_pd, "COMPLETE")
    viewer.sync()
    return _hold_viewer(viewer, options.hold_seconds)


def configure_demo_camera(
    model: mujoco.MjModel,
    viewer: ViewerHandle,
    camera_name: str = "demo_camera",
) -> None:
    """Select the fixed camera declared by the experiment model."""
    camera_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_CAMERA,
        camera_name,
    )
    if camera_id < 0:
        raise ValueError(f"model does not define the {camera_name}")
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    viewer.cam.fixedcamid = camera_id


def load_playback_model(model_path: Path) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Load an independent model used only for visual replay."""
    model = mujoco.MjModel.from_xml_path(str(model_path))
    return model, mujoco.MjData(model)


def _apply_frame(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    frame: PlaybackFrame,
) -> None:
    data.time = frame.time_s
    data.qpos[0] = frame.position_rad
    data.qvel[0] = frame.velocity_rad_s
    data.ctrl[0] = frame.applied_torque_nm
    mujoco.mj_forward(model, data)


def _apply_comparison_frame(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    p_frame: PlaybackFrame,
    pd_frame: PlaybackFrame,
) -> None:
    data.time = p_frame.time_s
    data.qpos[:] = (p_frame.position_rad, pd_frame.position_rad)
    data.qvel[:] = (p_frame.velocity_rad_s, pd_frame.velocity_rad_s)
    data.ctrl[:] = (p_frame.applied_torque_nm, pd_frame.applied_torque_nm)
    mujoco.mj_forward(model, data)


def _set_controller_color(model: mujoco.MjModel, controller: ControllerName) -> None:
    arm_geom_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "arm_link",
    )
    if arm_geom_id < 0:
        raise ValueError("model does not define the arm_link geometry")
    model.geom_rgba[arm_geom_id] = ARM_COLORS[controller]


def _set_overlay(
    viewer: ViewerHandle,
    config: ExperimentConfig,
    controller: ControllerName,
    frame: PlaybackFrame,
    phase: str,
) -> None:
    title, lesson = controller_description(config, controller)
    position_error = frame.target_position_rad - frame.position_rad
    saturation = "YES" if frame.saturated else "no"

    viewer.set_texts(
        [
            (
                mujoco.mjtFontScale.mjFONTSCALE_150,
                mujoco.mjtGridPos.mjGRID_TOPLEFT,
                "MOTION LAB\n" + title,
                f"{phase}\nRecorded deterministic trajectory",
            ),
            (
                mujoco.mjtFontScale.mjFONTSCALE_150,
                mujoco.mjtGridPos.mjGRID_TOPRIGHT,
                "Time\nPosition\nTarget\nError\nVelocity\nTorque\nSaturated",
                (
                    f"{frame.time_s:5.2f} s\n"
                    f"{frame.position_rad:+6.3f} rad\n"
                    f"{frame.target_position_rad:+6.3f} rad\n"
                    f"{position_error:+6.3f} rad\n"
                    f"{frame.velocity_rad_s:+6.3f} rad/s\n"
                    f"{frame.applied_torque_nm:+6.3f} N m\n"
                    f"{saturation}"
                ),
            ),
            (
                mujoco.mjtFontScale.mjFONTSCALE_150,
                mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                lesson,
                "Orange = P   |   Blue = PD   |   Red = target",
            ),
        ]
    )


def _set_comparison_overlay(
    viewer: ViewerHandle,
    config: ExperimentConfig,
    p_frame: PlaybackFrame,
    pd_frame: PlaybackFrame,
    phase: str,
) -> None:
    p_title, p_lesson = controller_description(config, "p")
    pd_title, pd_lesson = controller_description(config, "pd")
    viewer.set_texts(
        [
            _controller_text_box(
                mujoco.mjtGridPos.mjGRID_TOPLEFT,
                p_title,
                p_lesson,
                p_frame,
                phase,
            ),
            _controller_text_box(
                mujoco.mjtGridPos.mjGRID_TOPRIGHT,
                pd_title,
                pd_lesson,
                pd_frame,
                phase,
            ),
            (
                mujoco.mjtFontScale.mjFONTSCALE_150,
                mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                "FAIR COMPARISON",
                "Same start | target | torque limit | clock",
            ),
            (
                mujoco.mjtFontScale.mjFONTSCALE_150,
                mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT,
                "COLOR KEY",
                "Orange = P | Blue = PD | Red = target",
            ),
        ]
    )


def _controller_text_box(
    grid_position: object,
    title: str,
    lesson: str,
    frame: PlaybackFrame,
    phase: str,
) -> tuple[object, object, str, str]:
    error = frame.target_position_rad - frame.position_rad
    labels = (
        f"{title}\n"
        f"{lesson}\n\n"
        "Time\nPosition\nError\nVelocity\nTorque\nSaturated"
    )
    values = (
        f"{phase}\n\n\n"
        f"{frame.time_s:5.2f} s\n"
        f"{frame.position_rad:+6.3f} rad\n"
        f"{error:+6.3f} rad\n"
        f"{frame.velocity_rad_s:+6.3f} rad/s\n"
        f"{frame.applied_torque_nm:+6.3f} N m\n"
        f"{'YES' if frame.saturated else 'no'}"
    )
    return (
        mujoco.mjtFontScale.mjFONTSCALE_150,
        grid_position,
        labels,
        values,
    )


def _validate_comparison_model(model: mujoco.MjModel) -> None:
    if (model.nq, model.nv, model.nu) != (2, 2, 2):
        raise ValueError(
            "comparison display requires two positions, velocities, and actuators"
        )


def _hold_viewer(viewer: ViewerHandle, duration_s: float) -> bool:
    deadline = time.perf_counter() + duration_s
    while time.perf_counter() < deadline:
        if not viewer.is_running():
            return False
        time.sleep(min(1.0 / 60.0, max(0.0, deadline - time.perf_counter())))
    return viewer.is_running()


def _wait_until(viewer: ViewerHandle, deadline: float) -> bool:
    while True:
        if not viewer.is_running():
            return False
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            return True
        time.sleep(min(remaining, 1.0 / 120.0))
