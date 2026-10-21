"""Deterministic MP4 export for the side-by-side controller comparison."""

from bisect import bisect_right
from dataclasses import dataclass
import math
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from feedback_control.config import ExperimentConfig
from feedback_control.playback import PlaybackFrame, build_playback_frames
from feedback_control.simulation import EpisodeResult


@dataclass(frozen=True)
class RecordingOptions:
    """Video settings that do not alter the recorded experiment."""

    fps: int = 30
    width: int = 1280
    height: int = 720
    intro_seconds: float = 2.0
    hold_seconds: float = 2.0


@dataclass(frozen=True)
class RecordingFonts:
    title: ImageFont.FreeTypeFont
    subtitle: ImageFont.FreeTypeFont
    metric: ImageFont.FreeTypeFont
    footer: ImageFont.FreeTypeFont


def sample_playback_frame(
    frames: tuple[PlaybackFrame, ...],
    time_s: float,
) -> PlaybackFrame:
    """Sample recorded states at video time without inventing new actions.

    Position and velocity use linear interpolation for display. Applied torque
    and saturation use zero-order hold because the real controller holds each
    selected action until the next control decision.
    """
    if not frames:
        raise ValueError("cannot sample an empty trajectory")
    if not math.isfinite(time_s):
        raise ValueError("video sample time must be finite")
    if time_s <= frames[0].time_s:
        return frames[0]
    if time_s >= frames[-1].time_s:
        return frames[-1]

    frame_times = [frame.time_s for frame in frames]
    left_index = bisect_right(frame_times, time_s) - 1
    left = frames[left_index]
    right = frames[left_index + 1]
    interval = right.time_s - left.time_s
    if interval <= 0:
        raise ValueError("trajectory times must be strictly increasing")
    fraction = (time_s - left.time_s) / interval

    return PlaybackFrame(
        time_s=time_s,
        position_rad=_lerp(left.position_rad, right.position_rad, fraction),
        velocity_rad_s=_lerp(
            left.velocity_rad_s,
            right.velocity_rad_s,
            fraction,
        ),
        target_position_rad=_lerp(
            left.target_position_rad,
            right.target_position_rad,
            fraction,
        ),
        applied_torque_nm=left.applied_torque_nm,
        saturated=left.saturated,
    )


def record_comparison_video(
    *,
    config: ExperimentConfig,
    p_result: EpisodeResult,
    pd_result: EpisodeResult,
    comparison_model_path: Path,
    output_path: Path,
    options: RecordingOptions = RecordingOptions(),
) -> Path:
    """Render the synchronized P/PD comparison to an H.264 MP4."""
    _validate_recording_options(options)
    if p_result.controller != "p" or pd_result.controller != "pd":
        raise ValueError("recording requires P results followed by PD results")
    if not math.isclose(
        p_result.final_time_s,
        pd_result.final_time_s,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError("recording trajectories must have the same duration")

    p_frames = build_playback_frames(p_result)
    pd_frames = build_playback_frames(pd_result)
    fonts = _load_fonts()
    model = mujoco.MjModel.from_xml_path(str(comparison_model_path))
    if (model.nq, model.nv, model.nu) != (2, 2, 2):
        raise ValueError("recording model must contain two display arms")
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(
        model,
        height=options.height,
        width=options.width,
    )

    resolved_output = output_path.resolve()
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    intro_count = round(options.intro_seconds * options.fps)
    motion_count = math.ceil(p_result.final_time_s * options.fps)
    hold_count = round(options.hold_seconds * options.fps)
    total_count = intro_count + motion_count + hold_count

    try:
        with imageio.get_writer(
            resolved_output,
            fps=options.fps,
            codec="libx264",
            quality=8,
            macro_block_size=None,
            ffmpeg_log_level="error",
        ) as writer:
            for output_index in range(total_count):
                if output_index < intro_count:
                    sample_time = 0.0
                    phase = "STARTING"
                elif output_index < intro_count + motion_count:
                    motion_index = output_index - intro_count
                    sample_time = motion_index / options.fps
                    phase = "RUNNING"
                else:
                    sample_time = p_result.final_time_s
                    phase = "COMPLETE"

                p_frame = sample_playback_frame(p_frames, sample_time)
                pd_frame = sample_playback_frame(pd_frames, sample_time)
                _apply_comparison_state(model, data, p_frame, pd_frame)
                renderer.update_scene(data, camera="comparison_camera")
                rendered = renderer.render().copy()
                progress = min(1.0, sample_time / p_result.final_time_s)
                writer.append_data(
                    compose_comparison_frame(
                        rendered,
                        config=config,
                        p_frame=p_frame,
                        pd_frame=pd_frame,
                        phase=phase,
                        progress=progress,
                        fonts=fonts,
                    )
                )
    finally:
        renderer.close()

    return resolved_output


def compose_comparison_frame(
    rendered: np.ndarray,
    *,
    config: ExperimentConfig,
    p_frame: PlaybackFrame,
    pd_frame: PlaybackFrame,
    phase: str,
    progress: float,
    fonts: RecordingFonts | None = None,
) -> np.ndarray:
    """Add controller metrics and fair-comparison context to one RGB frame."""
    frame = Image.fromarray(rendered).convert("RGBA")
    overlay = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    fonts = fonts or _load_fonts()
    width, height = frame.size

    margin = 24
    gap = 60
    panel_width = (width - 2 * margin - gap) // 2
    panel_height = 240
    left_box = (margin, 20, margin + panel_width, 20 + panel_height)
    right_x = margin + panel_width + gap
    right_box = (right_x, 20, right_x + panel_width, 20 + panel_height)

    _draw_controller_panel(
        draw,
        left_box,
        title=f"P CONTROLLER   Kp={config.p.kp:.2f}",
        lesson="Position error only",
        frame=p_frame,
        phase=phase,
        accent=(250, 139, 35, 255),
        fonts=fonts,
    )
    _draw_controller_panel(
        draw,
        right_box,
        title=f"PD CONTROLLER   Kp={config.pd.kp:.2f}   Kd={config.pd.kd:.2f}",
        lesson="Velocity feedback adds braking",
        frame=pd_frame,
        phase=phase,
        accent=(33, 172, 245, 255),
        fonts=fonts,
    )

    footer_top = height - 64
    draw.rounded_rectangle(
        (margin, footer_top, width - margin, height - 18),
        radius=12,
        fill=(5, 10, 18, 218),
        outline=(116, 130, 150, 180),
        width=1,
    )
    draw.text(
        (margin + 18, footer_top + 13),
        "SAME START | SAME TARGET | SAME TORQUE LIMIT | SAME CLOCK",
        font=fonts.footer,
        fill=(224, 231, 240, 255),
    )
    legend = "ORANGE: P    BLUE: PD    RED: TARGET"
    legend_box = draw.textbbox((0, 0), legend, font=fonts.footer)
    legend_width = legend_box[2] - legend_box[0]
    draw.text(
        (width - margin - 18 - legend_width, footer_top + 13),
        legend,
        font=fonts.footer,
        fill=(224, 231, 240, 255),
    )

    progress_y = footer_top - 10
    draw.rounded_rectangle(
        (margin, progress_y, width - margin, progress_y + 4),
        radius=2,
        fill=(65, 76, 94, 220),
    )
    progress_width = round((width - 2 * margin) * max(0.0, min(1.0, progress)))
    if progress_width:
        draw.rounded_rectangle(
            (margin, progress_y, margin + progress_width, progress_y + 4),
            radius=2,
            fill=(235, 74, 65, 255),
        )

    return np.asarray(Image.alpha_composite(frame, overlay).convert("RGB"))


def _draw_controller_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    *,
    title: str,
    lesson: str,
    frame: PlaybackFrame,
    phase: str,
    accent: tuple[int, int, int, int],
    fonts: RecordingFonts,
) -> None:
    left, top, right, bottom = box
    draw.rounded_rectangle(
        box,
        radius=16,
        fill=(5, 10, 18, 218),
        outline=accent,
        width=3,
    )
    draw.text((left + 20, top + 14), title, font=fonts.title, fill=accent)
    draw.text(
        (left + 20, top + 51),
        lesson,
        font=fonts.subtitle,
        fill=(198, 208, 221, 255),
    )
    phase_box = draw.textbbox((0, 0), phase, font=fonts.subtitle)
    phase_width = phase_box[2] - phase_box[0]
    draw.text(
        (right - 20 - phase_width, top + 55),
        phase,
        font=fonts.subtitle,
        fill=(144, 157, 176, 255),
    )

    error = frame.target_position_rad - frame.position_rad
    rows = (
        ("Time", f"{frame.time_s:5.2f} s"),
        ("Position", f"{frame.position_rad:+6.3f} rad"),
        ("Error", f"{error:+6.3f} rad"),
        ("Velocity", f"{frame.velocity_rad_s:+6.3f} rad/s"),
        ("Torque", f"{frame.applied_torque_nm:+6.3f} N m"),
        ("Saturated", "YES" if frame.saturated else "no"),
    )
    row_y = top + 86
    for label, value in rows:
        draw.text(
            (left + 20, row_y),
            label,
            font=fonts.metric,
            fill=(180, 192, 207, 255),
        )
        value_box = draw.textbbox((0, 0), value, font=fonts.metric)
        value_width = value_box[2] - value_box[0]
        value_color = (
            (255, 99, 92, 255)
            if label == "Saturated" and value == "YES"
            else (242, 246, 250, 255)
        )
        draw.text(
            (right - 20 - value_width, row_y),
            value,
            font=fonts.metric,
            fill=value_color,
        )
        row_y += 24


def _apply_comparison_state(
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


def _load_fonts() -> RecordingFonts:
    # Use Matplotlib's bundled font directly. Font discovery scans every system
    # font on first run and can make a simple recording command appear hung.
    font_path = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"
    return RecordingFonts(
        title=ImageFont.truetype(font_path, 27),
        subtitle=ImageFont.truetype(font_path, 17),
        metric=ImageFont.truetype(font_path, 18),
        footer=ImageFont.truetype(font_path, 16),
    )


def _validate_recording_options(options: RecordingOptions) -> None:
    if options.fps <= 0:
        raise ValueError("fps must be greater than zero")
    if options.width <= 0 or options.height <= 0:
        raise ValueError("recording dimensions must be greater than zero")
    if options.width % 2 or options.height % 2:
        raise ValueError("H.264 recording dimensions must be even")
    for name, value in (
        ("intro_seconds", options.intro_seconds),
        ("hold_seconds", options.hold_seconds),
    ):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")


def _lerp(start: float, end: float, fraction: float) -> float:
    return start + (end - start) * fraction
