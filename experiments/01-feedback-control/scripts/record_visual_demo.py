"""Export the side-by-side P versus PD demonstration as an MP4."""

import argparse
from pathlib import Path

from feedback_control.config import load_experiment_config
from feedback_control.recording import RecordingOptions, record_comparison_video
from feedback_control.simulation import run_episode


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = EXPERIMENT_DIR / "configs" / "demo.toml"
COMPARISON_MODEL = EXPERIMENT_DIR / "models" / "one_link_comparison.xml"
DEFAULT_OUTPUT = EXPERIMENT_DIR / "results" / "p_vs_pd_side_by_side.mp4"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fps", type=int, default=30)
    arguments = parser.parse_args()

    config = load_experiment_config(arguments.config)
    p_result = run_episode(config, "p")
    pd_result = run_episode(config, "pd")
    output = record_comparison_video(
        config=config,
        p_result=p_result,
        pd_result=pd_result,
        comparison_model_path=COMPARISON_MODEL,
        output_path=arguments.output,
        options=RecordingOptions(fps=arguments.fps),
    )
    size_mb = output.stat().st_size / (1024 * 1024)
    print(f"Recorded: {output}")
    print(f"Size:     {size_mb:.2f} MiB")


if __name__ == "__main__":
    main()
