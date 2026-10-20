"""Run one P or PD episode and print a compact state summary."""

import argparse
from pathlib import Path

from feedback_control.config import load_experiment_config
from feedback_control.simulation import run_episode


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = EXPERIMENT_DIR / "configs" / "demo.toml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", choices=("p", "pd"), required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    arguments = parser.parse_args()

    config = load_experiment_config(arguments.config)
    result = run_episode(config, arguments.controller)
    start = result.samples[0]
    saturated_steps = sum(sample.saturated for sample in result.samples)

    print(f"Controller:       {result.controller.upper()}")
    print(f"Start:            {start.position_rad:+.6f} rad")
    print(f"Target:           {start.target_position_rad:+.6f} rad")
    print(f"Final:            {result.final_position_rad:+.6f} rad")
    print(f"Final velocity:   {result.final_velocity_rad_s:+.6f} rad/s")
    print(f"Control samples:  {len(result.samples)}")
    print(f"Duration:         {result.final_time_s:.3f} s")
    print(f"Saturated steps:  {saturated_steps}")


if __name__ == "__main__":
    main()
