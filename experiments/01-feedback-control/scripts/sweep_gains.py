"""Run the declared P and PD gain sweep and print response diagnostics."""

import argparse
from pathlib import Path

from feedback_control.config import load_experiment_config
from feedback_control.tuning import (
    estimate_critical_derivative_gain,
    load_gain_sweep_config,
    nearest_candidate,
    run_gain_sweep,
)


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = EXPERIMENT_DIR / "configs" / "demo.toml"
DEFAULT_SWEEP = EXPERIMENT_DIR / "configs" / "gain_sweep.toml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--sweep", type=Path, default=DEFAULT_SWEEP)
    arguments = parser.parse_args()

    config = load_experiment_config(arguments.config)
    sweep = load_gain_sweep_config(arguments.sweep)
    critical_kd = estimate_critical_derivative_gain(config, kp=sweep.pd_kp)
    selected_kd = nearest_candidate(sweep.pd_kd_values, critical_kd)
    rows = run_gain_sweep(config, sweep)

    print("Critical-damping estimate")
    print(f"  fixed Kp:                {sweep.pd_kp:.3f}")
    print(f"  estimated Kd:            {critical_kd:.6f}")
    print(f"  nearest declared Kd:     {selected_kd:.3f}")
    print("  assumptions: continuous time, local linear inertia, no saturation")
    print()
    print(
        "Use Ctl   Kp    Kd  Err(rad)  Speed(rad/s)  Over(rad)  "
        "Peak(rad/s)  Sat  Cross"
    )
    for row in rows:
        use = ""
        if row.controller == "p" and row.kp == sweep.pd_kp:
            use = "base"
        elif row.controller == "pd" and row.kd == selected_kd:
            use = "demo"
        print(
            f"{use:4} {row.controller.upper():>3} "
            f"{row.kp:5.2f} {row.kd:5.2f} "
            f"{row.final_abs_error_rad:9.6f} "
            f"{row.final_abs_velocity_rad_s:10.6f} "
            f"{row.overshoot_rad:9.6f} "
            f"{row.max_abs_velocity_rad_s:8.4f} "
            f"{row.saturated_steps:4d} "
            f"{row.target_crossings:6d}"
        )


if __name__ == "__main__":
    main()
