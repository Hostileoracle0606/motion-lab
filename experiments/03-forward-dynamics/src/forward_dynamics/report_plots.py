"""Pure plotted numeric data and zero-safe physical-unit rendering."""

from pathlib import Path

from forward_dynamics.evidence import EvaluationEvidence
from forward_dynamics.report_tables import table_rows


def plot_series(evidence: EvaluationEvidence) -> dict[str, list[dict]]:
    """Expose exactly the numbers/counts consumed by all four plots.

    Each plot contains one series per adapter/split. RMSE is transition/window
    weighted; missing horizons remain None, never invented zero errors.
    """
    result = {}
    for mode, prefix in (("one_step", "one_step"), ("rollouts", "rollout")):
        rows = table_rows(evidence, mode)
        for component, metric in (("position", "position_rmse_rad"), ("velocity", "velocity_rmse_rad_s")):
            values = []
            for name in evidence.evaluation["protocol"]["predictor_ids"]:
                for split in ("train", "validation", "test"):
                    selected = sorted((row for row in rows if row["group"] == "split" and
                                       row["predictor_id"] == name and row["split"] == split),
                                      key=lambda row: row["horizon_steps"])
                    values.append(dict(predictor_id=name, split=split,
                        horizons_steps=[row["horizon_steps"] for row in selected],
                        rmse=[row[metric] for row in selected],
                        sample_counts=[row["sample_count"] for row in selected]))
            result[f"{prefix}_{component}"] = values
    return result


def write_plots(evidence: EvaluationEvidence, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for name, series in plot_series(evidence).items():
        figure, axes = plt.subplots(1, 3, figsize=(17, 5), sharey=True)
        for axis, split in zip(axes, ("train", "validation", "test"), strict=True):
            selected = [row for row in series if row["split"] == split]
            if name.startswith("one_step"):
                labels = []
                for index, row in enumerate(selected):
                    value = row["rmse"][0]
                    axis.bar(index, value)
                    axis.scatter(index, value, color="black", s=12, zorder=3)
                    labels.append(f"{row['predictor_id']}\nn={row['sample_counts'][0]}")
                axis.set_xticks(range(len(selected)), labels, rotation=35, ha="right", fontsize=7)
                axis.set_xlabel("Predictor")
            else:
                for row in selected:
                    counts = ",".join(str(value) for value in row["sample_counts"])
                    axis.plot([step * 0.01 for step in row["horizons_steps"]], row["rmse"], marker="o",
                              label=f"{row['predictor_id']} (n={counts})")
                axis.set_xlabel("Horizon (s)")
                axis.legend(fontsize=6)
            axis.set_title(split)
            axis.grid(True, alpha=0.3)
        axes[0].set_ylabel("Position RMSE (rad)" if name.endswith("position") else "Velocity RMSE (rad/s)")
        figure.suptitle("Recorded-action open-loop prediction" if name.startswith("rollout") else "Observed-state one-step prediction")
        figure.tight_layout()
        figure.savefig(output / f"{name}.png", dpi=120)
        plt.close(figure)
