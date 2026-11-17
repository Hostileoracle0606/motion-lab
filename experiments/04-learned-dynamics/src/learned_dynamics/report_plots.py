"""Plot arrays and figures derived only from report-table rows."""

from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def horizon_series(rows, metric: str) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        value = row.get(metric)
        if value is not None:
            grouped[row["variant_id"]].append((row["horizon_steps"], float(value)))
    return {variant: {"horizons": [x for x, _ in sorted(values)],
                      "values": [y for _, y in sorted(values)]}
            for variant, values in sorted(grouped.items())}


def write_horizon_plot(rows, metric: str, ylabel: str, path: Path) -> dict:
    series = horizon_series(rows, metric)
    figure, axis = plt.subplots(figsize=(7.0, 4.5))
    for variant, values in series.items():
        axis.plot(values["horizons"], values["values"], marker="o", label=variant)
    axis.set_xlabel("Open-loop horizon (control steps)")
    axis.set_ylabel(ylabel)
    axis.set_xscale("log")
    axis.grid(True, alpha=0.3)
    if series:
        axis.legend(fontsize="small")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return series
