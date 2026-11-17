"""Pure report-table derivations from independently audited evidence."""

from collections import defaultdict
import math
import statistics


METRICS = (
    "position_mae_rad", "position_rmse_rad", "position_max_abs_rad",
    "velocity_mae_rad_s", "velocity_rmse_rad_s", "velocity_max_abs_rad_s",
)


def aggregate_seed_statistics(rows: tuple[dict, ...]) -> tuple[dict, ...]:
    if not isinstance(rows, tuple) or not rows:
        raise ValueError("seed aggregation requires nonempty immutable rows")
    grouped = defaultdict(list)
    for row in rows:
        seed = row.get("training_seed")
        if type(seed) is not int or seed not in range(5):
            raise ValueError("learned seed rows require declared seeds 0..4")
        grouped[(row["variant_id"], row["membership"], row["horizon_steps"])].append(row)
    output = []
    for (variant, membership, horizon), group in sorted(grouped.items()):
        seeds = sorted({row["training_seed"] for row in group})
        if len(seeds) != len(group):
            raise ValueError("duplicate seed summary row")
        missing = sorted(set(range(5)) - set(seeds))
        result = {
            "variant_id": variant, "membership": membership, "horizon_steps": horizon,
            "seed_count": len(seeds), "missing_seeds": missing,
            "failed_windows": sum(row["counts"]["failed"] for row in group),
            "unavailable_windows": sum(row["counts"]["unavailable"] for row in group),
            "eligible_windows": sum(row["counts"]["eligible"] for row in group),
        }
        metric_complete = True
        for metric in METRICS:
            values = [float(row["success_metrics"][metric]) for row in group
                      if row["success_metrics"] is not None]
            metric_complete = metric_complete and len(values) == 5
            result[f"{metric}_mean"] = statistics.fmean(values) if values else None
            result[f"{metric}_std"] = statistics.stdev(values) if len(values) >= 2 else None
        result["complete_seed_matrix"] = not missing and metric_complete
        output.append(result)
    return tuple(output)


def report_summary_rows(evidence: dict) -> tuple[dict, ...]:
    identities = {}
    for record in evidence["records"]:
        spec = record["candidate"]["run_spec"]
        if spec is not None:
            identities[record["predictor_id"]] = (spec["variant_id"], spec["seed"])
    rows = []
    for summary in evidence["summaries"]:
        if summary["predictor_id"] not in identities:
            continue
        variant, seed = identities[summary["predictor_id"]]
        rows.append({
            "variant_id": variant, "membership": summary["membership"],
            "horizon_steps": summary["horizon_steps"], "training_seed": seed,
            "counts": summary["counts"], "success_metrics": summary["success_metrics"],
        })
    return tuple(rows)
