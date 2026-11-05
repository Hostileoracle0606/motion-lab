"""Physical-unit CSV views, preserving empty episode/horizon coverage."""

import csv
from dataclasses import fields
from pathlib import Path

from forward_dynamics.evidence import EvaluationEvidence
from forward_dynamics.metrics import ErrorSummary


METRICS = tuple(field.name for field in fields(ErrorSummary))


def table_rows(evidence: EvaluationEvidence, mode: str) -> tuple[dict, ...]:
    """Return all grouping summaries with predictor-independent coverage counts."""
    if mode not in ("one_step", "rollouts"):
        raise ValueError("unknown table mode")
    rollout = mode == "rollouts"
    source = evidence.evaluation["rollout_aggregates" if rollout else "aggregates"]
    rows = [{key: value for key, value in row.items() if key != "summary"} | row["summary"]
            for row in source]
    if rollout:
        outcomes = {row["episode_id"]: row for row in evidence.evaluation["episode_outcomes"]}
        groups = {}
        for window in evidence.windows:
            episode = outcomes[window.episode_id]
            for group, field in (("episode", "episode_id"), ("scenario", "scenario_id"),
                                 ("controller", "controller_id"), ("split", "split")):
                key = (group, window.split, episode[field], window.horizon_steps)
                eligible, unavailable = groups.get(key, (0, 0))
                groups[key] = (eligible + window.eligible_count, unavailable + window.unavailable_count)
        indexed = {(row["predictor_id"], row["group"], row["split"], row["key"], row["horizon_steps"]): row
                   for row in rows}
        for name in evidence.evaluation["protocol"]["predictor_ids"]:
            for key, (eligible, unavailable) in groups.items():
                full_key = (name, *key)
                if full_key not in indexed:
                    group, split, value, horizon = key
                    row = dict(predictor_id=name, group=group, split=split, key=value, horizon_steps=horizon)
                    row.update({metric: None for metric in METRICS})
                    row["sample_count"] = 0
                    indexed[full_key] = row
                indexed[full_key].update(eligible_count=eligible, unavailable_count=unavailable)
        rows = list(indexed.values())
    else:
        for row in rows:
            row.update(horizon_steps=1, eligible_count=row["sample_count"], unavailable_count=0)
    return tuple(sorted(rows, key=lambda row: (row["predictor_id"], row["group"], row["split"], row["key"], row["horizon_steps"])))


def write_tables(evidence: EvaluationEvidence, output: Path) -> None:
    names = ("predictor_id", "group", "split", "key", "horizon_steps", "eligible_count", "unavailable_count", *METRICS)
    for mode in ("one_step", "rollouts"):
        with (output / f"{mode}.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=names, lineterminator="\n")
            writer.writeheader()
            writer.writerows(table_rows(evidence, mode))
