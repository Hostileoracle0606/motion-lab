"""Persist and audit the ordered inventory of episode attempts."""

import json
from pathlib import Path


def identity(scenario_id, controller_id, seed):
    return {"scenario_id": scenario_id, "controller_id": controller_id, "seed": seed}


def write_roster(output: Path, roster: list[dict]) -> None:
    with (output / "planned-attempts.json").open("x", encoding="utf-8") as stream:
        json.dump({"schema_version": 1, "attempts": roster}, stream, allow_nan=False, indent=2)
        stream.write("\n")


def record_started(output: Path, attempt: dict) -> None:
    with (output / "started-attempts.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"schema_version": 1, "attempt": attempt}, allow_nan=False) + "\n")


def _valid_identity(value):
    return (isinstance(value, dict) and set(value) == {"scenario_id", "controller_id", "seed"}
            and all(isinstance(value[k], str) and value[k] for k in ("scenario_id", "controller_id"))
            and type(value["seed"]) is int)


def audit_run(output: Path, *, require_complete: bool = True) -> dict:
    """Audit ordered evidence; forensic mode reports errors without accepting it as final."""
    errors = []
    truncated = {"started": False, "completed": False}

    def load_json(name):
        try:
            return json.loads((output / name).read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError) as exc:
            errors.append(f"{name}: {exc}")
            return {}

    roster_doc = load_json("planned-attempts.json")
    status_doc = load_json("run-status.json")
    if not isinstance(roster_doc, dict) or roster_doc.get("schema_version") != 1 or not isinstance(roster_doc.get("attempts"), list):
        errors.append("malformed planned roster")
        planned = []
    else:
        planned = roster_doc["attempts"]
    if not planned or any(not _valid_identity(x) for x in planned):
        errors.append("malformed or empty planned identities")
    if len({json.dumps(x, sort_keys=True) for x in planned}) != len(planned):
        errors.append("duplicate planned identities")

    def journal(name, kind):
        path = output / name
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            return []
        except OSError as exc:
            errors.append(f"{name}: {exc}")
            return []
        lines = data.splitlines(keepends=True)
        if lines and not lines[-1].endswith(b"\n"):
            truncated[kind] = True
            lines.pop()
        values = []
        for line in lines:
            try:
                row = json.loads(line)
                if not isinstance(row, dict) or row.get("schema_version") != 1:
                    raise ValueError("invalid schema")
                if kind == "started":
                    value = row["attempt"]
                else:
                    episode = row["episode"]
                    if (not isinstance(episode, dict) or not isinstance(episode.get("steps"), list)
                            or not isinstance(episode.get("termination"), str)):
                        raise ValueError("invalid completed episode")
                    value = identity(episode["scenario"]["scenario_id"], episode["controller_id"], episode["seed"])
                if not _valid_identity(value):
                    raise ValueError("invalid identity")
                values.append(value)
            except (ValueError, KeyError, TypeError, UnicodeError) as exc:
                errors.append(f"{name}: malformed record: {exc}")
        return values

    started = journal("started-attempts.jsonl", "started")
    completed = journal("completed-episodes.jsonl", "completed")
    for label, values, reference in (("started", started, planned), ("completed", completed, started)):
        if values != reference[:len(values)]:
            errors.append(f"{label} identities are not an ordered prefix")
        if len({json.dumps(x, sort_keys=True) for x in values}) != len(values):
            errors.append(f"duplicate {label} identities")
    if len(started) - len(completed) > 1:
        errors.append("more than one interrupted attempt")
    if not isinstance(status_doc, dict) or status_doc.get("schema_version") != 1:
        errors.append("malformed run status")
        status_doc = {}
    if status_doc.get("status") not in ("in_progress", "aborted", "complete"):
        errors.append("invalid run status")
    if type(status_doc.get("expected_episodes")) is not int or status_doc["expected_episodes"] != len(planned):
        errors.append("expected episode count disagrees with roster")
    if ((status_doc.get("status") in ("complete", "aborted") or "completed_episodes" in status_doc)
            and (type(status_doc.get("completed_episodes")) is not int
                 or status_doc["completed_episodes"] != len(completed))):
        errors.append("completed episode count disagrees with journal")
    result = {"status": status_doc.get("status"), "planned": planned, "started": started,
              "completed": completed, "interrupted": started[len(completed):],
              "missing": planned[len(started):], "truncated": truncated, "errors": errors}
    if require_complete and (errors or any(truncated.values()) or result["status"] != "complete"
                             or started != planned or completed != planned):
        raise ValueError(f"run is not complete final evidence: {json.dumps(result)}")
    return result
