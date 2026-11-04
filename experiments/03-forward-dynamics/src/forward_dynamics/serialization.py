"""Deterministic finite JSON and strict saved-evidence decoding."""

import hashlib
import json
from pathlib import Path


def json_bytes(value: object, *, indent: int | None = None) -> bytes:
    try:
        text = json.dumps(value, sort_keys=True, allow_nan=False, indent=indent,
                          separators=(",", ":") if indent is None else None)
    except (ValueError, TypeError) as error:
        raise ValueError("evidence must contain finite JSON values") from error
    return (text + "\n").encode("utf-8")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decode_object(content: str) -> dict:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError(f"non-finite JSON constant: {value}")

    try:
        value = json.loads(content, object_pairs_hook=unique, parse_constant=reject_constant)
    except (ValueError, UnicodeError) as error:
        raise ValueError("invalid evidence JSON") from error
    if not isinstance(value, dict):
        raise ValueError("evidence JSON must be an object")
    # Also rejects overflowed JSON exponents such as 1e400.
    json_bytes(value)
    return value


def require_fields(value: object, names: set[str], description: str) -> None:
    if not isinstance(value, dict) or set(value) != names:
        raise ValueError(f"{description} fields do not match schema")


def require_equal(actual: object, expected: object, description: str) -> None:
    # JSON canonical comparison prevents Python's True == 1 from accepting
    # booleans as counts, schema versions or other numeric evidence.
    if json_bytes(actual) != json_bytes(expected):
        raise ValueError(f"{description} does not match validated inputs/derivations")
