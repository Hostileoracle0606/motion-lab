"""Immutable, checked data boundaries; no training framework is required."""

from collections import defaultdict
from dataclasses import dataclass
import math
import re
from typing import Literal

from forward_dynamics.dataset import PredictionCase
from forward_dynamics.types import validate_prediction_inputs, validate_state
from forward_dynamics.types import validate_finite_scalar


Split = Literal["train", "validation", "test"]
DATASET_VERSION = "experiment-02-transition-v1"
CONSTANT_THRESHOLD = 1e-12


def validate_hash(value: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("sha256 must be 64 lowercase hexadecimal characters")


def checked_episodes(cases: tuple[PredictionCase, ...]) -> tuple[tuple[PredictionCase, ...], ...]:
    """Check contiguous episode slices, without discarding terminal states."""
    if not isinstance(cases, tuple) or not cases:
        raise ValueError("cases must be a nonempty tuple")
    grouped = defaultdict(list)
    for case in cases:
        if not isinstance(case, PredictionCase):
            raise ValueError("cases must contain PredictionCase values")
        if case.split not in ("train", "validation", "test"):
            raise ValueError("unsupported case split")
        for field in ("episode_id", "scenario_id", "controller_id"):
            value = getattr(case, field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field} must be nonempty")
        for field in ("step_index", "seed"):
            value = getattr(case, field)
            if type(value) is not int or value < 0:
                raise ValueError(f"{field} must be a nonnegative integer")
        if type(case.terminated) is not bool or type(case.truncated) is not bool:
            raise ValueError("boundary flags must be booleans")
        if case.terminated and case.truncated:
            raise ValueError("boundary cannot be both terminated and truncated")
        boundary = case.terminated or case.truncated
        if ((boundary and (not isinstance(case.termination_reason, str) or
                          not case.termination_reason.strip())) or
                (not boundary and case.termination_reason is not None)):
            raise ValueError("termination reason must agree with boundary flags")
        validate_prediction_inputs(case.state, case.applied_torque_nm, case.context)
        validate_state(case.expected_next_state)
        grouped[case.episode_id].append(case)
    episodes = []
    for name in sorted(grouped):
        rows = tuple(sorted(grouped[name], key=lambda c: c.step_index))
        first = rows[0]
        for i, case in enumerate(rows):
            if any(getattr(case, field) != getattr(first, field) for field in (
                "split", "scenario_id", "controller_id", "seed", "context",
            )):
                raise ValueError("episode identity/context must remain consistent")
            if case.step_index != first.step_index + i:
                raise ValueError("episode indices must be unique and contiguous")
            if i:
                previous = rows[i - 1]
                if previous.terminated or previous.truncated:
                    raise ValueError("episode crosses a terminal boundary")
                for field in ("position_rad", "velocity_rad_s"):
                    if not math.isclose(float(getattr(previous.expected_next_state, field)),
                                        float(getattr(case.state, field)),
                                        abs_tol=1e-12, rel_tol=0):
                        raise ValueError("episode state continuity is broken")
        episodes.append(rows)
    return tuple(episodes)


@dataclass(frozen=True)
class SplitView:
    split: Split
    cases: tuple[PredictionCase, ...]
    dataset_version: str
    manifest_sha256: str
    train_sha256: str

    def __post_init__(self) -> None:
        if self.split not in ("train", "validation", "test"):
            raise ValueError("unsupported split")
        if self.dataset_version != DATASET_VERSION:
            raise ValueError("unsupported dataset version")
        validate_hash(self.manifest_sha256)
        validate_hash(self.train_sha256)
        episodes = checked_episodes(self.cases)
        if any(c.split != self.split for rows in episodes for c in rows):
            raise ValueError("view contains mixed split membership")
        object.__setattr__(self, "cases", tuple(c for rows in episodes for c in rows))


@dataclass(frozen=True)
class TrainingView:
    view: SplitView

    def __post_init__(self) -> None:
        if not isinstance(self.view, SplitView) or self.view.split != "train":
            raise ValueError("TrainingView requires a train SplitView")

    @property
    def cases(self) -> tuple[PredictionCase, ...]:
        return self.view.cases


@dataclass(frozen=True)
class DataViews:
    train: TrainingView
    validation: SplitView
    test: SplitView

    def __post_init__(self) -> None:
        if not isinstance(self.train, TrainingView):
            raise ValueError("train must be a TrainingView")
        seen_episodes, seen_scenarios = set(), set()
        identity = (self.train.view.dataset_version, self.train.view.manifest_sha256,
                    self.train.view.train_sha256)
        for expected, view in (("train", self.train.view), ("validation", self.validation),
                               ("test", self.test)):
            if not isinstance(view, SplitView) or view.split != expected:
                raise ValueError("DataViews requires matching split views")
            if (view.dataset_version, view.manifest_sha256, view.train_sha256) != identity:
                raise ValueError("views must share dataset provenance")
            episodes = {c.episode_id for c in view.cases}
            scenarios = {c.scenario_id for c in view.cases}
            if seen_episodes & episodes or seen_scenarios & scenarios:
                raise ValueError("episode/scenario leakage across splits")
            seen_episodes.update(episodes)
            seen_scenarios.update(scenarios)


FEATURE_ORDER = ("position_rad", "velocity_rad_s", "applied_torque_nm")
FEATURE_UNITS = ("rad", "rad/s", "N m")


@dataclass(frozen=True)
class Normalizer:
    schema_version: int
    feature_order: tuple[str, ...]
    feature_units: tuple[str, ...]
    input_mean: tuple[float, ...]
    input_scale: tuple[float, ...]
    input_constant: tuple[bool, ...]
    delta_mean: tuple[float, float]
    delta_scale: tuple[float, float]
    delta_constant: tuple[bool, bool]
    fit_row_count: int
    dataset_version: str
    manifest_sha256: str
    train_sha256: str
    fit_keys: tuple[tuple[str, int], ...]
    target_transform: str = "next_minus_current"
    statistics: str = "float64_population_std"

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported normalizer schema version")
        if self.feature_order != FEATURE_ORDER or self.feature_units != FEATURE_UNITS:
            raise ValueError("normalizer feature order/units do not match contract")
        if self.dataset_version != DATASET_VERSION:
            raise ValueError("unsupported normalizer dataset version")
        if (self.target_transform != "next_minus_current" or
                self.statistics != "float64_population_std"):
            raise ValueError("unsupported target transform/statistics")
        validate_hash(self.manifest_sha256)
        validate_hash(self.train_sha256)
        for field, length in (("input_mean", 3), ("input_scale", 3),
                              ("delta_mean", 2), ("delta_scale", 2)):
            values = getattr(self, field)
            if not isinstance(values, tuple) or len(values) != length:
                raise ValueError(f"{field} must be an immutable length-{length} tuple")
            for value in values:
                validate_finite_scalar(field, value)
                if field.endswith("scale") and value <= 0:
                    raise ValueError("normalizer scales must be positive")
                if field.endswith("scale") and value < CONSTANT_THRESHOLD:
                    raise ValueError("normalizer scale is below the constant-feature threshold")
        for prefix, length in (("input", 3), ("delta", 2)):
            flags = getattr(self, prefix + "_constant")
            if (not isinstance(flags, tuple) or len(flags) != length or
                    any(type(f) is not bool for f in flags)):
                raise ValueError("constant flags must be immutable boolean tuples")
            if any(flag and scale != 1.0 for flag, scale in
                   zip(flags, getattr(self, prefix + "_scale"), strict=True)):
                raise ValueError("constant features require scale 1")
        if type(self.fit_row_count) is not int or self.fit_row_count <= 0:
            raise ValueError("fit_row_count must be a positive integer")
        if not isinstance(self.fit_keys, tuple) or len(self.fit_keys) != self.fit_row_count:
            raise ValueError("fit identities must match fit_row_count")
        for key in self.fit_keys:
            if (not isinstance(key, tuple) or len(key) != 2 or
                    not isinstance(key[0], str) or not key[0].strip() or
                    type(key[1]) is not int or key[1] < 0):
                raise ValueError("invalid fit episode/step identity")
        if tuple(sorted(set(self.fit_keys))) != self.fit_keys:
            raise ValueError("fit identities must be unique and ordered")


@dataclass(frozen=True)
class Window:
    cases: tuple[PredictionCase, ...]
    previous: PredictionCase | None


@dataclass(frozen=True)
class WindowCoverage:
    split: Split
    episode_id: str
    horizon: int
    eligible: int
    unavailable: int
    missing_history: int


@dataclass(frozen=True)
class ModelSpec:
    """The two widths and explicit history mode declared for experiment v1."""
    hidden_width: int = 32
    history_steps: int = 0

    def __post_init__(self) -> None:
        if type(self.hidden_width) is not int or self.hidden_width not in (16, 32):
            raise ValueError("hidden_width must be 16 or 32 for model v1")
        if type(self.history_steps) is not int or self.history_steps not in (0, 1):
            raise ValueError("history_steps must be zero or one")
