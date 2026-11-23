"""State, action and timing records for the nominal one-link pilot."""

from dataclasses import dataclass
from typing import Literal, Protocol

from forward_dynamics.mujoco_predictor import ENVIRONMENT_ID, MODEL_SHA256
from forward_dynamics.types import DynamicsContext, DynamicsState


PHYSICS_INTERVAL_S = 0.002
CONTROL_INTERVAL_S = 0.010
TORQUE_LIMIT_NM = 3.0
POSITION_LIMIT_RAD = 1.570796
NOMINAL_CONTEXT = DynamicsContext(
    ENVIRONMENT_ID, MODEL_SHA256, CONTROL_INTERVAL_S, TORQUE_LIMIT_NM,
)

Termination = Literal["time_limit", "joint_limit", "invalid_state"]
FallbackReason = Literal["no_plan", "deadline", "invalid_command"]


@dataclass(frozen=True)
class PlanResult:
    """Bounded proposed sequence, or None when every candidate was rejected."""

    actions_nm: tuple[float, ...] | None
    cost: float | None
    candidates: int
    feasible_candidates: int
    predictor_calls: int


@dataclass(frozen=True)
class ControlDecision:
    """One requested torque, or None when a controller cannot propose one."""

    requested_torque_nm: float | None
    predictor_calls: int = 0


class Controller(Protocol):
    def __call__(
        self, state: DynamicsState, target_position_rad: float,
    ) -> ControlDecision: ...


@dataclass(frozen=True)
class Scenario:
    """One fixed target and one independently initialized simulation episode."""

    scenario_id: str
    initial_state: DynamicsState
    target_position_rad: float
    max_steps: int


@dataclass(frozen=True)
class PlantStep:
    """State after held torque and actual number of completed physics steps."""

    next_state: DynamicsState | None
    physics_steps: int
    termination: Literal["joint_limit", "invalid_state"] | None = None


@dataclass(frozen=True)
class StepRecord:
    """One observed state, controller invocation and applied action interval."""

    state: DynamicsState
    next_state: DynamicsState | None
    requested_torque_nm: float | None
    applied_torque_nm: float
    physics_steps: int
    decision_time_ns: int
    predictor_calls: int
    deadline_missed: bool
    fallback_reason: FallbackReason | None
    saturated: bool


@dataclass(frozen=True)
class EpisodeRecord:
    """Chronological controller decisions for one scenario and run seed."""

    scenario: Scenario
    controller_id: str
    seed: int
    steps: tuple[StepRecord, ...]
    termination: Termination
