"""Controller-independent receding-horizon truth execution."""
from dataclasses import dataclass, field
import time

from feedback_control.controllers import pd_control
from forward_dynamics.types import validate_finite_scalar
from model_predictive_control.contracts import (
    CONTROL_INTERVAL_S, ControlDecision, EpisodeRecord, StepRecord, TORQUE_LIMIT_NM, POSITION_LIMIT_RAD,
)


@dataclass(frozen=True)
class PDController:
    kp: float
    kd: float

    def __call__(self, state, target_position_rad):
        result = pd_control(target_position=target_position_rad, position=state.position_rad,
                            velocity=state.velocity_rad_s, kp=self.kp, kd=self.kd,
                            torque_limit=TORQUE_LIMIT_NM)
        return ControlDecision(result.raw_torque)


@dataclass
class IntegralFeedbackController:
    """Add bounded integral torque to PD for one observed episode."""

    pd: PDController
    integral_gain: float
    integral_limit_nm: float
    _integral_torque_nm: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self):
        validate_finite_scalar("integral gain", self.integral_gain)
        validate_finite_scalar("integral torque limit", self.integral_limit_nm)
        if self.integral_gain < 0 or not 0 <= self.integral_limit_nm <= TORQUE_LIMIT_NM:
            raise ValueError("integral gain and torque limit must be nonnegative and the limit must fit the actuator")

    def __call__(self, state, target_position_rad):
        base = self.pd(state, target_position_rad).requested_torque_nm
        error = float(target_position_rad) - float(state.position_rad)
        old_torque = self._integral_torque_nm
        limit = self.integral_limit_nm
        candidate = max(-limit, min(limit, old_torque + self.integral_gain * CONTROL_INTERVAL_S * error))
        combined = base + candidate
        if ((combined > TORQUE_LIMIT_NM and candidate > old_torque)
                or (combined < -TORQUE_LIMIT_NM and candidate < old_torque)):
            return ControlDecision(base + old_torque)
        self._integral_torque_nm = candidate
        return ControlDecision(combined)


def _valid_request(value):
    try:
        validate_finite_scalar("requested torque", value)
    except ValueError:
        return False
    return True


def run_episode(plant, scenario, controller, *, controller_id, seed, fallback,
                deadline_ns=None, clock_ns=time.perf_counter_ns):
    validate_finite_scalar("target_position_rad", scenario.target_position_rad)
    if abs(float(scenario.target_position_rad)) > POSITION_LIMIT_RAD:
        raise ValueError("target_position_rad exceeds joint limit")
    if scenario.max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if deadline_ns is not None and deadline_ns < 0:
        raise ValueError("deadline_ns must be nonnegative")
    state = plant.reset(scenario.initial_state)
    steps = []
    termination = "time_limit"
    for _ in range(scenario.max_steps):
        start = clock_ns()
        decision = controller(state, scenario.target_position_rad)
        elapsed = clock_ns() - start
        request = decision.requested_torque_nm
        valid = _valid_request(request)
        recorded = float(request) if valid else None
        missed = deadline_ns is not None and elapsed > deadline_ns
        reason = ("deadline" if missed else "no_plan" if request is None else
                  "invalid_command" if not valid else None)
        calls = decision.predictor_calls
        slots = decision.candidate_slots
        rejected = decision.rejected_candidates
        selected = recorded
        action_ready = elapsed
        if reason is not None:
            if fallback is None:
                raise ValueError("fallback must provide a finite torque")
            fallback_decision = fallback(state, scenario.target_position_rad)
            action_ready = clock_ns() - start
            if not isinstance(fallback_decision, ControlDecision):
                raise ValueError("fallback must provide a decision with finite torque")
            calls += fallback_decision.predictor_calls
            slots += fallback_decision.candidate_slots
            rejected += fallback_decision.rejected_candidates
            if not _valid_request(fallback_decision.requested_torque_nm):
                raise ValueError("fallback must provide a finite torque")
            selected = float(fallback_decision.requested_torque_nm)
        applied = max(-TORQUE_LIMIT_NM, min(TORQUE_LIMIT_NM, selected))
        plant_step = plant.advance(applied)
        steps.append(StepRecord(
            state=state, next_state=plant_step.next_state, requested_torque_nm=recorded,
            applied_torque_nm=applied, physics_steps=plant_step.physics_steps,
            decision_time_ns=elapsed, predictor_calls=calls, deadline_missed=missed,
            fallback_reason=reason, saturated=applied != selected,
            selected_torque_nm=selected, action_ready_time_ns=action_ready,
            candidate_slots=slots, rejected_candidates=rejected,
        ))
        if plant_step.termination is not None:
            termination = plant_step.termination
            break
        state = plant_step.next_state
    return EpisodeRecord(scenario, controller_id, seed, tuple(steps), termination)
