#!/usr/bin/env python3
"""Physical-unit checks for both event locations in the staged solver."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np
from multi_stage_primer import (
    MultiStageCase,
    direct_propagate,
    make_case,
    physical_fixture_stages,
)
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from staged_two_arc_research import RTOL, PhysicalStage

Array = NDArray[np.float64]


@dataclass
class PhysicalTracker:
    stage_index: int = 0
    active_used: float = 0.0
    transition_count: int = 0


def physical_rhs(
    _time: float,
    state: Array,
    mu: float,
    thrust: float,
    exhaust_velocity: float,
    throttle: float,
    alpha: float,
) -> Array:
    radius, radial_velocity, tangential_velocity, mass = state
    if radius <= 0.0 or mass <= 0.0:
        return np.full(4, np.nan)
    acceleration = throttle * thrust / mass
    return np.array(
        [
            radial_velocity,
            tangential_velocity**2 / radius
            - mu / radius**2
            + acceleration * math.cos(alpha),
            -radial_velocity * tangential_velocity / radius
            + acceleration * math.sin(alpha),
            -throttle * thrust / exhaust_velocity,
        ],
        dtype=float,
    )


def integrate_physical(
    case: MultiStageCase,
    state: Array,
    duration: float,
    thrust: float,
    exhaust_velocity: float,
    throttle: float,
    alpha: float,
) -> Array:
    if duration == 0.0:
        return state.copy()
    solution = solve_ivp(
        physical_rhs,
        (0.0, duration),
        state,
        args=(case.mu, thrust, exhaust_velocity, throttle, alpha),
        method="DOP853",
        rtol=RTOL,
        atol=np.array([1e-6, 1e-9, 1e-9, 1e-8]),
    )
    if not solution.success or not np.all(np.isfinite(solution.y[:, -1])):
        raise ValueError(solution.message)
    return np.asarray(solution.y[:, -1], dtype=float)


def normalized_to_physical(
    case: MultiStageCase, state: Array, mass_scale: float
) -> Array:
    return np.array(
        [
            state[0] * case.target_radius,
            state[1] * case.velocity_scale,
            state[2] * case.velocity_scale,
            state[3] * mass_scale,
        ],
        dtype=float,
    )


def physical_to_normalized(
    case: MultiStageCase, state: Array, mass_scale: float
) -> Array:
    return np.array(
        [
            state[0] / case.target_radius,
            state[1] / case.velocity_scale,
            state[2] / case.velocity_scale,
            state[3] / mass_scale,
        ],
        dtype=float,
    )


def transition_physical(
    case: MultiStageCase,
    state: Array,
    tracker: PhysicalTracker,
    stages: tuple[PhysicalStage, PhysicalStage],
) -> Array:
    current = stages[tracker.stage_index]
    following = stages[tracker.stage_index + 1]
    state[3] -= current.end_mass - following.start_mass
    state = integrate_physical(
        case,
        state,
        case.staging_gap_seconds,
        current.thrust,
        current.exhaust_velocity,
        0.0,
        0.0,
    )
    tracker.stage_index += 1
    tracker.active_used = 0.0
    tracker.transition_count += 1
    return state


def consume_physical_burn(
    case: MultiStageCase,
    state: Array,
    duration: float,
    alpha: float,
    tracker: PhysicalTracker,
    stages: tuple[PhysicalStage, PhysicalStage],
    transition_at_endpoint: bool,
) -> Array:
    remaining = duration
    tolerance = 1e-10
    while remaining > tolerance:
        stage = stages[tracker.stage_index]
        capacity = stage.full_burn_time - tracker.active_used
        if capacity <= tolerance:
            if tracker.stage_index + 1 >= len(stages):
                raise ValueError("physical schedule exceeds final-stage capacity")
            state = transition_physical(case, state, tracker, stages)
            continue
        segment_duration = min(remaining, capacity)
        state = integrate_physical(
            case,
            state,
            segment_duration,
            stage.thrust,
            stage.exhaust_velocity,
            1.0,
            alpha,
        )
        tracker.active_used += segment_duration
        remaining -= segment_duration
        if remaining <= tolerance:
            remaining = 0.0
        exhausted = stage.full_burn_time - tracker.active_used <= tolerance
        if exhausted and (remaining > tolerance or transition_at_endpoint):
            if tracker.stage_index + 1 >= len(stages):
                if remaining > tolerance:
                    raise ValueError("physical schedule exceeds final-stage capacity")
                break
            state = transition_physical(case, state, tracker, stages)
    return state


def physical_propagate(
    case: MultiStageCase, parameters: Array, angle_intervals: int
) -> tuple[Array, int]:
    burn_one, coast, burn_two = (float(value) for value in parameters[:3])
    angles_one = parameters[3 : 3 + angle_intervals]
    angles_two = parameters[3 + angle_intervals :]
    stages = physical_fixture_stages(case.name)
    mass_scale = stages[0].start_mass
    state = normalized_to_physical(case, case.initial_state, mass_scale)
    tracker = PhysicalTracker()
    for angle in angles_one:
        state = consume_physical_burn(
            case,
            state,
            burn_one / angle_intervals * case.time_scale,
            float(angle),
            tracker,
            stages,
            True,
        )
    state = integrate_physical(
        case,
        state,
        coast * case.time_scale,
        stages[tracker.stage_index].thrust,
        stages[tracker.stage_index].exhaust_velocity,
        0.0,
        0.0,
    )
    for index, angle in enumerate(angles_two):
        state = consume_physical_burn(
            case,
            state,
            burn_two / angle_intervals * case.time_scale,
            float(angle),
            tracker,
            stages,
            index + 1 < angle_intervals,
        )
    return physical_to_normalized(case, state, mass_scale), tracker.transition_count


def run_checks() -> dict[str, float]:
    values: dict[str, float] = {}
    for case_name, parameters in (
        (
            "kerbin-stage-data",
            np.array(
                [0.31, 0.20, 0.07, 1.15, 1.30, 1.45, 1.55, 1.58, 1.59, 1.60, 1.61],
                dtype=float,
            ),
        ),
        (
            "kerbin-first-example",
            np.array(
                [0.05, 0.22, 0.28, 1.20, 1.35, 1.50, 1.58, 1.60, 1.61, 1.62, 1.63],
                dtype=float,
            ),
        ),
    ):
        case = make_case(case_name)
        normalized = direct_propagate(case, parameters, 4)
        physical, transitions = physical_propagate(case, parameters, 4)
        if transitions != 1:
            raise ValueError(f"{case_name}: expected one staging transition")
        values[f"{case_name}_error"] = float(
            np.linalg.norm(normalized.final_state - physical)
        )
    return values


def main() -> None:
    result = run_checks()
    print(json.dumps(result, indent=2, sort_keys=True))
    if max(result.values()) > 1e-8:
        raise RuntimeError(f"multi-stage unit check failed: {result}")


if __name__ == "__main__":
    main()
