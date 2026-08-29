#!/usr/bin/env python3
"""Independent unit and junction checks for the staged research model."""

from __future__ import annotations

import json
import math

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from staged_two_arc_research import (
    RTOL,
    StagedCase,
    integrate_state,
    jettison_mass_fraction,
    make_case,
    physical_fixture_stages,
)

Array = NDArray[np.float64]


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
    state: Array,
    duration: float,
    case: StagedCase,
    thrust: float,
    exhaust_velocity: float,
    throttle: float,
    alpha: float,
) -> Array:
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


def normalized_to_physical(case: StagedCase, state: Array, mass_ref: float) -> Array:
    return np.array(
        [
            state[0] * case.target_radius,
            state[1] * case.velocity_scale,
            state[2] * case.velocity_scale,
            state[3] * mass_ref,
        ],
        dtype=float,
    )


def physical_to_normalized(case: StagedCase, state: Array, mass_ref: float) -> Array:
    return np.array(
        [
            state[0] / case.target_radius,
            state[1] / case.velocity_scale,
            state[2] / case.velocity_scale,
            state[3] / mass_ref,
        ],
        dtype=float,
    )


def energy(
    state: Array, _mu: float, length_scale: float, velocity_scale: float
) -> float:
    radius = float(state[0])
    radial_velocity = float(state[1])
    tangential_velocity = float(state[2])
    return float(
        0.5
        * (
            (radial_velocity / velocity_scale) ** 2
            + (tangential_velocity / velocity_scale) ** 2
        )
        - length_scale / radius
    )


def angular_momentum(state: Array, length_scale: float, velocity_scale: float) -> float:
    return float(state[0] * state[2] / (length_scale * velocity_scale))


def run_checks() -> dict[str, float]:
    case = make_case()
    stage_one_physical, stage_two_physical = physical_fixture_stages()
    mass_ref = stage_one_physical.start_mass
    alpha_one = 1.2
    alpha_two = 1.4
    second_burn_seconds = 59.8794026271531

    normalized_stage_one = integrate_state(
        case.initial_state,
        case.stage_one.full_burn_time,
        case.stage_one,
        1.0,
        alpha_one,
    )
    physical_initial = normalized_to_physical(case, case.initial_state, mass_ref)
    physical_stage_one = integrate_physical(
        physical_initial,
        stage_one_physical.full_burn_time,
        case,
        stage_one_physical.thrust,
        stage_one_physical.exhaust_velocity,
        1.0,
        alpha_one,
    )
    physical_stage_one_normalized = physical_to_normalized(
        case, physical_stage_one, mass_ref
    )
    stage_one_error = float(
        np.linalg.norm(normalized_stage_one - physical_stage_one_normalized)
    )

    normalized_after_jump = normalized_stage_one.copy()
    normalized_after_jump[3] -= jettison_mass_fraction(case)
    physical_after_jump = physical_stage_one.copy()
    physical_after_jump[3] -= jettison_mass_fraction(case) * mass_ref
    jump_error = float(
        abs(
            normalized_after_jump[3]
            - physical_to_normalized(case, physical_after_jump, mass_ref)[3]
        )
    )

    normalized_gap = integrate_state(
        normalized_after_jump,
        case.staging_gap,
        case.stage_one,
        0.0,
        0.0,
    )
    physical_gap = integrate_physical(
        physical_after_jump,
        case.staging_gap_seconds,
        case,
        stage_one_physical.thrust,
        stage_one_physical.exhaust_velocity,
        0.0,
        0.0,
    )
    physical_gap_normalized = physical_to_normalized(case, physical_gap, mass_ref)
    gap_error = float(np.linalg.norm(normalized_gap - physical_gap_normalized))
    gap_energy_drift = abs(
        energy(physical_gap, case.mu, case.target_radius, case.velocity_scale)
        - energy(
            physical_after_jump,
            case.mu,
            case.target_radius,
            case.velocity_scale,
        )
    )
    gap_angular_momentum_drift = abs(
        angular_momentum(physical_gap, case.target_radius, case.velocity_scale)
        - angular_momentum(physical_after_jump, case.target_radius, case.velocity_scale)
    )

    normalized_stage_two = integrate_state(
        normalized_gap,
        second_burn_seconds / case.time_scale,
        case.stage_two,
        1.0,
        alpha_two,
    )
    physical_stage_two = integrate_physical(
        physical_gap,
        second_burn_seconds,
        case,
        stage_two_physical.thrust,
        stage_two_physical.exhaust_velocity,
        1.0,
        alpha_two,
    )
    physical_stage_two_normalized = physical_to_normalized(
        case, physical_stage_two, mass_ref
    )
    stage_two_error = float(
        np.linalg.norm(normalized_stage_two - physical_stage_two_normalized)
    )

    return {
        "stage_one_normalized_physical_error": stage_one_error,
        "mass_jump_error": jump_error,
        "staging_gap_normalized_physical_error": gap_error,
        "staging_gap_energy_drift": gap_energy_drift,
        "staging_gap_angular_momentum_drift": gap_angular_momentum_drift,
        "stage_two_normalized_physical_error": stage_two_error,
        "stage_one_mass_budget_error": abs(
            normalized_stage_one[3] - case.stage_one.end_mass
        ),
        "stage_two_mass_budget_error": abs(
            normalized_stage_two[3]
            - (
                case.stage_two.start_mass
                - case.stage_two.gamma
                / case.stage_two.kappa
                * second_burn_seconds
                / case.time_scale
            )
        ),
    }


def main() -> None:
    result = run_checks()
    print(json.dumps(result, indent=2, sort_keys=True))
    if max(result.values()) > 1e-8:
        raise RuntimeError(f"staging unit check failed: {result}")


if __name__ == "__main__":
    main()
