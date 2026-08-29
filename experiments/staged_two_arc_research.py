#!/usr/bin/env python3
"""Numerical staging study for the normalized polar primer model.

The experiment is deliberately independent of the production simulator.  It
models one full first-stage burn, a fixed ballistic staging gap, and a second
stage burn with different thrust, exhaust velocity, and starting mass.

Run from the repository root with::

    PYTHONPATH=experiments python3 experiments/staged_two_arc_research.py
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from scipy.optimize import NonlinearConstraint, least_squares, minimize

Array = NDArray[np.float64]

RTOL = 2e-11
ATOL = 2e-13


@dataclass(frozen=True)
class PhysicalStage:
    """A stage that burns at full thrust for a fixed maximum duration."""

    name: str
    thrust: float
    exhaust_velocity: float
    full_burn_time: float
    start_mass: float

    @property
    def mass_flow(self) -> float:
        return self.thrust / self.exhaust_velocity

    @property
    def end_mass(self) -> float:
        return self.start_mass - self.mass_flow * self.full_burn_time


@dataclass(frozen=True)
class NormalizedStage:
    name: str
    gamma: float
    kappa: float
    full_burn_time: float
    start_mass: float
    end_mass: float


@dataclass(frozen=True)
class StagedCase:
    name: str
    mu: float
    target_radius: float
    velocity_scale: float
    time_scale: float
    initial_state: Array
    stage_one: NormalizedStage
    stage_two: NormalizedStage
    staging_gap: float
    staging_gap_seconds: float
    initial_time_to_apoapsis_seconds: float


@dataclass(frozen=True)
class DirectResult:
    success: bool
    message: str
    parameters: Array
    final_state: Array
    residual: Array
    objective: float
    angle_intervals: int
    attempts: int


@dataclass(frozen=True)
class PrimerArc:
    name: str
    stage: NormalizedStage
    throttle: float
    duration: float
    start: Array
    end: Array
    samples: Array


@dataclass(frozen=True)
class PrimerResult:
    success: bool
    message: str
    parameters: Array
    final_joint: Array
    residual: Array
    arcs: tuple[PrimerArc, ...]
    attempts: int


def norm(vector: Array) -> float:
    return float(np.linalg.norm(vector))


def jettison_mass_fraction(case: StagedCase) -> float:
    return case.stage_one.end_mass - case.stage_two.start_mass


def state_rhs(
    _time: float,
    state: Array,
    gamma: float,
    kappa: float,
    throttle: float,
    alpha: float,
) -> Array:
    rho, radial_velocity, tangential_velocity, mass = state
    if rho <= 0.0 or mass <= 0.0:
        return np.full(4, np.nan)
    acceleration = throttle * gamma / mass
    return np.array(
        [
            radial_velocity,
            tangential_velocity**2 / rho
            - 1.0 / rho**2
            + acceleration * math.cos(alpha),
            -radial_velocity * tangential_velocity / rho
            + acceleration * math.sin(alpha),
            -throttle * gamma / kappa,
        ],
        dtype=float,
    )


def integrate_state(
    state: Array,
    duration: float,
    stage: NormalizedStage,
    throttle: float,
    alpha: float,
) -> Array:
    if duration < 0.0:
        raise ValueError("duration must be non-negative")
    if duration == 0.0:
        return state.copy()
    solution = solve_ivp(
        state_rhs,
        (0.0, duration),
        state,
        args=(stage.gamma, stage.kappa, throttle, alpha),
        method="DOP853",
        rtol=RTOL,
        atol=ATOL,
    )
    if not solution.success or not np.all(np.isfinite(solution.y[:, -1])):
        raise ValueError(solution.message)
    return np.asarray(solution.y[:, -1], dtype=float)


def primer_rhs(
    _time: float,
    joint: Array,
    gamma: float,
    kappa: float,
    throttle: float,
) -> Array:
    (
        rho,
        radial_velocity,
        tangential_velocity,
        mass,
        lambda_rho,
        p_r,
        p_t,
        _lambda_mass,
    ) = joint
    if rho <= 0.0 or mass <= 0.0:
        return np.full(8, np.nan)
    primer_length = math.hypot(p_r, p_t)
    if primer_length <= 1e-14:
        return np.full(8, np.nan)
    alpha = math.atan2(p_t, p_r)
    state_dot = state_rhs(
        _time,
        np.array([rho, radial_velocity, tangential_velocity, mass]),
        gamma,
        kappa,
        throttle,
        alpha,
    )
    lambda_rho_dot = (
        p_r * (2.0 / rho**3 - tangential_velocity**2 / rho**2)
        + p_t * radial_velocity * tangential_velocity / rho**2
    )
    p_r_dot = lambda_rho + p_t * tangential_velocity / rho
    p_t_dot = -2.0 * p_r * tangential_velocity / rho + p_t * radial_velocity / rho
    lambda_mass_dot = -throttle * gamma * primer_length / mass**2
    return np.array(
        [
            *state_dot,
            lambda_rho_dot,
            p_r_dot,
            p_t_dot,
            lambda_mass_dot,
        ],
        dtype=float,
    )


def switch_function(joint: Array, stage: NormalizedStage) -> float:
    mass = float(joint[3])
    primer_length = math.hypot(joint[5], joint[6])
    return float(primer_length / mass + float(joint[7]) / stage.kappa)


def hamiltonian(joint: Array, stage: NormalizedStage, throttle: float) -> float:
    (
        rho,
        radial_velocity,
        tangential_velocity,
        mass,
        lambda_rho,
        p_r,
        p_t,
        lambda_mass,
    ) = joint
    primer_length = math.hypot(p_r, p_t)
    gravity_radial = tangential_velocity**2 / rho - 1.0 / rho**2
    return float(
        lambda_rho * radial_velocity
        - p_r * gravity_radial
        + p_t * radial_velocity * tangential_velocity / rho
        - throttle * stage.gamma * (primer_length / mass + lambda_mass / stage.kappa)
    )


def integrate_primer_arc(
    joint: Array,
    duration: float,
    stage: NormalizedStage,
    throttle: float,
    name: str,
    sample_count: int = 81,
) -> tuple[Array, PrimerArc]:
    if duration < 0.0:
        raise ValueError("duration must be non-negative")
    if duration == 0.0:
        samples = joint.reshape(1, -1).copy()
        return joint.copy(), PrimerArc(
            name, stage, throttle, duration, joint.copy(), joint.copy(), samples
        )
    solution = solve_ivp(
        primer_rhs,
        (0.0, duration),
        joint,
        args=(stage.gamma, stage.kappa, throttle),
        method="DOP853",
        rtol=RTOL,
        atol=ATOL,
        dense_output=True,
    )
    if not solution.success or solution.sol is None:
        raise ValueError(solution.message)
    end = np.asarray(solution.y[:, -1], dtype=float)
    sample_times = np.linspace(0.0, duration, sample_count)
    samples = np.asarray(solution.sol(sample_times).T, dtype=float)
    return end, PrimerArc(
        name, stage, throttle, duration, joint.copy(), end.copy(), samples
    )


def physical_stage_to_normalized(
    stage: PhysicalStage,
    mass_scale: float,
    velocity_scale: float,
    length_scale: float,
    time_scale: float,
) -> NormalizedStage:
    return NormalizedStage(
        name=stage.name,
        gamma=stage.thrust * length_scale / (mass_scale * velocity_scale**2),
        kappa=stage.exhaust_velocity / velocity_scale,
        full_burn_time=stage.full_burn_time / time_scale,
        start_mass=stage.start_mass / mass_scale,
        end_mass=stage.end_mass / mass_scale,
    )


def physical_fixture_stages() -> tuple[PhysicalStage, PhysicalStage]:
    return (
        PhysicalStage(
            name="Swivel",
            thrust=215_000.0,
            exhaust_velocity=3138.1279999999997,
            full_burn_time=59.0500960010656,
            start_mass=13_885.650390625,
        ),
        PhysicalStage(
            name="Terrier",
            thrust=60_000.0,
            exhaust_velocity=3383.2942499999995,
            full_burn_time=112.77647255563578,
            start_mass=4_449.999407536325,
        ),
    )


def make_case() -> StagedCase:
    mu = 3.5316e12
    target_radius = 680_000.0
    velocity_scale = math.sqrt(mu / target_radius)
    time_scale = target_radius / velocity_scale
    position = np.array(
        [424370.58766631, -1093.08696926, -470992.64951719], dtype=float
    )
    velocity = np.array([723.81414935, -1.20334290, -122.60883836], dtype=float)
    radius = norm(position)
    radial_hat = position / radius
    radial_velocity = float(np.dot(velocity, radial_hat))
    tangential_vector = velocity - radial_velocity * radial_hat
    tangential_hat = tangential_vector / norm(tangential_vector)
    tangential_velocity = float(np.dot(velocity, tangential_hat))
    initial_state = np.array(
        [
            radius / target_radius,
            radial_velocity / velocity_scale,
            tangential_velocity / velocity_scale,
            1.0,
        ],
        dtype=float,
    )

    stage_one_physical, stage_two_physical = physical_fixture_stages()
    if stage_two_physical.start_mass >= stage_one_physical.end_mass:
        raise ValueError("stage two must start below stage one end mass")
    stage_one = physical_stage_to_normalized(
        stage_one_physical,
        stage_one_physical.start_mass,
        velocity_scale,
        target_radius,
        time_scale,
    )
    stage_two = physical_stage_to_normalized(
        stage_two_physical,
        stage_one_physical.start_mass,
        velocity_scale,
        target_radius,
        time_scale,
    )
    if not math.isclose(stage_one.start_mass, 1.0, rel_tol=0.0, abs_tol=1e-14):
        raise ValueError("stage one must start at the reference mass")
    if stage_two.start_mass >= stage_one.end_mass:
        raise ValueError("stage two mass must be below stage one end mass")
    return StagedCase(
        name="kerbin-two-stage",
        mu=mu,
        target_radius=target_radius,
        velocity_scale=velocity_scale,
        time_scale=time_scale,
        initial_state=initial_state,
        stage_one=stage_one,
        stage_two=stage_two,
        staging_gap=2.0 / time_scale,
        staging_gap_seconds=2.0,
        initial_time_to_apoapsis_seconds=72.12194913376851,
    )


def propagate_direct(
    case: StagedCase,
    burn_two: float,
    angles_one: Array,
    angles_two: Array,
) -> Array:
    state = case.initial_state.copy()
    interval_one = case.stage_one.full_burn_time / len(angles_one)
    for angle in angles_one:
        state = integrate_state(state, interval_one, case.stage_one, 1.0, float(angle))
    stage_one_mass = state[3]
    if abs(stage_one_mass - case.stage_one.end_mass) > 2e-9:
        raise ValueError("stage one mass integration disagrees with its budget")
    state[3] -= jettison_mass_fraction(case)
    if abs(state[3] - case.stage_two.start_mass) > 2e-9:
        raise ValueError("fixed jettison does not reach stage two start mass")
    state = integrate_state(state, case.staging_gap, case.stage_one, 0.0, 0.0)
    if burn_two <= 0.0:
        raise ValueError("second-stage burn must be positive")
    interval_two = burn_two / len(angles_two)
    for angle in angles_two:
        state = integrate_state(state, interval_two, case.stage_two, 1.0, float(angle))
    return state


def direct_residual(case: StagedCase, parameters: Array, angle_intervals: int) -> Array:
    burn_two = float(parameters[0])
    angles_one = parameters[1 : 1 + angle_intervals]
    angles_two = parameters[1 + angle_intervals :]
    try:
        final = propagate_direct(case, burn_two, angles_one, angles_two)
    except (FloatingPointError, ValueError, ZeroDivisionError):
        return np.full(3, 1e3)
    return np.array([final[0] - 1.0, final[1], final[2] - 1.0])


def total_propellant_fraction(case: StagedCase, burn_two: float) -> float:
    return (
        case.stage_one.start_mass - case.stage_one.end_mass
    ) + case.stage_two.gamma / case.stage_two.kappa * burn_two


def direct_initial_guesses(case: StagedCase, angle_intervals: int) -> list[Array]:
    velocity_angle = math.atan2(case.initial_state[2], case.initial_state[1])
    first_patterns = [
        np.full(angle_intervals, velocity_angle),
        np.full(angle_intervals, math.pi / 2.0),
        np.linspace(velocity_angle, math.pi / 2.0, angle_intervals),
        np.linspace(math.pi / 2.0, 1.05, angle_intervals),
        np.linspace(1.10, 1.70, angle_intervals),
    ]
    second_patterns = [
        np.full(angle_intervals, math.pi / 2.0),
        np.full(angle_intervals, 1.35),
        np.full(angle_intervals, 1.70),
        np.linspace(1.20, 1.65, angle_intervals),
    ]
    guesses: list[Array] = []
    for first_index, first in enumerate(first_patterns):
        second = second_patterns[first_index % len(second_patterns)]
        guesses.extend(
            [
                np.concatenate(
                    (
                        [fraction * case.stage_two.full_burn_time],
                        first,
                        second,
                    ),
                    dtype=float,
                )
                for fraction in (0.20, 0.40, 0.60, 0.80)
            ]
        )
    return guesses


def solve_direct(
    case: StagedCase, angle_intervals: int, max_attempts: int
) -> DirectResult:
    lower = np.concatenate(
        ([1e-8], np.full(2 * angle_intervals, -math.pi)), dtype=float
    )
    upper = np.concatenate(
        ([case.stage_two.full_burn_time], np.full(2 * angle_intervals, math.pi)),
        dtype=float,
    )
    attempts: list[tuple[Any, Array, Array]] = []
    for initial in direct_initial_guesses(case, angle_intervals)[:max_attempts]:

        def objective(parameters: Array) -> float:
            return total_propellant_fraction(case, float(parameters[0]))

        def equality_constraint(parameters: Array) -> Array:
            return direct_residual(case, parameters, angle_intervals)

        result = minimize(
            objective,
            initial,
            method="SLSQP",
            bounds=list(zip(lower, upper, strict=True)),
            constraints=NonlinearConstraint(equality_constraint, 0.0, 0.0),
            options={"ftol": 2e-11, "maxiter": 900, "disp": False},
        )
        parameters = np.asarray(result.x, dtype=float)
        residual = direct_residual(case, parameters, angle_intervals)
        attempts.append((result, parameters, residual))
    accepted_indices = [
        index
        for index, item in enumerate(attempts)
        if bool(item[0].success) and norm(item[2]) < 2e-6
    ]
    candidate_indices = accepted_indices or list(range(len(attempts)))
    chosen_index = min(
        candidate_indices,
        key=lambda index: (
            total_propellant_fraction(case, float(attempts[index][1][0]))
            if index in accepted_indices
            else norm(attempts[index][2])
        ),
    )
    result, parameters, residual = attempts[chosen_index]
    return DirectResult(
        success=bool(bool(result.success) and norm(residual) < 2e-6),
        message=str(result.message),
        parameters=parameters,
        final_state=propagate_direct(
            case,
            float(parameters[0]),
            parameters[1 : 1 + angle_intervals],
            parameters[1 + angle_intervals :],
        ),
        residual=residual,
        objective=total_propellant_fraction(case, float(parameters[0])),
        angle_intervals=angle_intervals,
        attempts=len(attempts),
    )


def propagate_primer(
    case: StagedCase, parameters: Array
) -> tuple[Array, tuple[PrimerArc, ...]]:
    alpha, lambda_rho, lambda_mass, burn_two = parameters
    initial_costate = np.array(
        [lambda_rho, math.cos(alpha), math.sin(alpha), lambda_mass], dtype=float
    )
    joint = np.concatenate((case.initial_state, initial_costate))
    joint, arc_one = integrate_primer_arc(
        joint,
        case.stage_one.full_burn_time,
        case.stage_one,
        1.0,
        "stage-one burn",
    )
    if abs(joint[3] - case.stage_one.end_mass) > 2e-9:
        raise ValueError("stage one primer mass disagrees with its budget")

    # A fixed dry-stage jettison maps eta+ = eta- - delta. Its derivative is
    # one, so the position, velocity, and mass costates remain continuous.
    joint[3] -= jettison_mass_fraction(case)
    if abs(joint[3] - case.stage_two.start_mass) > 2e-9:
        raise ValueError("fixed jettison does not reach stage two start mass")
    joint, arc_gap = integrate_primer_arc(
        joint,
        case.staging_gap,
        case.stage_one,
        0.0,
        "staging coast",
    )
    joint, arc_two = integrate_primer_arc(
        joint,
        float(burn_two),
        case.stage_two,
        1.0,
        "stage-two burn",
    )
    return joint, (arc_one, arc_gap, arc_two)


def primer_residual(case: StagedCase, parameters: Array) -> Array:
    try:
        final, _arcs = propagate_primer(case, parameters)
        switch_at_final = switch_function(final, case.stage_two)
        return np.array([final[0] - 1.0, final[1], final[2] - 1.0, switch_at_final])
    except (FloatingPointError, ValueError, ZeroDivisionError):
        return np.full(4, 1e3)


def lambda_rho_hamiltonian_guess(
    case: StagedCase, alpha: float, lambda_mass: float
) -> float:
    rho = float(case.initial_state[0])
    radial_velocity = float(case.initial_state[1])
    tangential_velocity = float(case.initial_state[2])
    mass = float(case.initial_state[3])
    p_r = math.cos(alpha)
    p_t = math.sin(alpha)
    gravity_radial = tangential_velocity**2 / rho - 1.0 / rho**2
    return (
        float(
            p_r * gravity_radial
            - p_t * radial_velocity * tangential_velocity / rho
            + case.stage_one.gamma * (1.0 / mass + lambda_mass / case.stage_one.kappa)
        )
        / radial_velocity
    )


def primer_initial_guesses(case: StagedCase, direct: DirectResult) -> list[Array]:
    alpha_direct = float(direct.parameters[1])
    guesses: list[Array] = []
    for angle_offset in (0.0, -0.15, 0.15):
        alpha = alpha_direct + angle_offset
        for mass_factor in (0.5, 1.0, 1.5, 2.0):
            lambda_mass = -mass_factor * case.stage_two.kappa
            lambda_rho = lambda_rho_hamiltonian_guess(case, alpha, lambda_mass)
            guesses.append(
                np.array(
                    [alpha, lambda_rho, lambda_mass, direct.parameters[0]],
                    dtype=float,
                )
            )
    return guesses


def arc_diagnostics(case: StagedCase, arcs: tuple[PrimerArc, ...]) -> dict[str, object]:
    records: list[dict[str, float | str]] = []
    for arc in arcs:
        h_values = np.array(
            [
                hamiltonian(arc.samples[index], arc.stage, arc.throttle)
                for index in range(len(arc.samples))
            ],
            dtype=float,
        )
        phi_values = np.array(
            [switch_function(sample, arc.stage) for sample in arc.samples],
            dtype=float,
        )
        record: dict[str, float | str] = {
            "name": arc.name,
            "duration": arc.duration,
            "hamiltonian_start": float(h_values[0]),
            "hamiltonian_end": float(h_values[-1]),
            "hamiltonian_drift": float(h_values[-1] - h_values[0]),
            "switch_min": float(phi_values.min()),
            "switch_max": float(phi_values.max()),
        }
        records.append(record)
    stage_one_end = arcs[0].end
    stage_two_start = arcs[1].start
    final = arcs[-1].end
    records.append(
        {
            "name": "staging-junction",
            "mass_before": float(stage_one_end[3]),
            "mass_after": float(stage_two_start[3]),
            "mass_jump": float(stage_two_start[3] - stage_one_end[3]),
            "lambda_mass_before": float(stage_one_end[7]),
            "lambda_mass_after": float(stage_two_start[7]),
            "lambda_mass_jump": float(stage_two_start[7] - stage_one_end[7]),
            "hamiltonian_before": hamiltonian(stage_one_end, case.stage_one, 1.0),
            "hamiltonian_after": hamiltonian(stage_two_start, case.stage_one, 0.0),
        }
    )
    final_switch = switch_function(final, case.stage_two)
    final_hamiltonian = hamiltonian(final, case.stage_two, 1.0)
    records.append(
        {
            "name": "terminal",
            "switch": final_switch,
            "hamiltonian": final_hamiltonian,
            "hamiltonian_from_switch": -case.stage_two.gamma * final_switch,
        }
    )
    return {"arcs": records}


def solve_primer(
    case: StagedCase, direct: DirectResult, max_attempts: int
) -> PrimerResult:
    lower = np.array([-math.pi, -100.0, -100.0, 1e-8], dtype=float)
    upper = np.array(
        [math.pi, 100.0, 100.0, case.stage_two.full_burn_time], dtype=float
    )
    attempts: list[tuple[Any, Array, Array, tuple[PrimerArc, ...]]] = []
    for initial in primer_initial_guesses(case, direct)[:max_attempts]:

        def residual_function(parameters: Array) -> Array:
            return primer_residual(case, parameters)

        result = least_squares(
            residual_function,
            initial,
            bounds=(lower, upper),
            x_scale="jac",
            ftol=2e-12,
            xtol=2e-12,
            gtol=2e-12,
            max_nfev=1600,
        )
        parameters = np.asarray(result.x, dtype=float)
        residual = primer_residual(case, parameters)
        final, arcs = propagate_primer(case, parameters)
        attempts.append((result, parameters, residual, arcs))
    accepted_indices = [
        index
        for index, item in enumerate(attempts)
        if bool(item[0].success) and norm(item[2]) < 2e-6
    ]
    candidate_indices = accepted_indices or list(range(len(attempts)))
    chosen_index = min(
        candidate_indices,
        key=lambda index: (
            float(attempts[index][1][3])
            if index in accepted_indices
            else norm(attempts[index][2])
        ),
    )
    result, parameters, residual, arcs = attempts[chosen_index]
    final, _ = propagate_primer(case, parameters)
    return PrimerResult(
        success=bool(bool(result.success) and norm(residual) < 2e-6),
        message=str(result.message),
        parameters=parameters,
        final_joint=final,
        residual=residual,
        arcs=arcs,
        attempts=len(attempts),
    )


def result_record(
    case: StagedCase, direct: DirectResult, primer: PrimerResult
) -> dict[str, object]:
    stage_one = case.stage_one
    stage_two = case.stage_two
    burn_two = float(direct.parameters[0])
    final_mass_direct = float(direct.final_state[3])
    final_mass_primer = float(primer.final_joint[3])
    return {
        "case": case.name,
        "normalization": {
            "target_radius_m": case.target_radius,
            "velocity_scale_m_per_s": case.velocity_scale,
            "time_scale_s": case.time_scale,
            "initial_time_to_apoapsis_seconds": case.initial_time_to_apoapsis_seconds,
            "initial_state": case.initial_state.tolist(),
        },
        "staging": {
            "gap_seconds": case.staging_gap_seconds,
            "gap_normalized": case.staging_gap,
            "stage_one": {
                "name": stage_one.name,
                "gamma": stage_one.gamma,
                "kappa": stage_one.kappa,
                "full_burn_time": stage_one.full_burn_time,
                "start_mass_fraction": stage_one.start_mass,
                "end_mass_fraction": stage_one.end_mass,
            },
            "stage_two": {
                "name": stage_two.name,
                "gamma": stage_two.gamma,
                "kappa": stage_two.kappa,
                "full_burn_time": stage_two.full_burn_time,
                "start_mass_fraction": stage_two.start_mass,
                "end_mass_fraction": stage_two.end_mass,
            },
            "jettison_mass_fraction": jettison_mass_fraction(case),
        },
        "direct_reference": {
            "success": direct.success,
            "message": direct.message,
            "attempts": direct.attempts,
            "angle_intervals_per_stage": direct.angle_intervals,
            "burn_two_normalized": burn_two,
            "burn_two_seconds": burn_two * case.time_scale,
            "total_elapsed_seconds": (
                stage_one.full_burn_time + case.staging_gap + burn_two
            )
            * case.time_scale,
            "total_propellant_fraction": direct.objective,
            "final_mass_fraction": final_mass_direct,
            "parameters": direct.parameters.tolist(),
            "final_state": direct.final_state.tolist(),
            "residual_norm": norm(direct.residual),
        },
        "primer_solution": {
            "success": primer.success,
            "message": primer.message,
            "attempts": primer.attempts,
            "parameters": primer.parameters.tolist(),
            "burn_two_normalized": float(primer.parameters[3]),
            "burn_two_seconds": float(primer.parameters[3]) * case.time_scale,
            "total_elapsed_seconds": (
                stage_one.full_burn_time + case.staging_gap + primer.parameters[3]
            )
            * case.time_scale,
            "total_propellant_fraction": total_propellant_fraction(
                case, float(primer.parameters[3])
            ),
            "final_mass_fraction": final_mass_primer,
            "final_state": primer.final_joint[:4].tolist(),
            "residual_norm": norm(primer.residual),
            "diagnostics": arc_diagnostics(case, primer.arcs),
        },
        "comparison": {
            "burn_two_relative_difference": float(primer.parameters[3]) / burn_two
            - 1.0,
            "final_mass_difference": final_mass_primer - final_mass_direct,
            "propellant_fraction_difference": total_propellant_fraction(
                case, float(primer.parameters[3])
            )
            - direct.objective,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--angles", type=int, default=4)
    parser.add_argument("--direct-attempts", type=int, default=20)
    parser.add_argument("--primer-attempts", type=int, default=12)
    parser.add_argument("--output", type=str, default="")
    arguments = parser.parse_args()

    case = make_case()
    direct = solve_direct(case, arguments.angles, arguments.direct_attempts)
    if not direct.success:
        raise RuntimeError(
            f"direct reference failed: {direct.message}; "
            f"residual={norm(direct.residual):.3e}"
        )
    primer = solve_primer(case, direct, arguments.primer_attempts)
    if not primer.success:
        raise RuntimeError(
            f"primer solve failed: {primer.message}; "
            f"residual={norm(primer.residual):.3e}"
        )
    record = result_record(case, direct, primer)
    primer_fuel = total_propellant_fraction(case, float(primer.parameters[3]))
    print(
        f"{case.name}: direct={direct.success} primer={primer.success} "
        f"direct_fuel={direct.objective:.9f} "
        f"primer_fuel={primer_fuel:.9f} "
        f"direct_res={norm(direct.residual):.3e} "
        f"primer_res={norm(primer.residual):.3e}"
    )
    print(json.dumps(record, indent=2))
    if arguments.output:
        with open(arguments.output, "w", encoding="utf-8") as output_file:
            json.dump(record, output_file, indent=2)
            output_file.write("\n")


if __name__ == "__main__":
    main()
