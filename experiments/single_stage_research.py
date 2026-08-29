#!/usr/bin/env python3
"""Disposable numerical study for the single-stage primer-vector model.

This file deliberately does not modify or import the production optimizer's
implementation.  It contains an independent normalized polar model, a
Cartesian propagation reference, a direct piecewise-angle reference solve, and
the two single-stage primer shooting variants under consideration.

Run from the repository root with::

    PYTHONPATH=. python3 experiments/single_stage_research.py
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from scipy.optimize import NonlinearConstraint, least_squares, minimize

Array = NDArray[np.float64]


@dataclass(frozen=True)
class Stage:
    gamma: float
    kappa: float
    max_burn: float


@dataclass(frozen=True)
class Case:
    name: str
    x0: Array
    stage: Stage
    time_to_apoapsis: float
    first_arc_limit: float


@dataclass
class DirectResult:
    success: bool
    message: str
    x_final: Array
    coast_time: float
    burn_time: float
    angles: Array
    objective: float
    residual: Array
    n_intervals: int


@dataclass
class PrimerResult:
    success: bool
    message: str
    z: Array
    x_final: Array
    residual: Array
    burn_start: float
    burn_time: float
    switch_time: float | None
    switch_times: list[float]
    event_count: int


@dataclass
class RelaxedResult:
    success: bool
    message: str
    horizon: float
    throttles: Array
    angles: Array
    x_final: Array
    objective: float
    residual: Array
    n_intervals: int


@dataclass
class PrimerArc:
    start: float
    end: float
    throttle: float
    times: Array
    joint: Array


def norm(vector: Array) -> float:
    return float(np.linalg.norm(vector))


def polar_rhs(
    _time: float,
    state: Array,
    gamma: float,
    kappa: float,
    throttle: float,
    alpha: float,
) -> Array:
    rho, ur, ut, mass = state
    if rho <= 0.0 or mass <= 0.0:
        return np.full(4, np.nan)
    acceleration = throttle * gamma / mass
    return np.array(
        [
            ur,
            ut * ut / rho - 1.0 / (rho * rho) + acceleration * math.cos(alpha),
            -ur * ut / rho + acceleration * math.sin(alpha),
            -throttle * gamma / kappa,
        ],
        dtype=float,
    )


def cartesian_rhs(
    _time: float,
    state: Array,
    gamma: float,
    kappa: float,
    throttle: float,
    alpha: float,
) -> Array:
    x, y, vx, vy, mass = state
    radius = math.hypot(x, y)
    if radius <= 0.0 or mass <= 0.0:
        return np.full(5, np.nan)
    radial = np.array([x, y]) / radius
    tangent = np.array([-radial[1], radial[0]])
    thrust = (
        throttle * gamma / mass * (math.cos(alpha) * radial + math.sin(alpha) * tangent)
    )
    gravity = -np.array([x, y]) / radius**3
    return np.array([vx, vy, *(gravity + thrust), -throttle * gamma / kappa])


def polar_to_cartesian(state: Array) -> Array:
    rho, ur, ut, mass = state
    return np.array([rho, 0.0, ur, ut, mass], dtype=float)


def cartesian_to_polar(state: Array) -> Array:
    x, y, vx, vy, mass = state
    radius = math.hypot(x, y)
    radial = np.array([x, y]) / radius
    tangent = np.array([-radial[1], radial[0]])
    velocity = np.array([vx, vy])
    return np.array(
        [
            radius,
            float(np.dot(velocity, radial)),
            float(np.dot(velocity, tangent)),
            mass,
        ],
        dtype=float,
    )


def integrate_polar(
    initial: Array,
    duration: float,
    stage: Stage,
    throttle: float,
    alpha: float,
    *,
    rtol: float = 2e-11,
    atol: float = 2e-13,
) -> Array:
    solution = solve_ivp(
        polar_rhs,
        (0.0, duration),
        initial,
        args=(stage.gamma, stage.kappa, throttle, alpha),
        method="DOP853",
        rtol=rtol,
        atol=atol,
    )
    if not solution.success or not np.all(np.isfinite(solution.y[:, -1])):
        raise ValueError(solution.message)
    return solution.y[:, -1]


def integrate_cartesian(
    initial: Array,
    duration: float,
    stage: Stage,
    throttle: float,
    alpha: float,
    *,
    rtol: float = 2e-11,
    atol: float = 2e-13,
) -> Array:
    solution = solve_ivp(
        cartesian_rhs,
        (0.0, duration),
        initial,
        args=(stage.gamma, stage.kappa, throttle, alpha),
        method="DOP853",
        rtol=rtol,
        atol=atol,
    )
    if not solution.success or not np.all(np.isfinite(solution.y[:, -1])):
        raise ValueError(solution.message)
    return solution.y[:, -1]


def physical_polar_rhs(
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


def integrate_physical_polar(
    initial: Array,
    duration: float,
    mu: float,
    thrust: float,
    exhaust_velocity: float,
    throttle: float,
    alpha: float,
) -> Array:
    solution = solve_ivp(
        physical_polar_rhs,
        (0.0, duration),
        initial,
        args=(mu, thrust, exhaust_velocity, throttle, alpha),
        method="DOP853",
        rtol=2e-11,
        atol=2e-10,
    )
    if not solution.success or not np.all(np.isfinite(solution.y[:, -1])):
        raise ValueError(solution.message)
    return solution.y[:, -1]


def coast_to_apoapsis(initial: Array, stage: Stage) -> tuple[Array, float]:
    def apoapsis_event(_time: float, state: Array, *_args: float) -> float:
        return float(state[1])

    apoapsis_event.terminal = True  # type: ignore[attr-defined]
    apoapsis_event.direction = -1.0  # type: ignore[attr-defined]
    solution = solve_ivp(
        polar_rhs,
        (0.0, 20.0),
        initial,
        args=(stage.gamma, stage.kappa, 0.0, 0.0),
        events=apoapsis_event,
        method="DOP853",
        rtol=2e-12,
        atol=2e-14,
    )
    if solution.t_events is None or solution.y_events is None:
        raise ValueError("event output was not created")
    if len(solution.t_events[0]) != 1:
        raise ValueError("failed to find apoapsis")
    return solution.y_events[0][0], float(solution.t_events[0][0])


def orbital_period(semi_major_axis: float) -> float:
    return 2.0 * math.pi * math.sqrt(semi_major_axis**3)


def synthetic_case(
    name: str,
    stage: Stage,
    rp: float,
    ra: float,
    true_anomaly: float,
) -> Case:
    semi_major_axis = 0.5 * (rp + ra)
    eccentricity = (ra - rp) / (ra + rp)
    semilatus = semi_major_axis * (1.0 - eccentricity * eccentricity)
    radius = semilatus / (1.0 + eccentricity * math.cos(true_anomaly))
    radial_velocity = eccentricity * math.sin(true_anomaly) / math.sqrt(semilatus)
    tangential_velocity = (1.0 + eccentricity * math.cos(true_anomaly)) / math.sqrt(
        semilatus
    )
    initial = np.array([radius, radial_velocity, tangential_velocity, 1.0])
    _, time_to_apoapsis = coast_to_apoapsis(initial, stage)
    return Case(
        name=name,
        x0=initial,
        stage=stage,
        time_to_apoapsis=time_to_apoapsis,
        first_arc_limit=time_to_apoapsis + 0.75 * orbital_period(semi_major_axis),
    )


def kerbin_case() -> Case:
    mu = 3.5316e12
    target_radius = 680_000.0
    target_velocity = math.sqrt(mu / target_radius)
    time_scale = target_radius / target_velocity
    position = np.array([424370.58766631, -1093.08696926, -470992.64951719])
    velocity = np.array([723.81414935, -1.2033429, -122.60883836])
    radius = norm(position)
    radial_hat = position / radius
    radial_velocity = float(np.dot(velocity, radial_hat)) / target_velocity
    tangent_velocity_vector = velocity - np.dot(velocity, radial_hat) * radial_hat
    tangential_velocity = norm(tangent_velocity_vector) / target_velocity
    initial = np.array(
        [radius / target_radius, radial_velocity, tangential_velocity, 1.0]
    )
    stage = Stage(
        gamma=215_000.0 * target_radius / (13_885.650390625 * target_velocity**2),
        kappa=3138.1279999999997 / target_velocity,
        # A single-stage study needs enough propellant to finish the target
        # orbit; this deliberately extends the example's first-stage burn
        # budget rather than pretending the original first stage can do both
        # jobs alone.
        max_burn=150.0 / time_scale,
    )
    _, time_to_apoapsis = coast_to_apoapsis(initial, stage)
    energy = 0.5 * (initial[1] ** 2 + initial[2] ** 2) - 1.0 / initial[0]
    semi_major_axis = -1.0 / (2.0 * energy)
    return Case(
        name="kerbin-example",
        x0=initial,
        stage=stage,
        time_to_apoapsis=time_to_apoapsis,
        first_arc_limit=time_to_apoapsis + 0.75 * orbital_period(semi_major_axis),
    )


def estimated_burn(case: Case) -> float:
    initial = case.x0
    orbit_energy = 0.5 * (initial[1] ** 2 + initial[2] ** 2) - 1.0 / initial[0]
    semi_major_axis = -1.0 / (2.0 * orbit_energy)
    coast_apo, _ = coast_to_apoapsis(initial, case.stage)
    apoapsis = coast_apo[0]
    first_speed = math.sqrt(2.0 / apoapsis - 1.0 / semi_major_axis)
    new_semi_major_axis = 0.5 * (apoapsis + 1.0)
    target_speed = math.sqrt(2.0 / apoapsis - 1.0 / new_semi_major_axis)
    delta_v = max(0.0, target_speed - first_speed)
    final_mass = math.exp(-delta_v / case.stage.kappa)
    return min(
        case.stage.max_burn * 0.9,
        case.stage.kappa / case.stage.gamma * (1.0 - final_mass),
    )


def propagate_piecewise_burn(
    case: Case,
    coast_time: float,
    burn_time: float,
    angles: Array,
) -> Array:
    if coast_time < 0.0 or burn_time <= 0.0:
        raise ValueError("invalid coast/burn time")
    state = integrate_polar(case.x0, coast_time, case.stage, 0.0, 0.0)
    interval = burn_time / len(angles)
    for alpha in angles:
        state = integrate_polar(state, interval, case.stage, 1.0, float(alpha))
    return state


def direct_initial_guess(case: Case, n_intervals: int) -> Array:
    burn = estimated_burn(case)
    coast = max(0.0, case.time_to_apoapsis - 0.5 * burn)
    return np.concatenate(
        ([coast, burn], np.zeros(n_intervals, dtype=float)), dtype=float
    )


def direct_solve(
    case: Case, n_intervals: int, previous: DirectResult | None = None
) -> DirectResult:
    if previous is None:
        initial = direct_initial_guess(case, n_intervals)
    else:
        old_centers = (np.arange(previous.n_intervals) + 0.5) / previous.n_intervals
        new_centers = (np.arange(n_intervals) + 0.5) / n_intervals
        refined_angles = np.interp(
            new_centers,
            old_centers,
            previous.angles,
            left=previous.angles[0],
            right=previous.angles[-1],
        )
        initial = np.concatenate(
            ([previous.coast_time, previous.burn_time], refined_angles), dtype=float
        )
    lower = np.concatenate(([0.0, 1e-6], np.full(n_intervals, -math.pi)))
    upper = np.concatenate(
        ([case.time_to_apoapsis, case.stage.max_burn], np.full(n_intervals, math.pi))
    )

    def evaluate(parameters: Array) -> tuple[Array, Array]:
        try:
            state = propagate_piecewise_burn(
                case, float(parameters[0]), float(parameters[1]), parameters[2:]
            )
            residual = np.array([state[0] - 1.0, state[1], state[2] - 1.0])
            return state, residual
        except (FloatingPointError, ValueError, ZeroDivisionError):
            return np.full(4, np.nan), np.full(3, 1e3)

    def objective(parameters: Array) -> float:
        return float(parameters[1])

    def constraints(parameters: Array) -> Array:
        return evaluate(parameters)[1]

    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=list(zip(lower, upper, strict=True)),
        constraints=NonlinearConstraint(constraints, 0.0, 0.0),
        options={"ftol": 2e-11, "maxiter": 450, "disp": False},
    )
    state, residual = evaluate(result.x)
    return DirectResult(
        success=bool(result.success and np.linalg.norm(residual) < 2e-6),
        message=str(result.message),
        x_final=state,
        coast_time=float(result.x[0]),
        burn_time=float(result.x[1]),
        angles=np.asarray(result.x[2:], dtype=float),
        objective=float(result.fun),
        residual=residual,
        n_intervals=n_intervals,
    )


def primer_rhs(
    _time: float,
    state_costate: Array,
    gamma: float,
    kappa: float,
    throttle: float,
) -> Array:
    rho, ur, ut, mass, lambda_rho, p_r, p_t, _lambda_mass = state_costate
    if rho <= 0.0 or mass <= 0.0:
        return np.full(8, np.nan)
    primer_length = math.hypot(p_r, p_t)
    if primer_length <= 1e-14:
        return np.full(8, np.nan)
    alpha = math.atan2(p_t, p_r)
    state_dot = polar_rhs(
        _time, np.array([rho, ur, ut, mass]), gamma, kappa, throttle, alpha
    )
    lambda_rho_dot = p_r * (2.0 / rho**3 - ut * ut / rho**2) + p_t * ur * ut / rho**2
    p_r_dot = lambda_rho + p_t * ut / rho
    p_t_dot = -2.0 * p_r * ut / rho + p_t * ur / rho
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


def primer_hamiltonian(
    state_costate: Array,
    gamma: float,
    kappa: float,
    throttle: float,
) -> float:
    rho, ur, ut, mass, lambda_rho, p_r, p_t, lambda_mass = state_costate
    primer_length = math.hypot(p_r, p_t)
    return float(
        lambda_rho * ur
        - p_r * (ut * ut / rho - 1.0 / rho**2)
        + p_t * ur * ut / rho
        - throttle * gamma * (primer_length / mass + lambda_mass / kappa)
    )


def integrate_primer(
    initial: Array,
    costate: Array,
    duration: float,
    stage: Stage,
    throttle: float,
) -> Array:
    initial_joint = np.concatenate((initial, costate))
    solution = solve_ivp(
        primer_rhs,
        (0.0, duration),
        initial_joint,
        args=(stage.gamma, stage.kappa, throttle),
        method="DOP853",
        rtol=2e-11,
        atol=2e-13,
    )
    if not solution.success or not np.all(np.isfinite(solution.y[:, -1])):
        raise ValueError(solution.message)
    return solution.y[:, -1]


def initial_lambda_rho(state: Array, p_r: float, p_t: float) -> float:
    rho, ur, ut, _mass = state
    gravity_kinematic = ut * ut / rho - 1.0 / rho**2
    if abs(ur) < 1e-10:
        raise ValueError("radial velocity too small to eliminate lambda_rho")
    return float((p_r * gravity_kinematic - p_t * ur * ut / rho) / ur)


def explicit_residual(case: Case, parameters: Array) -> Array:
    alpha0, coast_time, burn_time, lambda_rho = parameters
    coast_state = integrate_polar(case.x0, coast_time, case.stage, 0.0, 0.0)
    p_r = math.cos(alpha0)
    p_t = math.sin(alpha0)
    lambda_mass = -case.stage.kappa / coast_state[3]
    final_joint = integrate_primer(
        coast_state,
        np.array([lambda_rho, p_r, p_t, lambda_mass]),
        burn_time,
        case.stage,
        1.0,
    )
    return np.array(
        [
            final_joint[0] - 1.0,
            final_joint[1],
            final_joint[2] - 1.0,
            primer_hamiltonian(
                np.concatenate((coast_state, [lambda_rho, p_r, p_t, lambda_mass])),
                case.stage.gamma,
                case.stage.kappa,
                1.0,
            ),
        ]
    )


def explicit_solve(case: Case, direct: DirectResult) -> PrimerResult:
    alpha0 = float(direct.angles[0])
    coast_state = integrate_polar(case.x0, direct.coast_time, case.stage, 0.0, 0.0)
    lambda_rho = initial_lambda_rho(coast_state, math.cos(alpha0), math.sin(alpha0))
    initial = np.array([alpha0, direct.coast_time, direct.burn_time, lambda_rho])
    lower = np.array([-math.pi, 0.0, 1e-6, -100.0])
    upper = np.array([math.pi, case.time_to_apoapsis, case.stage.max_burn, 100.0])

    def residual_function(parameters: Array) -> Array:
        return explicit_residual(case, parameters)

    result = least_squares(
        residual_function,
        initial,
        bounds=(lower, upper),
        x_scale="jac",
        ftol=2e-12,
        xtol=2e-12,
        gtol=2e-12,
        max_nfev=500,
    )
    z = np.asarray(result.x, dtype=float)
    coast_state = integrate_polar(case.x0, z[1], case.stage, 0.0, 0.0)
    p = np.array([math.cos(z[0]), math.sin(z[0])])
    lambda_mass = -case.stage.kappa / coast_state[3]
    final_joint = integrate_primer(
        coast_state,
        np.array([z[3], p[0], p[1], lambda_mass]),
        z[2],
        case.stage,
        1.0,
    )
    residual = explicit_residual(case, z)
    return PrimerResult(
        success=bool(result.success and np.linalg.norm(residual) < 2e-6),
        message=str(result.message),
        z=z,
        x_final=final_joint[:4],
        residual=residual,
        burn_start=float(z[1]),
        burn_time=float(z[2]),
        switch_time=float(z[1]),
        switch_times=[float(z[1])],
        event_count=1,
    )


def coast_costate_guess(
    case: Case,
    coast_time: float,
    alpha_at_burn: float,
    lambda_rho_at_burn: float,
) -> tuple[Array, Array]:
    burn_state = integrate_polar(case.x0, coast_time, case.stage, 0.0, 0.0)
    burn_costate = np.array(
        [
            lambda_rho_at_burn,
            math.cos(alpha_at_burn),
            math.sin(alpha_at_burn),
            -case.stage.kappa / burn_state[3],
        ]
    )
    joint = np.concatenate((burn_state, burn_costate))
    solution = solve_ivp(
        primer_rhs,
        (coast_time, 0.0),
        joint,
        args=(case.stage.gamma, case.stage.kappa, 0.0),
        method="DOP853",
        rtol=2e-11,
        atol=2e-13,
    )
    if not solution.success:
        raise ValueError(solution.message)
    initial_joint = solution.y[:, -1]
    scale = norm(initial_joint[5:7])
    initial_joint[5:8] /= scale
    initial_joint[4] /= scale
    return initial_joint[:4], initial_joint[4:]


def switch_function(joint: Array, kappa: float) -> float:
    mass = joint[3]
    p_r = joint[5]
    p_t = joint[6]
    lambda_mass = joint[7]
    return float(math.hypot(p_r, p_t) / mass + lambda_mass / kappa)


def implicit_propagate(
    case: Case,
    parameters: Array,
) -> tuple[Array, list[float], int, Array, float, list[PrimerArc]]:
    alpha0, final_time, lambda_rho, lambda_mass = parameters
    initial_state = case.x0
    costate = np.array([lambda_rho, math.cos(alpha0), math.sin(alpha0), lambda_mass])
    joint = np.concatenate((initial_state, costate))
    time = 0.0
    throttle = 1.0 if switch_function(joint, case.stage.kappa) > 0.0 else 0.0
    switch_times: list[float] = []
    thrust_time = 0.0
    event_count = 0
    arcs: list[PrimerArc] = []

    for _ in range(20):
        if time >= final_time - 1e-12:
            break
        segment_start = time
        direction = -1.0 if throttle > 0.5 else 1.0

        def switch_event(
            event_time: float,
            event_joint: Array,
            *_args: float,
            segment_start: float = segment_start,
            direction: float = direction,
        ) -> float:
            # Avoid rediscovering the event at the initial point of the next
            # arc.  The tiny signed offset preserves the intended crossing
            # direction while the integrator takes its first step.
            if event_time <= segment_start + 1e-9:
                return -direction * 1e-10
            return switch_function(event_joint, case.stage.kappa)

        switch_event.terminal = True  # type: ignore[attr-defined]
        switch_event.direction = direction  # type: ignore[attr-defined]
        solution = solve_ivp(
            primer_rhs,
            (time, final_time),
            joint,
            args=(case.stage.gamma, case.stage.kappa, throttle),
            events=switch_event,
            method="DOP853",
            rtol=2e-11,
            atol=2e-13,
            dense_output=True,
        )
        if not solution.success:
            raise ValueError(solution.message)
        end_time = float(solution.t[-1])
        thrust_time += throttle * (end_time - segment_start)
        arcs.append(
            PrimerArc(
                start=segment_start,
                end=end_time,
                throttle=throttle,
                times=np.asarray(solution.t, dtype=float),
                joint=np.asarray(solution.y, dtype=float),
            )
        )
        joint = solution.y[:, -1]
        time = end_time
        if solution.t_events is None:
            raise ValueError("event output was not created")
        if len(solution.t_events[0]) == 0:
            break
        event_count += 1
        switch_times.append(time)
        throttle = 1.0 - throttle
    else:
        raise ValueError("too many switching events")

    return joint[:4], switch_times, event_count, joint, thrust_time, arcs


def implicit_residual(case: Case, parameters: Array) -> Array:
    try:
        (
            _state,
            _switch_times,
            _events,
            final_joint,
            _thrust_time,
            _arcs,
        ) = implicit_propagate(case, parameters)
        initial_costate = np.array(
            [
                parameters[2],
                math.cos(parameters[0]),
                math.sin(parameters[0]),
                parameters[3],
            ]
        )
        initial_joint = np.concatenate((case.x0, initial_costate))
        initial_throttle = (
            1.0 if switch_function(initial_joint, case.stage.kappa) > 0.0 else 0.0
        )
        return np.array(
            [
                final_joint[0] - 1.0,
                final_joint[1],
                final_joint[2] - 1.0,
                primer_hamiltonian(
                    initial_joint,
                    case.stage.gamma,
                    case.stage.kappa,
                    initial_throttle,
                ),
            ]
        )
    except (FloatingPointError, ValueError, ZeroDivisionError):
        return np.full(4, 1e3)


def implicit_initial_guess(case: Case, explicit: PrimerResult) -> Array:
    # Back-propagate the explicit burn-start primer guess through the estimated
    # coast.  This keeps alpha_0, t_f, and lambda_rho,0 informed rather than
    # random, while giving lambda_eta,0 the switch-based estimate requested by
    # the research plan.
    _initial_state, initial_costate = coast_costate_guess(
        case,
        explicit.burn_start,
        float(explicit.z[0]),
        float(explicit.z[3]),
    )
    return np.array(
        [
            math.atan2(initial_costate[2], initial_costate[1]),
            explicit.burn_start + explicit.burn_time,
            initial_costate[0],
            initial_costate[3],
        ]
    )


def implicit_solve_from_initial(case: Case, initial: Array) -> PrimerResult:
    lower = np.array([-math.pi, 1e-5, -100.0, -100.0])
    upper = np.array([math.pi, case.first_arc_limit, 100.0, 100.0])

    def residual_function(parameters: Array) -> Array:
        return implicit_residual(case, parameters)

    result = least_squares(
        residual_function,
        initial,
        bounds=(lower, upper),
        x_scale="jac",
        ftol=2e-12,
        xtol=2e-12,
        gtol=2e-12,
        max_nfev=500,
    )
    z = np.asarray(result.x, dtype=float)
    (
        state,
        switch_times,
        event_count,
        _final_joint,
        thrust_time,
        _arcs,
    ) = implicit_propagate(case, z)
    residual = implicit_residual(case, z)
    return PrimerResult(
        success=bool(result.success and np.linalg.norm(residual) < 2e-6),
        message=str(result.message),
        z=z,
        x_final=state,
        residual=residual,
        burn_start=0.0 if not switch_times else switch_times[0],
        burn_time=thrust_time,
        switch_time=None if not switch_times else switch_times[0],
        switch_times=switch_times,
        event_count=event_count,
    )


def implicit_solve(case: Case, explicit: PrimerResult) -> PrimerResult:
    return implicit_solve_from_initial(case, implicit_initial_guess(case, explicit))


def relaxed_seed_from_primer(
    case: Case, primer: PrimerResult, n_intervals: int
) -> Array:
    """Sample an event-based primer trajectory into a relaxed-control mesh."""
    _state, _switches, _events, _final, _thrust, arcs = implicit_propagate(
        case, primer.z
    )
    horizon = float(primer.z[1])
    centers = (np.arange(n_intervals) + 0.5) * horizon / n_intervals
    throttles = np.zeros(n_intervals, dtype=float)
    angles = np.zeros(n_intervals, dtype=float)
    for index, center in enumerate(centers):
        arc = next(
            (
                candidate
                for candidate in arcs
                if candidate.start - 1e-12 <= center <= candidate.end + 1e-12
            ),
            arcs[-1],
        )
        joint = np.array(
            [
                np.interp(center, arc.times, arc.joint[row])
                for row in range(arc.joint.shape[0])
            ]
        )
        throttles[index] = arc.throttle
        angles[index] = math.atan2(joint[6], joint[5])
    return np.concatenate(([horizon], throttles, angles), dtype=float)


def explicit_switch_diagnostics(
    case: Case, primer: PrimerResult
) -> dict[str, float | bool]:
    alpha0, coast_time, burn_time, lambda_rho = primer.z
    coast_state = integrate_polar(case.x0, coast_time, case.stage, 0.0, 0.0)
    joint = np.concatenate(
        (
            coast_state,
            [
                lambda_rho,
                math.cos(alpha0),
                math.sin(alpha0),
                -case.stage.kappa / coast_state[3],
            ],
        )
    )
    burn_solution = solve_ivp(
        primer_rhs,
        (0.0, burn_time),
        joint,
        args=(case.stage.gamma, case.stage.kappa, 1.0),
        method="DOP853",
        rtol=2e-11,
        atol=2e-13,
    )
    coast_solution = solve_ivp(
        primer_rhs,
        (coast_time, 0.0),
        joint,
        args=(case.stage.gamma, case.stage.kappa, 0.0),
        method="DOP853",
        rtol=2e-11,
        atol=2e-13,
    )
    burn_phi = np.array(
        [
            switch_function(burn_solution.y[:, i], case.stage.kappa)
            for i in range(burn_solution.y.shape[1])
        ]
    )
    coast_phi = np.array(
        [
            switch_function(coast_solution.y[:, i], case.stage.kappa)
            for i in range(coast_solution.y.shape[1])
        ]
    )
    return {
        "coast_phi_min": float(coast_phi.min()),
        "coast_phi_max": float(coast_phi.max()),
        "burn_phi_min": float(burn_phi.min()),
        "burn_phi_max": float(burn_phi.max()),
        "satisfies_candidate_sign": bool(
            coast_phi.max() <= 1e-8 and burn_phi.min() >= -1e-8
        ),
    }


def finite_difference_gradient(
    function: Callable[[Array], float], point: Array, step: float = 2e-6
) -> Array:
    gradient = np.empty_like(point)
    for index in range(len(point)):
        delta = np.zeros_like(point)
        delta[index] = step
        gradient[index] = (function(point + delta) - function(point - delta)) / (
            2.0 * step
        )
    return gradient


def validate_equations() -> dict[str, float]:
    stage = Stage(gamma=1.8, kappa=1.35, max_burn=0.4)
    state = np.array([0.83, 0.21, 0.92, 1.0])
    duration = 0.17
    alpha = 0.28
    polar_final = integrate_polar(state, duration, stage, 1.0, alpha)
    cart_final = cartesian_to_polar(
        integrate_cartesian(polar_to_cartesian(state), duration, stage, 1.0, alpha)
    )
    polar_cartesian_error = norm(polar_final - cart_final)

    joint = np.array([*state, 0.37, -0.72, 0.69, -1.1])
    gamma = stage.gamma
    kappa = stage.kappa

    def h_of_state(point: Array) -> float:
        return primer_hamiltonian(np.concatenate((point, joint[4:])), gamma, kappa, 1.0)

    numerical_gradient = finite_difference_gradient(h_of_state, state)
    analytic = primer_rhs(0.0, joint, gamma, kappa, 1.0)[4:]
    conventional_costate_dot = np.array(
        [analytic[0], -analytic[1], -analytic[2], analytic[3]]
    )
    costate_error = norm(conventional_costate_dot + numerical_gradient)

    # The same normalized trajectory is integrated in two arbitrary physical
    # unit systems and converted back.  This catches hidden dimensional terms.
    normalized_physical_errors: list[float] = []
    for radius_scale, velocity_scale, mass_scale in (
        (680_000.0, 2_300.0, 1_000.0),
        (4_200_000.0, 7_100.0, 8_000.0),
    ):
        time_scale = radius_scale / velocity_scale
        mu = radius_scale * velocity_scale**2
        thrust = stage.gamma * mass_scale * velocity_scale**2 / radius_scale
        exhaust_velocity = stage.kappa * velocity_scale
        physical_initial = np.array(
            [
                state[0] * radius_scale,
                state[1] * velocity_scale,
                state[2] * velocity_scale,
                state[3] * mass_scale,
            ]
        )
        physical_final = integrate_physical_polar(
            physical_initial,
            duration * time_scale,
            mu,
            thrust,
            exhaust_velocity,
            1.0,
            alpha,
        )
        converted = np.array(
            [
                physical_final[0] / radius_scale,
                physical_final[1] / velocity_scale,
                physical_final[2] / velocity_scale,
                physical_final[3] / mass_scale,
            ]
        )
        normalized_physical_errors.append(norm(converted - polar_final))

    rng = np.random.default_rng(20260826)
    random_coordinate_errors: list[float] = []
    random_costate_errors: list[float] = []
    for _ in range(32):
        random_stage = Stage(
            gamma=float(rng.uniform(0.8, 4.0)),
            kappa=float(rng.uniform(1.0, 2.0)),
            max_burn=0.25,
        )
        random_state = np.array(
            [
                rng.uniform(0.62, 0.94),
                rng.uniform(0.04, 0.35),
                rng.uniform(0.75, 1.25),
                1.0,
            ]
        )
        random_duration = float(rng.uniform(0.02, 0.16))
        random_alpha = float(rng.uniform(-math.pi, math.pi))
        random_polar = integrate_polar(
            random_state,
            random_duration,
            random_stage,
            1.0,
            random_alpha,
        )
        random_cartesian = cartesian_to_polar(
            integrate_cartesian(
                polar_to_cartesian(random_state),
                random_duration,
                random_stage,
                1.0,
                random_alpha,
            )
        )
        random_coordinate_errors.append(norm(random_polar - random_cartesian))

        random_joint = np.array(
            [
                *random_state,
                rng.uniform(-2.0, 2.0),
                rng.uniform(-1.0, 1.0),
                rng.uniform(-1.0, 1.0),
                rng.uniform(-2.0, 0.0),
            ]
        )

        def random_hamiltonian(
            point: Array,
            random_joint: Array = random_joint,
            random_stage: Stage = random_stage,
        ) -> float:
            return primer_hamiltonian(
                np.concatenate((point, random_joint[4:])),
                random_stage.gamma,
                random_stage.kappa,
                1.0,
            )

        random_gradient = finite_difference_gradient(random_hamiltonian, random_state)
        random_analytic = primer_rhs(
            0.0,
            random_joint,
            random_stage.gamma,
            random_stage.kappa,
            1.0,
        )[4:]
        random_costate_dot = np.array(
            [
                random_analytic[0],
                -random_analytic[1],
                -random_analytic[2],
                random_analytic[3],
            ]
        )
        random_costate_errors.append(norm(random_costate_dot + random_gradient))

    coast_solution = solve_ivp(
        polar_rhs,
        (0.0, 0.7),
        state,
        args=(stage.gamma, stage.kappa, 0.0, 0.0),
        method="DOP853",
        rtol=2e-12,
        atol=2e-14,
    )
    coast_initial = coast_solution.y[:, 0]
    coast_final = coast_solution.y[:, -1]
    energy_initial = (
        0.5 * (coast_initial[1] ** 2 + coast_initial[2] ** 2) - 1.0 / coast_initial[0]
    )
    energy_final = (
        0.5 * (coast_final[1] ** 2 + coast_final[2] ** 2) - 1.0 / coast_final[0]
    )
    angular_momentum_initial = coast_initial[0] * coast_initial[2]
    angular_momentum_final = coast_final[0] * coast_final[2]

    return {
        "polar_cartesian_error": polar_cartesian_error,
        "costate_gradient_error": costate_error,
        "normalization_error": max(normalized_physical_errors),
        "random_polar_cartesian_max": max(random_coordinate_errors),
        "random_costate_gradient_max": max(random_costate_errors),
        "coast_energy_drift": abs(energy_final - energy_initial),
        "coast_angular_momentum_drift": abs(
            angular_momentum_final - angular_momentum_initial
        ),
    }


def contiguous_control_seed(case: Case, n_intervals: int) -> Array:
    """Build a full-horizon relaxed-control seed from a contiguous burn."""
    direct = direct_solve(case, max(4, n_intervals // 2))
    horizon = case.first_arc_limit
    interval = horizon / n_intervals
    q = np.zeros(n_intervals, dtype=float)
    angles = np.zeros(n_intervals, dtype=float)
    for index in range(n_intervals):
        center = (index + 0.5) * interval
        if direct.coast_time <= center <= direct.coast_time + direct.burn_time:
            q[index] = 1.0
            burn_fraction = (center - direct.coast_time) / direct.burn_time
            old_index = min(
                direct.n_intervals - 1,
                int(burn_fraction * direct.n_intervals),
            )
            angles[index] = direct.angles[old_index]
    return np.concatenate(([horizon], q, angles), dtype=float)


def propagate_relaxed_control(case: Case, parameters: Array) -> Array:
    horizon = float(parameters[0])
    n_intervals = (len(parameters) - 1) // 2
    throttles = parameters[1 : 1 + n_intervals]
    angles = parameters[1 + n_intervals :]
    state = case.x0.copy()
    interval = horizon / n_intervals
    for throttle, alpha in zip(throttles, angles, strict=True):
        state = integrate_polar(
            state,
            interval,
            case.stage,
            float(throttle),
            float(alpha),
        )
    return state


def relaxed_direct_solve(
    case: Case,
    n_intervals: int,
    previous: RelaxedResult | None = None,
    initial: Array | None = None,
) -> RelaxedResult:
    if initial is not None:
        control_initial = initial
    elif previous is None:
        control_initial = contiguous_control_seed(case, n_intervals)
    else:
        old_centers = (np.arange(previous.n_intervals) + 0.5) / previous.n_intervals
        new_centers = (np.arange(n_intervals) + 0.5) / n_intervals
        control_initial = np.concatenate(
            (
                [previous.horizon],
                np.interp(
                    new_centers,
                    old_centers,
                    previous.throttles,
                    left=previous.throttles[0],
                    right=previous.throttles[-1],
                ),
                np.interp(
                    new_centers,
                    old_centers,
                    previous.angles,
                    left=previous.angles[0],
                    right=previous.angles[-1],
                ),
            ),
            dtype=float,
        )
    if len(control_initial) != 1 + 2 * n_intervals:
        raise ValueError("relaxed-control seed has the wrong mesh size")
    initial = control_initial
    lower = np.concatenate(
        (
            [case.time_to_apoapsis * 0.25],
            np.zeros(n_intervals),
            np.full(n_intervals, -math.pi),
        )
    )
    upper = np.concatenate(
        ([case.first_arc_limit], np.ones(n_intervals), np.full(n_intervals, math.pi))
    )

    def evaluate(parameters: Array) -> tuple[Array, Array]:
        try:
            state = propagate_relaxed_control(case, parameters)
            residual = np.array([state[0] - 1.0, state[1], state[2] - 1.0])
            return state, residual
        except (FloatingPointError, ValueError, ZeroDivisionError):
            return np.full(4, np.nan), np.full(3, 1e3)

    def objective(parameters: Array) -> float:
        horizon = parameters[0]
        throttles = parameters[1 : 1 + n_intervals]
        return float(
            case.stage.gamma
            * horizon
            * np.sum(throttles)
            / (case.stage.kappa * n_intervals)
        )

    def residual_function(parameters: Array) -> Array:
        return evaluate(parameters)[1]

    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=list(zip(lower, upper, strict=True)),
        constraints=NonlinearConstraint(residual_function, 0.0, 0.0),
        options={"ftol": 2e-11, "maxiter": 700, "disp": False},
    )
    state, residual = evaluate(result.x)
    throttles = np.asarray(result.x[1 : 1 + n_intervals], dtype=float)
    angles = np.asarray(result.x[1 + n_intervals :], dtype=float)
    return RelaxedResult(
        success=bool(result.success and np.linalg.norm(residual) < 2e-6),
        message=str(result.message),
        horizon=float(result.x[0]),
        throttles=throttles,
        angles=angles,
        x_final=state,
        objective=float(result.fun),
        residual=residual,
        n_intervals=n_intervals,
    )


def primer_seeded_relaxed_solve(
    case: Case, primer: PrimerResult, n_intervals: int
) -> RelaxedResult:
    """Refine an event-based primer solution with direct mesh controls."""
    seed = relaxed_seed_from_primer(case, primer, n_intervals)
    return relaxed_direct_solve(case, n_intervals, initial=seed)


def make_cases() -> list[Case]:
    cases = [
        synthetic_case(
            "synthetic-moderate",
            Stage(gamma=1.8, kappa=1.35, max_burn=0.6),
            rp=0.62,
            ra=0.86,
            true_anomaly=math.radians(118.0),
        ),
        synthetic_case(
            "synthetic-high-thrust",
            Stage(gamma=4.2, kappa=1.35, max_burn=0.3),
            rp=0.66,
            ra=0.88,
            true_anomaly=math.radians(128.0),
        ),
        kerbin_case(),
    ]
    rng = np.random.default_rng(20260826)
    for index in range(8):
        rp = float(rng.uniform(0.58, 0.78))
        ra = float(rng.uniform(max(rp + 0.08, 0.78), 0.96))
        anomaly = float(rng.uniform(math.radians(95.0), math.radians(145.0)))
        gamma = float(rng.uniform(1.0, 3.8))
        kappa = float(rng.uniform(1.05, 1.8))
        max_burn = 0.99 * kappa / gamma
        cases.append(
            synthetic_case(
                f"random-{index:02d}",
                Stage(gamma=gamma, kappa=kappa, max_burn=max_burn),
                rp=rp,
                ra=ra,
                true_anomaly=anomaly,
            )
        )
    return cases


def result_record(
    case: Case,
    direct: DirectResult,
    explicit: PrimerResult | None,
    implicit: PrimerResult | None,
) -> dict[str, object]:
    def primer_arcs(primer: PrimerResult) -> list[dict[str, float]]:
        (
            _state,
            _switches,
            _events,
            _final,
            _thrust,
            arcs,
        ) = implicit_propagate(case, primer.z)
        records: list[dict[str, float]] = []
        for arc in arcs:
            phi_values = np.array(
                [
                    switch_function(arc.joint[:, i], case.stage.kappa)
                    for i in range(arc.joint.shape[1])
                ]
            )
            records.append(
                {
                    "start": arc.start,
                    "end": arc.end,
                    "throttle": arc.throttle,
                    "phi_min": float(phi_values.min()),
                    "phi_max": float(phi_values.max()),
                }
            )
        return records

    record: dict[str, object] = {
        "case": case.name,
        "initial_state": case.x0.tolist(),
        "stage": {
            "gamma": case.stage.gamma,
            "kappa": case.stage.kappa,
            "max_burn": case.stage.max_burn,
        },
        "direct": {
            "success": direct.success,
            "message": direct.message,
            "coast_time": direct.coast_time,
            "burn_time": direct.burn_time,
            "fuel_used": case.stage.gamma * direct.burn_time / case.stage.kappa,
            "objective": direct.objective,
            "residual_norm": norm(direct.residual),
            "final_state": direct.x_final.tolist(),
            "intervals": direct.n_intervals,
        },
    }
    if explicit is not None:
        record["explicit"] = {
            "success": explicit.success,
            "message": explicit.message,
            "parameters": explicit.z.tolist(),
            "residual_norm": norm(explicit.residual),
            "burn_start": explicit.burn_start,
            "burn_time": explicit.burn_time,
            "fuel_used": case.stage.gamma * explicit.burn_time / case.stage.kappa,
            "switch_times": explicit.switch_times,
            "switch_diagnostics": explicit_switch_diagnostics(case, explicit),
            "final_state": explicit.x_final.tolist(),
        }
    if implicit is not None:
        record["implicit"] = {
            "success": implicit.success,
            "message": implicit.message,
            "parameters": implicit.z.tolist(),
            "residual_norm": norm(implicit.residual),
            "burn_start": implicit.burn_start,
            "burn_time": implicit.burn_time,
            "fuel_used": case.stage.gamma * implicit.burn_time / case.stage.kappa,
            "switch_time": implicit.switch_time,
            "switch_times": implicit.switch_times,
            "events": implicit.event_count,
            "arcs": primer_arcs(implicit),
            "final_state": implicit.x_final.tolist(),
        }
    return record


def run_solve_cases(
    cases: Iterable[Case], intervals: list[int]
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for case in cases:
        direct_results: list[DirectResult] = []
        previous: DirectResult | None = None
        for n_intervals in intervals:
            previous = direct_solve(case, n_intervals, previous)
            direct_results.append(previous)
        direct = direct_results[-1]
        explicit: PrimerResult | None
        implicit: PrimerResult | None
        try:
            explicit = explicit_solve(case, direct)
        except (FloatingPointError, ValueError, ZeroDivisionError) as error:
            print(f"{case.name}: explicit failed: {error}")
            explicit = None
        if explicit is not None:
            try:
                implicit = implicit_solve(case, explicit)
            except (FloatingPointError, ValueError, ZeroDivisionError) as error:
                print(f"{case.name}: implicit failed: {error}")
                implicit = None
        else:
            implicit = None
        for direct_result in direct_results:
            print(
                f"{case.name:22s} direct N={direct_result.n_intervals:2d} "
                f"ok={direct_result.success!s:5s} "
                f"burn={direct_result.burn_time:.8f} "
                f"res={norm(direct_result.residual):.3e}"
            )
        if explicit is not None:
            print(
                f"{case.name:22s} explicit             "
                f"ok={explicit.success!s:5s} "
                f"burn={explicit.burn_time:.8f} "
                f"res={norm(explicit.residual):.3e}"
            )
        if implicit is not None:
            print(
                f"{case.name:22s} implicit             "
                f"ok={implicit.success!s:5s} "
                f"burn={implicit.burn_time:.8f} "
                f"res={norm(implicit.residual):.3e}"
            )
        records.append(result_record(case, direct, explicit, implicit))
    return records


def run_relaxed_cases(
    cases: Iterable[Case], intervals: list[int]
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for case in cases:
        previous: RelaxedResult | None = None
        results: list[RelaxedResult] = []
        for n_intervals in intervals:
            previous = relaxed_direct_solve(case, n_intervals, previous)
            results.append(previous)
            print(
                f"{case.name:22s} relaxed N={n_intervals:2d} "
                f"ok={previous.success!s:5s} "
                f"fuel={previous.objective:.8f} "
                f"res={norm(previous.residual):.3e} "
                f"q-range=({previous.throttles.min():.3f},"
                f"{previous.throttles.max():.3f})"
            )
        final = results[-1]
        records.append(
            {
                "case": case.name,
                "intervals": [
                    {
                        "n": result.n_intervals,
                        "success": result.success,
                        "message": result.message,
                        "horizon": result.horizon,
                        "fuel": result.objective,
                        "residual_norm": norm(result.residual),
                        "final_state": result.x_final.tolist(),
                        "throttles": result.throttles.tolist(),
                        "angles": result.angles.tolist(),
                    }
                    for result in results
                ],
                "final": {
                    "fuel": final.objective,
                    "final_mass": float(final.x_final[3]),
                },
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("equations", "solve", "relaxed", "all"), default="all"
    )
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--intervals", type=int, nargs="+", default=[8, 16])
    parser.add_argument("--output", type=str, default="")
    arguments = parser.parse_args()

    output: dict[str, object] = {}
    if arguments.mode in ("equations", "all"):
        equations = validate_equations()
        print("equations", json.dumps(equations, sort_keys=True))
        output["equations"] = equations
    if arguments.mode in ("solve", "all"):
        records = run_solve_cases(make_cases()[: arguments.limit], arguments.intervals)
        output["cases"] = records
    if arguments.mode in ("relaxed", "all"):
        records = run_relaxed_cases(
            make_cases()[: arguments.limit], arguments.intervals
        )
        output["relaxed_cases"] = records
    if arguments.output:
        with open(arguments.output, "w", encoding="utf-8") as output_file:
            json.dump(output, output_file, indent=2)
            output_file.write("\n")


if __name__ == "__main__":
    main()
