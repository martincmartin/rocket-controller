#!/usr/bin/env python3
"""Direct and sigmoid-relaxed single-stage primer-vector experiments.

The experiment is independent of the production simulator.  The indirect
solver has only four shooting variables:

``(alpha_0, final_time, lambda_rho_0, lambda_mass_0)``.

It does not receive burn or coast durations.  The throttle is the smooth
switching law ``expit(S / epsilon)`` throughout one integration.

Run from the repository root with::

    PYTHONPATH=experiments python3 experiments/sigmoid_primer.py
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from itertools import pairwise
from typing import cast

import numpy as np
from initial_guess_comparison import (
    apoapsis,
    impulse_timing_estimate,
    initial_parameters,
    kerbin_case_from_vectors,
)
from numpy.typing import NDArray
from scipy.integrate import solve_ivp, trapezoid
from scipy.integrate._ivp.ivp import OdeResult
from scipy.optimize import NonlinearConstraint, least_squares, minimize
from scipy.special import expit
from single_stage_research import (
    Case,
    DirectResult,
    direct_solve,
    integrate_polar,
    make_cases,
    norm,
)

Array = NDArray[np.float64]

RTOL = 2e-10
ATOL = 2e-12
TIME_SCALE_SECONDS = 298.3854313969623
EPS_SCHEDULE = (1e0, 3e-1, 1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4)
SINGLE_STAGE_BURN_LIMIT_SECONDS = 150.0


@dataclass(frozen=True)
class NoCoastResult:
    success: bool
    message: str
    burn_time: float
    fuel_fraction: float
    residual: Array
    final_state: Array
    n_intervals: int


@dataclass(frozen=True)
class SigmoidResult:
    epsilon: float
    success: bool
    solver_success: bool
    message: str
    parameters: Array
    residual: Array
    final_joint: Array
    switch_times: tuple[float, ...]
    switching_function_initial: float
    switching_function_final: float
    throttle_initial: float
    throttle_final: float
    throttle_min: float
    throttle_max: float
    effective_burn_time: float
    hamiltonian_initial: float
    hamiltonian_final: float
    hamiltonian_drift: float
    nfev: int


def kerbin_cases() -> list[Case]:
    """Return the two existing Kerbin cases and the new coast-first case."""
    first_example = kerbin_case_from_vectors(
        "kerbin-first-example",
        np.array([428392.15435586, -1053.61873734, -455905.93323801]),
        np.array([1.03031015e3, -9.32270447e-1, -1.19588146e2]),
        initial_mass=13057.14453125,
        max_burn_seconds=SINGLE_STAGE_BURN_LIMIT_SECONDS,
    )
    coast_first = kerbin_case_from_vectors(
        "kerbin-coast-first-example",
        np.array([433284.5917063, -704.8282711, -459791.995176]),
        np.array([1323.15860984, 11.49193645, 135.66254872]),
        initial_mass=11777.2275390625,
        max_burn_seconds=SINGLE_STAGE_BURN_LIMIT_SECONDS,
    )
    return [make_cases()[2], first_example, coast_first]


def terminal_residual(state: Array) -> Array:
    return np.array([state[0] - 1.0, state[1], state[2] - 1.0], dtype=float)


def propagate_one_burn_from_initial(
    case: Case, burn_time: float, angles: Array
) -> Array:
    if burn_time <= 0.0:
        raise ValueError("burn time must be positive")
    state = case.x0.copy()
    interval = burn_time / len(angles)
    for angle in angles:
        state = integrate_polar(state, interval, case.stage, 1.0, float(angle))
    return state


def solve_no_initial_coast(
    case: Case, direct: DirectResult, n_intervals: int = 16
) -> NoCoastResult:
    """Solve the direct one-burn problem with its coast duration fixed at zero."""
    old_centers = (np.arange(direct.n_intervals) + 0.5) / direct.n_intervals
    new_centers = (np.arange(n_intervals) + 0.5) / n_intervals
    angles = np.interp(
        new_centers,
        old_centers,
        direct.angles,
        left=direct.angles[0],
        right=direct.angles[-1],
    )
    initial = np.concatenate(([direct.burn_time], angles), dtype=float)
    lower = np.concatenate(([1e-6], np.full(n_intervals, -math.pi)))
    upper = np.concatenate(([case.stage.max_burn], np.full(n_intervals, math.pi)))

    def evaluate(parameters: Array) -> tuple[Array, Array]:
        try:
            state = propagate_one_burn_from_initial(
                case, float(parameters[0]), parameters[1:]
            )
            return state, terminal_residual(state)
        except (FloatingPointError, ValueError, ZeroDivisionError):
            return np.full(4, np.nan), np.full(3, 1e3)

    def objective(parameters: Array) -> float:
        return float(parameters[0])

    def residual_function(parameters: Array) -> Array:
        return evaluate(parameters)[1]

    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=list(zip(lower, upper, strict=True)),
        constraints=NonlinearConstraint(residual_function, 0.0, 0.0),
        options={"ftol": 2e-11, "maxiter": 1000, "disp": False},
    )
    state, residual = evaluate(np.asarray(result.x, dtype=float))
    burn_time = float(result.x[0])
    return NoCoastResult(
        success=bool(result.success and norm(residual) < 2e-6),
        message=str(result.message),
        burn_time=burn_time,
        fuel_fraction=case.stage.gamma * burn_time / case.stage.kappa,
        residual=residual,
        final_state=state,
        n_intervals=n_intervals,
    )


def direct_reference(case: Case) -> tuple[list[DirectResult], NoCoastResult]:
    results: list[DirectResult] = []
    previous: DirectResult | None = None
    for n_intervals in (4, 8, 16, 32):
        previous = direct_solve(case, n_intervals, previous)
        results.append(previous)
    return results, solve_no_initial_coast(case, results[-1])


def direct_two_arc_check(case: Case, direct: DirectResult) -> dict[str, object]:
    """Allow a first powered arc and check whether its duration leaves zero."""
    from two_burn_reference import solve_two_burn

    n_angles = 8
    old_centers = (np.arange(direct.n_intervals) + 0.5) / direct.n_intervals
    new_centers = (np.arange(n_angles) + 0.5) / n_angles
    angles = np.interp(
        new_centers,
        old_centers,
        direct.angles,
        left=direct.angles[0],
        right=direct.angles[-1],
    )
    initial = np.concatenate(
        (
            [0.0, 1e-8, direct.coast_time, direct.burn_time],
            angles,
            angles,
        ),
        dtype=float,
    )
    result = solve_two_burn(case, None, n_angles, initial=initial)
    return {
        "success": result.success,
        "message": result.message,
        "intervals_per_burn": n_angles,
        "start_coast": float(result.parameters[0]),
        "first_burn_time": float(result.parameters[1]),
        "middle_coast": float(result.parameters[2]),
        "second_burn_time": float(result.parameters[3]),
        "first_burn_seconds": float(result.parameters[1]) * TIME_SCALE_SECONDS,
        "second_burn_seconds": float(result.parameters[3]) * TIME_SCALE_SECONDS,
        "fuel_fraction": result.fuel,
        "residual_norm": norm(result.residual),
        "final_state": result.final_state.tolist(),
    }


def initial_sigmoid_guess(case: Case) -> Array:
    """Build an analytic seed without supplying a mode schedule to the solver."""
    if apoapsis(case.x0) < 1.0:
        timing = impulse_timing_estimate(case)
        parameters = initial_parameters(
            case,
            math.atan2(float(case.x0[2]), float(case.x0[1])),
            timing,
        )
        return np.array(
            [parameters[0], np.sum(parameters[1:4]), parameters[4], parameters[5]],
            dtype=float,
        )

    # The current apoapsis is already above the target.  Estimate a single
    # tangential circularization burn at apoapsis for the initial guess.
    apo_state = integrate_polar(case.x0, case.time_to_apoapsis, case.stage, 0.0, 0.0)
    delta_v = max(0.0, 1.0 / math.sqrt(float(apo_state[0])) - float(apo_state[2]))
    burn_time = (
        case.stage.kappa
        / case.stage.gamma
        * (1.0 - math.exp(-delta_v / case.stage.kappa))
    )
    alpha = math.pi / 2.0
    p_r = math.cos(alpha)
    p_t = math.sin(alpha)
    lambda_mass = -case.stage.kappa
    rho, radial_velocity, tangential_velocity, mass = case.x0
    gravity_radial = tangential_velocity**2 / rho - 1.0 / rho**2
    lambda_rho = (
        p_r * gravity_radial
        - p_t * radial_velocity * tangential_velocity / rho
        + case.stage.gamma * (1.0 / mass + lambda_mass / case.stage.kappa)
    ) / radial_velocity
    return np.array(
        [alpha, case.time_to_apoapsis + 0.5 * burn_time, lambda_rho, lambda_mass],
        dtype=float,
    )


def switching_function(joint: Array, kappa: float) -> float:
    mass = float(joint[3])
    primer_length = math.hypot(float(joint[5]), float(joint[6]))
    return primer_length / mass + float(joint[7]) / kappa


def sigmoid_primer_rhs(
    _time: float,
    joint: Array,
    gamma: float,
    kappa: float,
    epsilon: float,
) -> Array:
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
    if rho <= 0.0 or mass <= 0.0:
        return np.full(8, np.nan)
    primer_length = math.hypot(p_r, p_t)
    if primer_length <= 1e-14:
        return np.full(8, np.nan)
    switch = primer_length / mass + lambda_mass / kappa
    throttle = float(expit(switch / epsilon))
    acceleration = throttle * gamma / mass
    return np.array(
        [
            radial_velocity,
            tangential_velocity**2 / rho
            - 1.0 / rho**2
            + acceleration * p_r / primer_length,
            -radial_velocity * tangential_velocity / rho
            + acceleration * p_t / primer_length,
            -throttle * gamma / kappa,
            p_r * (2.0 / rho**3 - tangential_velocity**2 / rho**2)
            + p_t * radial_velocity * tangential_velocity / rho**2,
            lambda_rho + p_t * tangential_velocity / rho,
            -2.0 * p_r * tangential_velocity / rho + p_t * radial_velocity / rho,
            -throttle * gamma * primer_length / mass**2,
        ],
        dtype=float,
    )


def initial_joint(case: Case, parameters: Array) -> Array:
    alpha, _final_time, lambda_rho, lambda_mass = parameters
    return np.concatenate(
        (
            case.x0,
            [
                lambda_rho,
                math.cos(float(alpha)),
                math.sin(float(alpha)),
                lambda_mass,
            ],
        ),
        dtype=float,
    )


def integrate_sigmoid(
    case: Case, parameters: Array, epsilon: float, *, dense_output: bool = False
) -> OdeResult:
    solution = solve_ivp(
        sigmoid_primer_rhs,
        (0.0, float(parameters[1])),
        initial_joint(case, parameters),
        args=(case.stage.gamma, case.stage.kappa, epsilon),
        method="DOP853",
        rtol=RTOL,
        atol=ATOL,
        dense_output=dense_output,
    )
    if not solution.success or not np.all(np.isfinite(solution.y[:, -1])):
        raise ValueError(solution.message)
    return solution


def primer_hamiltonian(
    joint: Array, gamma: float, kappa: float, epsilon: float
) -> float:
    rho, radial_velocity, tangential_velocity, mass = joint[:4]
    lambda_rho, p_r, p_t, lambda_mass = joint[4:]
    primer_length = math.hypot(p_r, p_t)
    switch = primer_length / mass + lambda_mass / kappa
    throttle = float(expit(switch / epsilon))
    gravity_radial = tangential_velocity**2 / rho - 1.0 / rho**2
    return float(
        lambda_rho * radial_velocity
        - p_r * gravity_radial
        + p_t * radial_velocity * tangential_velocity / rho
        - throttle * gamma * switch
    )


def sigmoid_residual(case: Case, parameters: Array, epsilon: float) -> Array:
    """Return terminal orbit errors plus the free-time condition.

    At the exact circular target the non-thrust part of the Hamiltonian is
    zero.  Since ``expit(S / epsilon)`` is strictly positive, ``H(tf)=0`` is
    algebraically equivalent to ``S(tf)=0``.  Using ``S`` directly avoids
    multiplying the residual by a throttle that can be exponentially small on
    a coast.
    """
    try:
        solution = integrate_sigmoid(case, parameters, epsilon)
        final = np.asarray(solution.y[:, -1], dtype=float)
        return np.concatenate(
            (
                terminal_residual(final[:4]),
                [switching_function(final, case.stage.kappa)],
            )
        )
    except (FloatingPointError, ValueError, ZeroDivisionError):
        return np.full(4, 1e3)


def switch_crossings(
    times: Array, values: Array, final_time: float
) -> tuple[float, ...]:
    crossings: list[float] = []
    for index, (first, second) in enumerate(pairwise(values)):
        if first == 0.0:
            crossings.append(float(times[index]))
        elif first * second < 0.0:
            fraction = -first / (second - first)
            crossings.append(
                float(times[index] + fraction * (times[index + 1] - times[index]))
            )
    if abs(float(values[-1])) < 1e-7 and (
        not crossings or abs(crossings[-1] - final_time) > 1e-6
    ):
        crossings.append(final_time)
    return tuple(crossings)


def solve_sigmoid_at_epsilon(
    case: Case, initial: Array, epsilon: float
) -> SigmoidResult:
    lower = np.array([-math.pi, 1e-5, -100.0, -100.0], dtype=float)
    upper = np.array([math.pi, case.first_arc_limit, 100.0, 100.0], dtype=float)

    def residual_function(parameters: Array) -> Array:
        return sigmoid_residual(case, parameters, epsilon)

    result = least_squares(
        residual_function,
        initial,
        bounds=(lower, upper),
        x_scale="jac",
        ftol=2e-10,
        xtol=2e-10,
        gtol=2e-10,
        max_nfev=1000,
    )
    parameters = np.asarray(result.x, dtype=float)
    residual = sigmoid_residual(case, parameters, epsilon)
    try:
        solution = integrate_sigmoid(case, parameters, epsilon, dense_output=True)
        final_joint = np.asarray(solution.y[:, -1], dtype=float)
        if solution.sol is None:
            raise ValueError("dense output was not created")
        samples = np.linspace(0.0, parameters[1], 1601)
        joints = np.asarray(solution.sol(samples).T, dtype=float)
        switches = np.array(
            [switching_function(joint, case.stage.kappa) for joint in joints]
        )
        throttles = expit(switches / epsilon)
        hamiltonians = np.array(
            [
                primer_hamiltonian(joint, case.stage.gamma, case.stage.kappa, epsilon)
                for joint in joints
            ]
        )
        crossing_times = switch_crossings(samples, switches, float(parameters[1]))
        effective_burn_time = float(trapezoid(throttles, samples))
    except (FloatingPointError, ValueError, ZeroDivisionError):
        final_joint = np.full(8, np.nan)
        switches = np.full(2, np.nan)
        throttles = np.full(2, np.nan)
        hamiltonians = np.full(2, np.nan)
        crossing_times = ()
        effective_burn_time = float("nan")
    return SigmoidResult(
        epsilon=epsilon,
        success=bool(result.success and norm(residual) < 2e-6),
        solver_success=bool(result.success),
        message=str(result.message),
        parameters=parameters,
        residual=residual,
        final_joint=final_joint,
        switch_times=crossing_times,
        switching_function_initial=float(switches[0]),
        switching_function_final=float(switches[-1]),
        throttle_initial=float(throttles[0]),
        throttle_final=float(throttles[-1]),
        throttle_min=float(np.nanmin(throttles)),
        throttle_max=float(np.nanmax(throttles)),
        effective_burn_time=effective_burn_time,
        hamiltonian_initial=float(hamiltonians[0]),
        hamiltonian_final=float(hamiltonians[-1]),
        hamiltonian_drift=float(hamiltonians[-1] - hamiltonians[0]),
        nfev=int(result.nfev),
    )


def continuation(case: Case) -> list[SigmoidResult]:
    parameters = initial_sigmoid_guess(case)
    results: list[SigmoidResult] = []
    for epsilon in EPS_SCHEDULE:
        result = solve_sigmoid_at_epsilon(case, parameters, epsilon)
        results.append(result)
        parameters = result.parameters
    return results


def direct_record(case: Case, result: DirectResult) -> dict[str, object]:
    return {
        "intervals": result.n_intervals,
        "success": result.success,
        "message": result.message,
        "coast_time": result.coast_time,
        "burn_time": result.burn_time,
        "coast_seconds": result.coast_time * TIME_SCALE_SECONDS,
        "burn_seconds": result.burn_time * TIME_SCALE_SECONDS,
        "fuel_fraction": case.stage.gamma * result.burn_time / case.stage.kappa,
        "residual_norm": norm(result.residual),
        "final_state": result.x_final.tolist(),
    }


def sigmoid_record(case: Case, result: SigmoidResult) -> dict[str, object]:
    return {
        "epsilon": result.epsilon,
        "success": result.success,
        "solver_success": result.solver_success,
        "message": result.message,
        "parameters": result.parameters.tolist(),
        "residual_norm": norm(result.residual),
        "final_state": result.final_joint[:4].tolist(),
        "final_mass": float(result.final_joint[3]),
        "fuel_fraction": float(1.0 - result.final_joint[3]),
        "switch_times": list(result.switch_times),
        "switching_function_initial": result.switching_function_initial,
        "switching_function_final": result.switching_function_final,
        "throttle_initial": result.throttle_initial,
        "throttle_final": result.throttle_final,
        "throttle_min": result.throttle_min,
        "throttle_max": result.throttle_max,
        "effective_burn_time": result.effective_burn_time,
        "effective_burn_seconds": result.effective_burn_time * TIME_SCALE_SECONDS,
        "hamiltonian_initial": result.hamiltonian_initial,
        "hamiltonian_final": result.hamiltonian_final,
        "hamiltonian_drift": result.hamiltonian_drift,
        "nfev": result.nfev,
    }


def run() -> dict[str, object]:
    cases = kerbin_cases()
    coast_first = cases[-1]
    direct_results, no_coast = direct_reference(coast_first)
    two_arc = direct_two_arc_check(coast_first, direct_results[-1])
    continuation_results = {
        case.name: [sigmoid_record(case, result) for result in continuation(case)]
        for case in cases
    }
    return {
        "normalization": {
            "target_radius_m": 680000.0,
            "velocity_scale_m_per_s": 2278.931638238564,
            "time_scale_s": TIME_SCALE_SECONDS,
            "single_stage_burn_limit_seconds": SINGLE_STAGE_BURN_LIMIT_SECONDS,
            "epsilon_schedule": list(EPS_SCHEDULE),
        },
        "coast_first_case": {
            "name": coast_first.name,
            "initial_state": coast_first.x0.tolist(),
            "gamma": coast_first.stage.gamma,
            "kappa": coast_first.stage.kappa,
            "time_to_apoapsis": coast_first.time_to_apoapsis,
            "time_to_apoapsis_seconds": coast_first.time_to_apoapsis
            * TIME_SCALE_SECONDS,
            "initial_apoapsis": apoapsis(coast_first.x0),
            "first_arc_limit": coast_first.first_arc_limit,
            "direct_meshes": [
                direct_record(coast_first, result) for result in direct_results
            ],
            "direct_no_initial_coast": {
                "success": no_coast.success,
                "message": no_coast.message,
                "intervals": no_coast.n_intervals,
                "burn_time": no_coast.burn_time,
                "burn_seconds": no_coast.burn_time * TIME_SCALE_SECONDS,
                "fuel_fraction": no_coast.fuel_fraction,
                "residual_norm": norm(no_coast.residual),
                "final_state": no_coast.final_state.tolist(),
            },
            "direct_two_arc_check": two_arc,
        },
        "sigmoid_continuation": continuation_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="")
    arguments = parser.parse_args()
    output = run()
    continuation_output = cast(
        dict[str, list[dict[str, object]]], output["sigmoid_continuation"]
    )
    for case_name, records in continuation_output.items():
        final = records[-1]
        parameters = cast(list[float], final["parameters"])
        switch_times = cast(list[float], final["switch_times"])
        print(
            f"{case_name:28s} eps=1e-4 "
            f"ok={final['success']!s:5s} "
            f"fuel={final['fuel_fraction']:.9f} "
            f"res={final['residual_norm']:.3e} "
            f"tf={parameters[1]:.9f} "
            f"switches={len(switch_times)}"
        )
    print(json.dumps(output, indent=2))
    if arguments.output:
        with open(arguments.output, "w", encoding="utf-8") as output_file:
            json.dump(output, output_file, indent=2)
            output_file.write("\n")


if __name__ == "__main__":
    main()
