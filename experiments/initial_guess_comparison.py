#!/usr/bin/env python3
"""Compare analytic initial estimates with fixed two-arc primer solutions."""

from __future__ import annotations

import argparse
import json
import math

import numpy as np
from scipy.optimize import brentq
from single_stage_research import (
    Case,
    Stage,
    direct_solve,
    estimated_burn,
    make_cases,
    norm,
)
from two_arc_shooting import solve, solve_from_initial
from two_burn_reference import solve_two_burn_multistart


def kerbin_case_from_vectors(
    name: str,
    position: np.ndarray,
    velocity: np.ndarray,
    initial_mass: float,
    max_burn_seconds: float,
) -> Case:
    mu = 3.5316e12
    target_radius = 680_000.0
    target_velocity = math.sqrt(mu / target_radius)
    time_scale = target_radius / target_velocity
    radius = norm(position)
    radial_hat = position / radius
    radial_velocity = float(np.dot(velocity, radial_hat))
    tangent_velocity = norm(velocity - radial_velocity * radial_hat)
    state = np.array(
        [
            radius / target_radius,
            radial_velocity / target_velocity,
            tangent_velocity / target_velocity,
            1.0,
        ]
    )
    stage = Stage(
        gamma=215_000.0 * target_radius / (initial_mass * target_velocity**2),
        kappa=3138.128 / target_velocity,
        max_burn=max_burn_seconds / time_scale,
    )
    _, time_to_apoapsis = _coast_to_apo(state, stage)
    energy = 0.5 * (state[1] ** 2 + state[2] ** 2) - 1.0 / state[0]
    semi_major_axis = -1.0 / (2.0 * energy)
    return Case(
        name=name,
        x0=state,
        stage=stage,
        time_to_apoapsis=time_to_apoapsis,
        first_arc_limit=time_to_apoapsis
        + 0.75 * 2.0 * math.pi * math.sqrt(semi_major_axis**3),
    )


def _coast_to_apo(state: np.ndarray, stage: Stage) -> tuple[np.ndarray, float]:
    from single_stage_research import coast_to_apoapsis

    return coast_to_apoapsis(state, stage)


def apoapsis(state: np.ndarray) -> float:
    rho, ur, ut, _mass = state
    energy = 0.5 * (ur * ur + ut * ut) - 1.0 / rho
    if energy >= 0.0:
        return float("inf")
    semi_major_axis = -1.0 / (2.0 * energy)
    h = rho * ut
    eccentricity = math.sqrt(max(0.0, 1.0 + 2.0 * energy * h * h))
    return float(semi_major_axis * (1.0 + eccentricity))


def angle_difference(first: float, second: float) -> float:
    return math.atan2(math.sin(first - second), math.cos(first - second))


def prograde_at_apoapsis_angle(state: np.ndarray) -> float:
    rho, ur, ut, _mass = state
    h = rho * ut
    eccentricity_vector = np.array([h * ut - 1.0, -h * ur])
    eccentricity = norm(eccentricity_vector)
    if eccentricity < 1e-12:
        return math.pi / 2.0
    apoapsis = -eccentricity_vector / eccentricity
    prograde = np.array([-apoapsis[1], apoapsis[0]])
    return math.atan2(prograde[1], prograde[0])


def impulse_timing_estimate(case: Case) -> dict[str, float]:
    state = case.x0
    rho, ur, ut, _mass = state
    current_apo = apoapsis(state)
    if not math.isfinite(current_apo) or current_apo >= 1.0:
        raise ValueError("case does not have an apoapsis below the target")

    velocity_magnitude = math.hypot(ur, ut)
    velocity_direction = np.array([ur / velocity_magnitude, ut / velocity_magnitude])

    def target_apo_error(delta_v: float) -> float:
        post_velocity = np.array([ur, ut]) + delta_v * velocity_direction
        return apoapsis(np.array([rho, *post_velocity, 1.0])) - 1.0

    upper = max(velocity_magnitude * 0.1, 0.1)
    while target_apo_error(upper) < 0.0:
        upper *= 1.5
        if upper > 20.0:
            raise ValueError("could not bracket target-apoapsis impulse")
    delta_v_one = brentq(target_apo_error, 0.0, upper)
    eta_after_first = math.exp(-delta_v_one / case.stage.kappa)
    burn_one = case.stage.kappa / case.stage.gamma * (1.0 - eta_after_first)
    post_velocity = np.array([ur, ut]) + delta_v_one * velocity_direction
    post_impulse = np.array([rho, *post_velocity, eta_after_first])
    apo_state, time_to_new_apoapsis = _coast_to_apo(post_impulse, case.stage)
    delta_v_two = max(0.0, 1.0 / math.sqrt(apo_state[0]) - apo_state[2])
    eta_after_second = eta_after_first * math.exp(-delta_v_two / case.stage.kappa)
    burn_two = (
        case.stage.kappa / case.stage.gamma * (eta_after_first - eta_after_second)
    )
    coast_gap = max(0.0, time_to_new_apoapsis - burn_one - 0.5 * burn_two)
    coast_gap = min(
        coast_gap,
        max(0.0, case.first_arc_limit - burn_one - burn_two),
    )
    return {
        "current_apoapsis": current_apo,
        "delta_v_one": delta_v_one,
        "burn_one": burn_one,
        "time_to_new_apoapsis": time_to_new_apoapsis,
        "delta_v_two": delta_v_two,
        "burn_two": burn_two,
        "coast_gap": coast_gap,
        "final_time": burn_one + coast_gap + burn_two,
        "existing_apo_burn_estimate": estimated_burn(case),
    }


def initial_parameters(
    case: Case, alpha: float, timing: dict[str, float]
) -> np.ndarray:
    p_r = math.cos(alpha)
    p_t = math.sin(alpha)
    rho, ur, ut, mass = case.x0
    if abs(ur) < 1e-10:
        lambda_rho = 0.0
    else:
        gravity_kinematic = ut * ut / rho - 1.0 / rho**2
        lambda_rho = (
            p_r * gravity_kinematic
            - p_t * ur * ut / rho
            + case.stage.gamma * (1.0 / mass - 1.0)
        ) / ur
    return np.array(
        [
            alpha,
            timing["burn_one"],
            timing["coast_gap"],
            timing["burn_two"],
            lambda_rho,
            -case.stage.kappa,
        ]
    )


def optimize_reference(case: Case) -> tuple[dict[str, object], dict[str, float]]:
    direct = None
    for mesh in (4, 8, 16):
        direct = direct_solve(case, mesh, direct)
    if direct is None or not direct.success:
        raise ValueError(f"restricted direct seed failed for {case.name}")
    direct_two_burn = solve_two_burn_multistart(case, direct, 8)
    reference = solve(case, direct_two_burn)
    return reference, {
        "direct_fuel": direct_two_burn.fuel,
        "direct_residual": norm(direct_two_burn.residual),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--output", type=str, default="")
    arguments = parser.parse_args()
    kerbin_first = kerbin_case_from_vectors(
        "kerbin-first-example",
        np.array([428392.15435586, -1053.61873734, -455905.93323801]),
        np.array([1.03031015e3, -9.32270447e-1, -1.19588146e2]),
        initial_mass=13057.14453125,
        max_burn_seconds=150.0,
    )
    cases = (
        [*make_cases(), kerbin_first]
        if arguments.all
        else [
            make_cases()[0],
            make_cases()[1],
            make_cases()[2],
            kerbin_first,
        ]
    )
    records: list[dict[str, object]] = []
    for case in cases:
        reference, direct_info = optimize_reference(case)
        timing = impulse_timing_estimate(case)
        alpha_tangential = math.pi / 2.0
        alpha_apo = prograde_at_apoapsis_angle(case.x0)
        alpha_velocity = math.atan2(case.x0[2], case.x0[1])
        estimates = {
            "tangential": initial_parameters(case, alpha_tangential, timing),
            "prograde_at_apoapsis": initial_parameters(case, alpha_apo, timing),
            "velocity_direction": initial_parameters(case, alpha_velocity, timing),
        }
        initial_solves: dict[str, dict[str, object]] = {}
        for name, parameters in estimates.items():
            result = solve_from_initial(
                case,
                parameters,
                final_time_active=timing["final_time"] >= case.first_arc_limit - 1e-6,
            )
            initial_solves[name] = result
        optimal = np.asarray(reference["parameters"], dtype=float)
        records.append(
            {
                "case": case.name,
                "normalized_state": case.x0.tolist(),
                "stage": {
                    "gamma": case.stage.gamma,
                    "kappa": case.stage.kappa,
                    "max_burn": case.stage.max_burn,
                },
                "optimal": {
                    "alpha0": optimal[0],
                    "burn_one": optimal[1],
                    "coast_gap": optimal[2],
                    "burn_two": optimal[3],
                    "lambda_rho0": optimal[4],
                    "lambda_eta0": optimal[5],
                    "fuel": reference["fuel"],
                    "final_time": reference["final_time"],
                    "residual_norm": reference["residual_norm"],
                    "lambda_eta_ratio_to_minus_kappa": optimal[5] / (-case.stage.kappa),
                },
                "angles": {
                    "tangential": alpha_tangential,
                    "prograde_at_apoapsis": alpha_apo,
                    "velocity_direction": alpha_velocity,
                    "optimal": optimal[0],
                    "error_tangential": angle_difference(optimal[0], alpha_tangential),
                    "error_apoapsis": angle_difference(optimal[0], alpha_apo),
                    "error_velocity": angle_difference(optimal[0], alpha_velocity),
                },
                "timing_estimate": timing,
                "timing_error": {
                    "burn_one": timing["burn_one"] - optimal[1],
                    "coast_gap": timing["coast_gap"] - optimal[2],
                    "burn_two": timing["burn_two"] - optimal[3],
                },
                "initial_guess_solves": initial_solves,
                "direct_reference": direct_info,
            }
        )
    print(json.dumps(records, indent=2))
    if arguments.output:
        with open(arguments.output, "w", encoding="utf-8") as output_file:
            json.dump(records, output_file, indent=2)
            output_file.write("\n")


if __name__ == "__main__":
    main()
