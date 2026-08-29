#!/usr/bin/env python3
"""Inspect energy and angular-momentum effects of the first optimal burn."""

from __future__ import annotations

import argparse
import json
import math

import numpy as np
from initial_guess_comparison import kerbin_case_from_vectors, optimize_reference
from scipy.integrate import solve_ivp, trapezoid
from scipy.integrate._ivp.ivp import OdeResult
from single_stage_research import Case, make_cases, primer_rhs


def angle_difference(first: float, second: float) -> float:
    return math.atan2(math.sin(first - second), math.cos(first - second))


def orbital_elements(state: np.ndarray) -> tuple[float, float, float, float]:
    rho, ur, ut, _mass = state
    energy = 0.5 * (ur * ur + ut * ut) - 1.0 / rho
    semi_major_axis = -1.0 / (2.0 * energy)
    angular_momentum = rho * ut
    eccentricity = math.sqrt(max(0.0, 1.0 + 2.0 * energy * angular_momentum**2))
    return (
        energy,
        angular_momentum,
        semi_major_axis * (1.0 - eccentricity),
        semi_major_axis * (1.0 + eccentricity),
    )


def integrate_arc(
    initial: np.ndarray, duration: float, gamma: float, kappa: float, throttle: float
) -> OdeResult:
    solution = solve_ivp(
        primer_rhs,
        (0.0, duration),
        initial,
        args=(gamma, kappa, throttle),
        method="DOP853",
        rtol=2e-11,
        atol=2e-13,
        dense_output=True,
    )
    if not solution.success or solution.sol is None:
        raise ValueError(solution.message)
    return solution


def analyze_case(case: Case) -> dict[str, object]:
    reference, _direct = optimize_reference(case)
    parameters = np.asarray(reference["parameters"], dtype=float)
    alpha0, burn_one, _coast_gap, _burn_two, lambda_rho, lambda_eta = parameters
    initial_joint = np.array(
        [
            *case.x0,
            lambda_rho,
            math.cos(alpha0),
            math.sin(alpha0),
            lambda_eta,
        ]
    )
    first = integrate_arc(
        initial_joint,
        burn_one,
        case.stage.gamma,
        case.stage.kappa,
        1.0,
    )
    samples = np.linspace(0.0, burn_one, 401)
    if first.sol is None:
        raise ValueError("dense output was not created")
    joints = first.sol(samples)
    states = joints[:4]
    primer_r = joints[5]
    primer_t = joints[6]
    angles = np.arctan2(primer_t, primer_r)
    velocity_angles = np.arctan2(states[2], states[1])
    differences = np.array(
        [
            angle_difference(float(a), float(v))
            for a, v in zip(angles, velocity_angles, strict=True)
        ]
    )
    acceleration = case.stage.gamma / states[3]
    parallel_power = acceleration * (
        states[1] * np.cos(angles) + states[2] * np.sin(angles)
    )
    perpendicular_acceleration = acceleration * np.sin(differences)
    angular_momentum = states[0] * states[2]
    energy = 0.5 * (states[1] ** 2 + states[2] ** 2) - 1.0 / states[0]
    h_dot = states[0] * acceleration * np.sin(angles)
    return {
        "case": case.name,
        "normalized_state": case.x0.tolist(),
        "parameters": parameters.tolist(),
        "first_burn_seconds": burn_one * 298.3854313969623,
        "current_apoapsis_seconds": case.time_to_apoapsis * 298.3854313969623,
        "initial": {
            "primer_angle_rad": float(angles[0]),
            "primer_angle_deg": math.degrees(float(angles[0])),
            "velocity_angle_rad": float(velocity_angles[0]),
            "velocity_angle_deg": math.degrees(float(velocity_angles[0])),
            "angle_difference_rad": float(differences[0]),
            "angle_difference_deg": math.degrees(float(differences[0])),
            "parallel_energy_alignment": math.cos(float(differences[0])),
            "perpendicular_alignment": math.sin(float(differences[0])),
        },
        "first_burn_end": {
            "state": states[:, -1].tolist(),
            "primer_angle_deg": math.degrees(float(angles[-1])),
            "velocity_angle_deg": math.degrees(float(velocity_angles[-1])),
            "angle_difference_deg": math.degrees(float(differences[-1])),
            "energy": float(energy[-1]),
            "angular_momentum": float(angular_momentum[-1]),
            "periapsis": orbital_elements(states[:, -1])[2],
            "apoapsis": orbital_elements(states[:, -1])[3],
        },
        "first_burn_integrals": {
            "energy_change": float(energy[-1] - energy[0]),
            "integrated_thrust_power": float(trapezoid(parallel_power, samples)),
            "angular_momentum_change": float(
                angular_momentum[-1] - angular_momentum[0]
            ),
            "integrated_h_dot": float(trapezoid(h_dot, samples)),
            "integrated_perpendicular_acceleration": float(
                trapezoid(perpendicular_acceleration, samples)
            ),
            "parallel_alignment_min": float(np.cos(differences).min()),
            "parallel_alignment_max": float(np.cos(differences).max()),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="")
    arguments = parser.parse_args()
    first_kerbin = kerbin_case_from_vectors(
        "kerbin-first-example",
        np.array([428392.15435586, -1053.61873734, -455905.93323801]),
        np.array([1.03031015e3, -9.32270447e-1, -1.19588146e2]),
        initial_mass=13057.14453125,
        max_burn_seconds=150.0,
    )
    cases = [make_cases()[2], first_kerbin]
    records = [analyze_case(case) for case in cases]
    print(json.dumps(records, indent=2))
    if arguments.output:
        with open(arguments.output, "w", encoding="utf-8") as output_file:
            json.dump(records, output_file, indent=2)
            output_file.write("\n")


if __name__ == "__main__":
    main()
