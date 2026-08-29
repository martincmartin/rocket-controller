#!/usr/bin/env python3
"""Fixed-sequence two-arc primer shooting, seeded by direct optimization."""

from __future__ import annotations

import argparse
import json
import math

import numpy as np
from scipy.optimize import least_squares
from single_stage_research import (
    Case,
    direct_solve,
    integrate_primer,
    make_cases,
    norm,
    primer_hamiltonian,
    switch_function,
)
from two_burn_reference import solve_two_burn_multistart


def residual(
    case: Case, parameters: np.ndarray, final_time_active: bool = False
) -> np.ndarray:
    alpha, burn_one, coast_gap, burn_two, lambda_rho, lambda_mass = parameters
    initial_costate = np.array(
        [lambda_rho, math.cos(alpha), math.sin(alpha), lambda_mass]
    )
    initial_joint = np.concatenate((case.x0, initial_costate))
    first = integrate_primer(case.x0, initial_costate, burn_one, case.stage, 1.0)
    first_switch = switch_function(first, case.stage.kappa)
    coast = integrate_primer(first[:4], first[4:], coast_gap, case.stage, 0.0)
    second_switch = switch_function(coast, case.stage.kappa)
    final = integrate_primer(coast[:4], coast[4:], burn_two, case.stage, 1.0)
    residual_values = [
        final[0] - 1.0,
        final[1],
        final[2] - 1.0,
        first_switch,
        second_switch,
    ]
    if final_time_active:
        residual_values.append(float(np.sum(parameters[1:4])) - case.first_arc_limit)
    else:
        residual_values.append(
            primer_hamiltonian(initial_joint, case.stage.gamma, case.stage.kappa, 1.0)
        )
    return np.asarray(residual_values, dtype=float)


def seed(case: Case, direct_two_burn: object) -> np.ndarray:
    parameters = direct_two_burn.parameters  # type: ignore[attr-defined]
    alpha = float(parameters[4])
    lambda_mass = -case.stage.kappa
    p_r = math.cos(alpha)
    p_t = math.sin(alpha)
    rho, ur, ut, mass = case.x0
    gravity_kinematic = ut * ut / rho - 1.0 / rho**2
    lambda_rho = (
        p_r * gravity_kinematic
        - p_t * ur * ut / rho
        + case.stage.gamma * (1.0 / mass + lambda_mass / case.stage.kappa)
    ) / ur
    return np.array(
        [
            alpha,
            parameters[1],
            parameters[2],
            parameters[3],
            lambda_rho,
            lambda_mass,
        ]
    )


def solve_from_initial(
    case: Case,
    initial: np.ndarray,
    final_time_active: bool,
    direct_fuel: float | None = None,
) -> dict[str, object]:
    lower = np.array([-math.pi, 1e-8, 0.0, 1e-8, -100.0, -100.0])
    upper = np.array(
        [
            math.pi,
            case.stage.max_burn,
            case.first_arc_limit,
            case.stage.max_burn,
            100.0,
            100.0,
        ]
    )

    def residual_function(parameters: np.ndarray) -> np.ndarray:
        return residual(case, parameters, final_time_active)

    result = least_squares(
        residual_function,
        initial,
        bounds=(lower, upper),
        x_scale="jac",
        ftol=2e-12,
        xtol=2e-12,
        gtol=2e-12,
        max_nfev=800,
    )
    parameters = np.asarray(result.x, dtype=float)
    residual_value = residual(case, parameters, final_time_active)
    total_time = float(np.sum(parameters[1:4]))
    fuel = case.stage.gamma * float(parameters[1] + parameters[3]) / case.stage.kappa
    accepted = bool(
        result.success
        and norm(residual_value) < 2e-6
        and total_time <= case.first_arc_limit + 1e-8
    )
    return {
        "success": accepted,
        "solver_success": bool(result.success),
        "message": str(result.message),
        "parameters": parameters.tolist(),
        "residual_norm": norm(residual_value),
        "final_time": total_time,
        "final_time_active": final_time_active,
        "fuel": fuel,
        "direct_fuel": direct_fuel,
        "relative_fuel_difference": (
            None if direct_fuel is None else fuel / direct_fuel - 1.0
        ),
    }


def solve(case: Case, direct_two_burn: object) -> dict[str, object]:
    initial = seed(case, direct_two_burn)
    direct_parameters = direct_two_burn.parameters  # type: ignore[attr-defined]
    final_time_active = bool(
        float(np.sum(direct_parameters[:4])) >= case.first_arc_limit - 1e-6
    )
    return solve_from_initial(
        case,
        initial,
        final_time_active,
        direct_fuel=float(direct_two_burn.fuel),  # type: ignore[attr-defined]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--output", type=str, default="")
    arguments = parser.parse_args()
    records: list[dict[str, object]] = []
    for case in make_cases()[: arguments.limit]:
        direct = None
        for mesh in (4, 8, 16):
            direct = direct_solve(case, mesh, direct)
        if direct is None or not direct.success:
            raise ValueError(f"restricted direct seed failed for {case.name}")
        direct_two_burn = solve_two_burn_multistart(case, direct, 8)
        shooting = solve(case, direct_two_burn)
        print(
            f"{case.name:22s} two-arc shooting "
            f"ok={shooting['success']!s:5s} "
            f"fuel={shooting['fuel']:.9f} "
            f"direct={shooting['direct_fuel']:.9f} "
            f"res={shooting['residual_norm']:.3e}"
        )
        records.append(
            {
                "case": case.name,
                "shooting": shooting,
                "direct_two_burn": {
                    "fuel": direct_two_burn.fuel,
                    "residual_norm": norm(direct_two_burn.residual),
                    "parameters": direct_two_burn.parameters.tolist(),
                },
            }
        )
    print(json.dumps(records, indent=2))
    if arguments.output:
        with open(arguments.output, "w", encoding="utf-8") as output_file:
            json.dump(records, output_file, indent=2)
            output_file.write("\n")


if __name__ == "__main__":
    main()
