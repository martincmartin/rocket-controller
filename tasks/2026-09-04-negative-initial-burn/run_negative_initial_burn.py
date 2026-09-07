#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Record the fixed on/off/on solve for the coast-first Kerbin case."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from initial_guess_comparison import kerbin_case_from_vectors
from single_stage_research import Case, DirectResult, direct_solve, norm
from two_arc_shooting import residual, seed, solve_from_initial
from two_burn_reference import solve_two_burn_multistart

OUTPUT = Path(__file__).with_name("negative-initial-burn-results.json")


def make_coast_first_case() -> Case:
    return kerbin_case_from_vectors(
        "kerbin-coast-first-example",
        np.array([433284.5917063, -704.8282711, -459791.995176]),
        np.array([1323.15860984, 11.49193645, 135.66254872]),
        initial_mass=11777.2275390625,
        max_burn_seconds=150.0,
    )


def main() -> None:
    case = make_coast_first_case()
    direct: DirectResult | None = None
    direct_records: list[dict[str, object]] = []
    for intervals in (4, 8, 16, 32):
        direct_result = direct_solve(case, intervals, direct)
        direct = direct_result
        direct_records.append(
            {
                "intervals": intervals,
                "success": direct_result.success,
                "coast_time": direct_result.coast_time,
                "burn_time": direct_result.burn_time,
                "fuel": case.stage.gamma * direct_result.burn_time / case.stage.kappa,
                "residual_norm": norm(direct_result.residual),
            }
        )
    if direct is None or not direct.success:
        raise RuntimeError("direct reference did not converge")

    direct_two_burn = solve_two_burn_multistart(case, direct, 8)
    shooting_initial = seed(case, direct_two_burn)
    final_time_active = bool(
        float(np.sum(direct_two_burn.parameters[:4])) >= case.first_arc_limit - 1e-6
    )
    shooting = solve_from_initial(
        case,
        shooting_initial,
        final_time_active,
        direct_fuel=direct_two_burn.fuel,
    )
    initial_parameters = shooting_initial.tolist()

    unconstrained_residual_at_oracle = residual(
        case,
        np.array(
            [
                2.180458606526267,
                0.0,
                0.314482092435442,
                0.195766154962485,
                1.087412012182590,
                -1.370608266588306,
            ],
            dtype=float,
        ),
        final_time_active=False,
    )
    record = {
        "case": {
            "name": case.name,
            "initial_state": case.x0.tolist(),
            "gamma": case.stage.gamma,
            "kappa": case.stage.kappa,
            "max_burn": case.stage.max_burn,
            "time_to_apoapsis": case.time_to_apoapsis,
            "time_to_apoapsis_seconds": case.time_to_apoapsis * 298.3854313969623,
            "first_arc_limit": case.first_arc_limit,
        },
        "solver_bounds": {
            "burn_one_lower": 1e-8,
            "coast_lower": 0.0,
            "burn_two_lower": 1e-8,
            "burn_one_upper": case.stage.max_burn,
            "coast_upper": case.first_arc_limit,
            "burn_two_upper": case.stage.max_burn,
        },
        "direct_one_burn_meshes": direct_records,
        "direct_two_burn_seed": {
            "success": direct_two_burn.success,
            "message": direct_two_burn.message,
            "parameters_start_b1_coast_b2": direct_two_burn.parameters[:4].tolist(),
            "angles": direct_two_burn.parameters[4:].tolist(),
            "fuel": direct_two_burn.fuel,
            "residual_norm": norm(direct_two_burn.residual),
            "final_state": direct_two_burn.final_state.tolist(),
        },
        "shooting_initial": {
            "parameters_alpha_b1_coast_b2_lambda_rho_lambda_mass": initial_parameters,
            "residual_norm": norm(residual(case, shooting_initial, final_time_active)),
            "residual": residual(case, shooting_initial, final_time_active).tolist(),
        },
        "shooting_final": shooting,
        "oracle_at_zero_first_burn": {
            "parameters_alpha_b1_coast_b2_lambda_rho_lambda_mass": [
                2.180458606526267,
                0.0,
                0.314482092435442,
                0.195766154962485,
                1.087412012182590,
                -1.370608266588306,
            ],
            "residual": unconstrained_residual_at_oracle.tolist(),
            "residual_norm": norm(unconstrained_residual_at_oracle),
        },
    }
    OUTPUT.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
