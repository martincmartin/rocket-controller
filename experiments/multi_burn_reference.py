#!/usr/bin/env python3
"""Test whether adding a third full-thrust arc improves a two-burn reference."""

from __future__ import annotations

import argparse
import json
import math
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from single_stage_research import Case, make_cases, norm
from two_burn_reference import (
    get_primer,
    get_restricted_direct,
    solve_two_burn,
    solve_two_burn_multistart,
)

Array = NDArray[np.float64]


def propagate(case: Case, parameters: Array, n_burns: int, n_angles: int) -> Array:
    from single_stage_research import integrate_polar

    start = float(parameters[0])
    timing = parameters[1 : 2 * n_burns]
    angles = parameters[2 * n_burns :]
    state = integrate_polar(case.x0, start, case.stage, 0.0, 0.0)
    for burn_index in range(n_burns):
        burn_time = float(timing[2 * burn_index])
        for angle in angles[burn_index * n_angles : (burn_index + 1) * n_angles]:
            state = integrate_polar(
                state, burn_time / n_angles, case.stage, 1.0, float(angle)
            )
        if burn_index + 1 < n_burns:
            coast_time = float(timing[2 * burn_index + 1])
            state = integrate_polar(state, coast_time, case.stage, 0.0, 0.0)
    return state


def seed_from_two_burn(case: Case, n_angles: int) -> Array:
    direct = get_restricted_direct(case)
    direct_for_seed = direct if direct.success else None
    try:
        primer = get_primer(case)
    except (FloatingPointError, ValueError, ZeroDivisionError):
        two = solve_two_burn_multistart(case, direct_for_seed, n_angles)
    else:
        two = solve_two_burn(case, primer, n_angles)
    timing = two.parameters[:4]
    angles_one = two.parameters[4 : 4 + n_angles]
    angles_two = two.parameters[4 + n_angles :]
    return np.concatenate(
        (
            [timing[0], timing[1], timing[2], timing[3], 0.0, 1e-7],
            angles_one,
            angles_two,
            angles_two,
        ),
        dtype=float,
    )


def solve_three_burn(case: Case, n_angles: int) -> dict[str, object]:
    initial = seed_from_two_burn(case, n_angles)
    n_burns = 3
    timing_lower = [value for _ in range(n_burns - 1) for value in (1e-8, 0.0)]
    timing_lower.append(1e-8)
    timing_upper = [
        value
        for _ in range(n_burns - 1)
        for value in (case.stage.max_burn, case.first_arc_limit)
    ]
    timing_upper.append(case.stage.max_burn)
    lower = np.concatenate(
        (
            [0.0],
            timing_lower,
            np.full(n_burns * n_angles, -math.pi),
        )
    )
    upper = np.concatenate(
        (
            [case.time_to_apoapsis],
            timing_upper,
            np.full(n_burns * n_angles, math.pi),
        )
    )

    def terminal_residual(parameters: Array) -> Array:
        try:
            state = propagate(case, parameters, n_burns, n_angles)
            return np.array([state[0] - 1.0, state[1], state[2] - 1.0])
        except (FloatingPointError, ValueError, ZeroDivisionError):
            return np.full(3, 1e3)

    def time_constraint(parameters: Array) -> float:
        return float(case.first_arc_limit - parameters[0] - np.sum(parameters[1:6]))

    def objective(parameters: Array) -> float:
        burn_times = parameters[1:6:2]
        return float(case.stage.gamma * np.sum(burn_times) / case.stage.kappa)

    # SciPy accepts this legacy dictionary form at runtime, but its stubs do not.
    constraints: Any = [
        {"type": "eq", "fun": terminal_residual},
        {"type": "ineq", "fun": time_constraint},
    ]
    options: Any = {"ftol": 2e-11, "maxiter": 900, "disp": False}
    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=list(zip(lower, upper, strict=True)),
        constraints=constraints,
        options=options,
    )
    residual = terminal_residual(result.x)
    total_time = float(result.x[0] + np.sum(result.x[1:6]))
    return {
        "success": bool(
            result.success
            and norm(residual) < 2e-6
            and total_time <= case.first_arc_limit + 1e-8
        ),
        "message": str(result.message),
        "fuel": float(result.fun),
        "residual_norm": norm(residual),
        "final_time": total_time,
        "parameters": np.asarray(result.x, dtype=float).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--output", type=str, default="")
    arguments = parser.parse_args()
    records: list[dict[str, object]] = []
    for case in make_cases()[: arguments.limit]:
        results = [solve_three_burn(case, n) for n in (2, 4)]
        for n_angles, result in zip((2, 4), results, strict=True):
            print(
                f"{case.name:22s} three-burn angles={n_angles} "
                f"ok={result['success']!s:5s} fuel={result['fuel']:.9f} "
                f"res={result['residual_norm']:.3e}"
            )
        records.append({"case": case.name, "results": results})
    print(json.dumps(records, indent=2))
    if arguments.output:
        with open(arguments.output, "w", encoding="utf-8") as output_file:
            json.dump(records, output_file, indent=2)
            output_file.write("\n")


if __name__ == "__main__":
    main()
