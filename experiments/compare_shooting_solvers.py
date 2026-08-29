#!/usr/bin/env python3
"""Compare local shooting solvers using informed single-stage guesses."""

from __future__ import annotations

import argparse
import json
import time
from contextlib import suppress
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares, root
from single_stage_research import (
    direct_solve,
    explicit_solve,
    implicit_initial_guess,
    implicit_propagate,
    implicit_residual,
    make_cases,
    norm,
)

SolverMethod = Literal["hybr", "lm", "least_squares"]
Array = NDArray[np.float64]


def solve_one(case_name: str, method: SolverMethod, factor: float) -> dict[str, object]:
    case = next(case for case in make_cases() if case.name == case_name)
    direct = None
    for mesh in (4, 8, 16):
        direct = direct_solve(case, mesh, direct)
    if direct is None or not direct.success:
        raise ValueError(f"direct seed failed for {case.name}")
    explicit = explicit_solve(case, direct)
    initial = implicit_initial_guess(case, explicit)
    initial[3] *= factor
    started = time.perf_counter()
    message = ""
    result: Any

    def residual_function(parameters: Array) -> Array:
        return implicit_residual(case, parameters)

    if method == "least_squares":
        result = least_squares(
            residual_function,
            initial,
            bounds=(
                np.array([-np.pi, 1e-5, -100.0, -100.0]),
                np.array([np.pi, case.first_arc_limit, 100.0, 100.0]),
            ),
            x_scale="jac",
            ftol=2e-12,
            xtol=2e-12,
            gtol=2e-12,
            max_nfev=500,
        )
        parameters = np.asarray(result.x, dtype=float)
        success = bool(result.success)
        message = str(result.message)
    else:
        if method == "hybr":
            options: Any = {"maxfev": 1500}
            result = root(
                residual_function,
                initial,
                method="hybr",
                options=options,
            )
        else:
            options = {"maxiter": 1500}
            result = root(
                residual_function,
                initial,
                method="lm",
                options=options,
            )
        parameters = np.asarray(result.x, dtype=float)
        success = bool(result.success)
        message = str(result.message)
    elapsed = time.perf_counter() - started
    residual = implicit_residual(case, parameters)
    switches: list[float] = []
    events = 0
    fuel = float("nan")
    with suppress(FloatingPointError, ValueError, ZeroDivisionError):
        _state, switches, events, _final, thrust_time, _arcs = implicit_propagate(
            case, parameters
        )
        fuel = case.stage.gamma * thrust_time / case.stage.kappa
    accepted = bool(success and norm(residual) < 2e-6)
    return {
        "case": case.name,
        "method": method,
        "lambda_eta_factor": factor,
        "solver_success": success,
        "accepted": accepted,
        "message": message,
        "elapsed_seconds": elapsed,
        "parameters": parameters.tolist(),
        "residual_norm": norm(residual),
        "fuel_used": fuel,
        "switch_times": switches,
        "event_count": events,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--output", type=str, default="")
    arguments = parser.parse_args()
    methods: tuple[SolverMethod, ...] = ("hybr", "lm", "least_squares")
    factors = [0.5, 1.0, 2.0]
    cases = make_cases()[: arguments.limit]
    records: list[dict[str, object]] = []
    for case in cases:
        for method in methods:
            for factor in factors:
                record = solve_one(case.name, method, factor)
                records.append(record)
                print(
                    f"{case.name:22s} {method:13s} factor={factor:3.1f} "
                    f"ok={record['accepted']!s:5s} "
                    f"res={record['residual_norm']:.3e} "
                    f"fuel={record['fuel_used']:.6f}"
                )
    print(json.dumps(records, indent=2))
    if arguments.output:
        with open(arguments.output, "w", encoding="utf-8") as output_file:
            json.dump(records, output_file, indent=2)
            output_file.write("\n")


if __name__ == "__main__":
    main()
