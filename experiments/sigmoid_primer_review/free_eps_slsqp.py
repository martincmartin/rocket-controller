#!/usr/bin/env python3
"""Experiment: free-epsilon shooting with SLSQP minimizing negative final
mass, with the four shooting residuals as equality constraints.

The five unknowns are (alpha0, tf, lambda_rho0, lambda_mass0, eps). Four
equality constraints (three terminal conditions plus S(tf)=0) leave one
degree of freedom; minimizing fuel should walk along the constraint manifold
toward small eps, because a sharper sigmoid wastes less fuel in the
transition layers, ideally landing on the bang-bang optimum at the eps
bound. Contrast with `free_eps_shooting.py`, where the plain least-squares
descent stopped at the nearest point of the manifold (eps ~ 0.8-1.0).

Run from the repository root with ``PYTHONPATH=experiments``::

    python3 experiments/sigmoid_primer_review/free_eps_slsqp.py
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import production_solver as ps
import sigmoid_primer as sp
from single_stage_research import Case, implicit_propagate

Array = NDArray[np.float64]

EPS_LOWER = 1e-8
EPS_UPPER = 10.0
MAX_ITERATIONS = 2000
REFERENCES = {
    "kerbin-example": {"fuel": 0.458673471, "tf": 0.648432961},
    "kerbin-first-example": {"fuel": 0.417870055, "tf": 0.522074908},
    "kerbin-coast-first-example": {"fuel": 0.339814106, "tf": 0.510560727},
}


def solve_free_eps_slsqp(case: Case) -> dict[str, Any]:
    guess = sp.initial_sigmoid_guess(case)
    initial = np.concatenate((guess, [1.0]))
    lower = np.array([-math.pi, 1e-5, -100.0, -100.0, EPS_LOWER], dtype=float)
    upper = np.array(
        [math.pi, case.first_arc_limit, 100.0, 100.0, EPS_UPPER], dtype=float
    )
    bounds = list(zip(lower, upper, strict=True))

    def objective(parameters: Array) -> float:
        base = parameters[:4]
        epsilon = float(parameters[4])
        if epsilon <= 0.0 or not np.isfinite(epsilon):
            return 1e3
        try:
            solution = ps.integrate_sigmoid_robust(case, base, epsilon)
            return -float(solution.y[3, -1])
        except (FloatingPointError, ValueError, ZeroDivisionError):
            return 1e3

    def constraints(parameters: Array) -> Array:
        base = parameters[:4]
        epsilon = float(parameters[4])
        return ps.sigmoid_residual_tight(case, base, epsilon, sp.RTOL, sp.ATOL)

    progress: list[list[float]] = []
    started = time.monotonic()

    def record_iterate(iterate: np.ndarray) -> None:
        progress.append(np.asarray(iterate, dtype=float).tolist())

    kwargs: dict[str, Any] = {
        "fun": objective,
        "x0": initial,
        "method": "SLSQP",
        "bounds": bounds,
        "constraints": {"type": "eq", "fun": constraints},
        "options": {"ftol": 1e-12, "maxiter": MAX_ITERATIONS, "disp": False},
        "callback": record_iterate,
    }
    result = minimize(**kwargs)
    wall = time.monotonic() - started
    parameters = np.asarray(result.x, dtype=float)
    base = parameters[:4]
    epsilon = float(parameters[4])
    residual = constraints(parameters)

    solution = ps.integrate_sigmoid_robust(
        case, base, epsilon, dense_output=True, rtol=sp.RTOL, atol=sp.ATOL
    )
    if solution.sol is None:
        raise ValueError("dense output was not created")
    samples = np.linspace(0.0, float(base[1]), 1601)
    joints = np.asarray(solution.sol(samples).T, dtype=float)
    switch = np.array([sp.switching_function(j, case.stage.kappa) for j in joints])
    throttle = expit(switch / epsilon)
    crossings = sp.switch_crossings(samples, switch, float(base[1]))

    sharp_state, sharp_switches, _events, sharp_joint, sharp_thrust, _arcs = (
        implicit_propagate(case, base)
    )
    return {
        "case": case.name,
        "success": bool(result.success),
        "message": str(result.message),
        "nfev": int(result.nfev),
        "nit": int(result.nit),
        "wall_seconds": wall,
        "parameters": parameters.tolist(),
        "epsilon": epsilon,
        "residual_norm": float(np.linalg.norm(residual)),
        "objective": float(result.fun),
        "q_min": float(np.min(throttle)),
        "q_max": float(np.max(throttle)),
        "crossings": [float(c) for c in crossings],
        "sigmoid_fuel": float(1.0 - joints[-1, 3]),
        "sharp_fuel": float(1.0 - sharp_joint[3]),
        "sharp_switch_times": [float(t) for t in sharp_switches],
        "sharp_thrust": float(sharp_thrust),
        "sharp_state": sharp_state.tolist(),
        "eps_history": [
            {"iteration": index, "epsilon": iterate[4], "tf": iterate[1]}
            for index, iterate in enumerate(progress)
        ],
        "reference": REFERENCES.get(case.name, {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=str, action="append", default=[])
    parser.add_argument("--output", type=str, default="")
    arguments = parser.parse_args()

    cases = sp.kerbin_cases()
    if arguments.case:
        by_name = {case.name: case for case in cases}
        cases = [by_name[name] for name in arguments.case if name in by_name]
        if not cases:
            raise SystemExit("no requested case matched a Kerbin case")

    records = []
    for case in cases:
        record = solve_free_eps_slsqp(case)
        records.append(record)
        reference = record["reference"]
        history = record["eps_history"]
        eps_path = (
            f"eps {history[0]['epsilon']:.3e} -> {history[-1]['epsilon']:.3e}"
            if history
            else "no callback history"
        )
        print(
            f"{record['case']:28s} ok={record['success']!s:5s} "
            f"eps={record['epsilon']:.3e} res={record['residual_norm']:.2e} "
            f"nit={record['nit']:4d} nfev={record['nfev']:5d} "
            f"sig_fuel={record['sigmoid_fuel']:.9f} "
            f"sharp_fuel={record['sharp_fuel']:.9f} "
            f"(oracle {reference.get('fuel', float('nan')):.9f}) "
            f"qmin={record['q_min']:.2e} qmax={record['q_max']:.2e} "
            f"crossings={[f'{c:.4f}' for c in record['crossings']]} "
            f"sharp_switches={[f'{t:.4f}' for t in record['sharp_switch_times']]} "
            f"{eps_path} [{record['wall_seconds']:.1f}s]"
        )
        print(f"    {record['message']}")
    if arguments.output:
        with open(arguments.output, "w", encoding="utf-8") as output_file:
            json.dump(records, output_file, indent=2)
            output_file.write("\n")


if __name__ == "__main__":
    main()
