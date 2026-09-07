#!/usr/bin/env python3
"""Experiment: epsilon as a free shooting parameter instead of an outer
continuation loop.

The shooting vector becomes (alpha0, tf, lambda_rho0, lambda_mass0, eps):
five unknowns against the four residuals (three terminal conditions plus
S(tf)), so the system is underdetermined by one. There is no continuation
and no eps-specific residual; `least_squares` with method='trf' descends the
squared residual from (analytic guess, eps=1.0).

Questions this answers:

- Does it converge to a root?
- At what epsilon does it land, and does that epsilon render a bang-bang
  profile (distinct burn/coast intervals)?
- How close is the fuel to the fixed-sequence oracle?

The residual manifold r(z)=0 is one-dimensional in the five unknowns (one
root per epsilon, at least over the epsilon range tested), so where the
solver lands along the manifold is determined by the descent path, not by
the equations.

Run from the repository root with ``PYTHONPATH=experiments``::

    python3 experiments/sigmoid_primer_review/free_eps_shooting.py

Use ``--case NAME`` to run a single case and ``--output FILE`` to write the
JSON record.
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
from scipy.optimize import least_squares
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sigmoid_primer as sp
from single_stage_research import Case, implicit_propagate

Array = NDArray[np.float64]

EPS_LOWER = 1e-8
EPS_UPPER = 10.0
MAX_NFEV = 3000
REFERENCES = {
    "kerbin-example": {"fuel": 0.458673471, "tf": 0.648432961},
    "kerbin-first-example": {"fuel": 0.417870055, "tf": 0.522074908},
    "kerbin-coast-first-example": {"fuel": 0.339814106, "tf": 0.510560727},
}


def free_eps_residual(parameters: Array, case: Case) -> Array:
    alpha, final_time, lambda_rho, lambda_mass, epsilon = parameters
    if epsilon <= 0.0 or not np.isfinite(epsilon):
        return np.full(4, 1e3)
    base = np.array([alpha, final_time, lambda_rho, lambda_mass], dtype=float)
    return sp.sigmoid_residual(case, base, float(epsilon))


def solve_free_eps(case: Case) -> dict[str, Any]:
    guess = sp.initial_sigmoid_guess(case)
    initial = np.concatenate((guess, [1.0]))
    lower = np.array([-math.pi, 1e-5, -100.0, -100.0, EPS_LOWER], dtype=float)
    upper = np.array(
        [math.pi, case.first_arc_limit, 100.0, 100.0, EPS_UPPER], dtype=float
    )
    started = time.monotonic()
    result = least_squares(
        free_eps_residual,
        initial,
        args=(case,),
        bounds=(lower, upper),
        x_scale="jac",
        ftol=2e-12,
        xtol=2e-12,
        gtol=2e-12,
        max_nfev=MAX_NFEV,
    )
    wall = time.monotonic() - started
    parameters = np.asarray(result.x, dtype=float)
    base = parameters[:4]
    epsilon = float(parameters[4])
    residual = free_eps_residual(parameters, case)

    solution = sp.integrate_sigmoid(case, base, epsilon, dense_output=True)
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
        "message": result.message,
        "nfev": int(result.nfev),
        "wall_seconds": wall,
        "parameters": parameters.tolist(),
        "epsilon": epsilon,
        "residual_norm": float(np.linalg.norm(residual)),
        "q_min": float(np.min(throttle)),
        "q_max": float(np.max(throttle)),
        "crossings": [float(c) for c in crossings],
        "sigmoid_fuel": float(1.0 - joints[-1, 3]),
        "sharp_fuel": float(1.0 - sharp_joint[3]),
        "sharp_switch_times": [float(t) for t in sharp_switches],
        "sharp_thrust": float(sharp_thrust),
        "sharp_state": sharp_state.tolist(),
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
        record = solve_free_eps(case)
        records.append(record)
        reference = record["reference"]
        print(
            f"{record['case']:28s} ok={record['success']!s:5s} "
            f"eps={record['epsilon']:.3e} res={record['residual_norm']:.2e} "
            f"nfev={record['nfev']:4d} sig_fuel={record['sigmoid_fuel']:.9f} "
            f"sharp_fuel={record['sharp_fuel']:.9f} "
            f"(oracle {reference.get('fuel', float('nan')):.9f}) "
            f"qmin={record['q_min']:.2e} qmax={record['q_max']:.2e} "
            f"crossings={[f'{c:.4f}' for c in record['crossings']]} "
            f"sharp_switches={[f'{t:.4f}' for t in record['sharp_switch_times']]} "
            f"[{record['wall_seconds']:.1f}s]"
        )
        print(f"    {record['message']}")
    if arguments.output:
        with open(arguments.output, "w", encoding="utf-8") as output_file:
            json.dump(records, output_file, indent=2)
            output_file.write("\n")


if __name__ == "__main__":
    main()
