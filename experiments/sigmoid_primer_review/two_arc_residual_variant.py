#!/usr/bin/env python3
"""Experiment: replace the fixed-sequence two-arc solver's H(0)=0 residual
with S(tf)=0 (the switching function at the end of the second burn).

The fixed on/off/on solver in `two_arc_shooting.py` shoots six unknowns
(alpha0, burn1, coast, burn2, lambda_rho0, lambda_mass0) against six
residuals: three terminal conditions, S=0 at the two interior switch
boundaries, and either H(0)=0 (free final time) or the final-time-cap
residual (active cap). This script re-solves the three Kerbin cases with
residual 6 switched from H(0)=0 to S(tf)=0 and compares fuel, burn/coast
partition, final time, residual norm, and evaluation count.

Because the fixed sequence ends with a burn, q(tf)=1 and at the target
H(tf) = -gamma*S(tf), so H(0)=0 and S(tf)=0 are equivalent by Hamiltonian
conservation -- the question is whether they behave identically in the
finite-difference shooting loop.

Run from the repository root with ``PYTHONPATH=experiments``::

    python3 experiments/sigmoid_primer_review/two_arc_residual_variant.py
"""

import argparse
import json
import math
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sigmoid_primer as sp
import two_arc_shooting
from single_stage_research import (
    Case,
    direct_solve,
    integrate_primer,
    norm,
    switch_function,
)
from two_burn_reference import solve_two_burn_multistart

Array = NDArray[np.float64]

REFERENCES = {
    "kerbin-example": {
        "fuel": 0.458673471,
        "tf": 0.648432961,
        "b1": 0.258909706,
        "coast": 0.336885349,
        "b2": 0.052637906,
    },
    "kerbin-first-example": {"fuel": 0.417870055, "tf": 0.522074908},
    "kerbin-coast-first-example": {"fuel": 0.339814106, "tf": 0.510560727},
}


def residual_switch_tf(
    case: Case, parameters: Array, final_time_active: bool = False
) -> Array:
    """The two-arc residual with S(tf)=0 in place of H(0)=0."""
    alpha, burn_one, coast_gap, burn_two, lambda_rho, lambda_mass = parameters
    initial_costate = np.array(
        [lambda_rho, math.cos(alpha), math.sin(alpha), lambda_mass], dtype=float
    )
    first = integrate_primer(case.x0, initial_costate, burn_one, case.stage, 1.0)
    first_switch = switch_function(first, case.stage.kappa)
    coast = integrate_primer(first[:4], first[4:], coast_gap, case.stage, 0.0)
    second_switch = switch_function(coast, case.stage.kappa)
    final = integrate_primer(coast[:4], coast[4:], burn_two, case.stage, 1.0)
    final_switch = switch_function(final, case.stage.kappa)
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
        residual_values.append(final_switch)
    return np.asarray(residual_values, dtype=float)


def solve_with(
    case: Case,
    initial: Array,
    final_time_active: bool,
    residual_function: Callable[[Case, np.ndarray, bool], np.ndarray],
) -> dict[str, Any]:
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

    def wrapped(parameters: np.ndarray) -> np.ndarray:
        return residual_function(case, parameters, final_time_active)

    result = least_squares(
        wrapped,
        initial,
        bounds=(lower, upper),
        x_scale="jac",
        ftol=2e-12,
        xtol=2e-12,
        gtol=2e-12,
        max_nfev=800,
    )
    parameters = np.asarray(result.x, dtype=float)
    residual_value = residual_function(case, parameters, final_time_active)
    total_time = float(np.sum(parameters[1:4]))
    fuel = case.stage.gamma * float(parameters[1] + parameters[3]) / case.stage.kappa
    return {
        "success": bool(
            result.success
            and norm(residual_value) < 2e-6
            and total_time <= case.first_arc_limit + 1e-8
        ),
        "message": str(result.message),
        "nfev": int(result.nfev),
        "parameters": parameters.tolist(),
        "residual_norm": float(norm(residual_value)),
        "final_time": total_time,
        "fuel": fuel,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="")
    arguments = parser.parse_args()

    records: list[dict[str, Any]] = []
    for case in sp.kerbin_cases():
        direct = None
        for mesh in (4, 8, 16):
            direct = direct_solve(case, mesh, direct)
        if direct is None or not direct.success:
            raise ValueError(f"restricted direct seed failed for {case.name}")
        direct_two_burn = solve_two_burn_multistart(case, direct, 8)
        initial = two_arc_shooting.seed(case, direct_two_burn)
        direct_parameters = direct_two_burn.parameters
        final_time_active = bool(
            float(np.sum(direct_parameters[:4])) >= case.first_arc_limit - 1e-6
        )
        h_result = solve_with(
            case, initial, final_time_active, two_arc_shooting.residual
        )
        stf_result = solve_with(case, initial, final_time_active, residual_switch_tf)
        reference = REFERENCES[case.name]
        print(f"--- {case.name} (final_time_active={final_time_active}) ---")
        for label, result in (("H(0)=0", h_result), ("S(tf)=0", stf_result)):
            print(
                f"  {label}: ok={result['success']!s:5s} "
                f"fuel={result['fuel']:.9f} res={result['residual_norm']:.2e} "
                f"tf={result['final_time']:.9f} nfev={result['nfev']:4d} "
                f"params={np.round(result['parameters'], 9).tolist()}"
            )
        print(f"  oracle: fuel={reference['fuel']:.9f} tf={reference['tf']:.9f}")
        delta = {
            "fuel": h_result["fuel"] - stf_result["fuel"],
            "tf": h_result["final_time"] - stf_result["final_time"],
        }
        print(f"  H-minus-Stf: fuel={delta['fuel']:.3e} tf={delta['tf']:.3e}")
        records.append(
            {
                "case": case.name,
                "final_time_active": final_time_active,
                "H": h_result,
                "S_tf": stf_result,
                "reference": reference,
            }
        )
    if arguments.output:
        with open(arguments.output, "w", encoding="utf-8") as output_file:
            json.dump(records, output_file, indent=2)
            output_file.write("\n")


if __name__ == "__main__":
    main()
