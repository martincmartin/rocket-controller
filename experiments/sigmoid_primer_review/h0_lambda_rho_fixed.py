#!/usr/bin/env python3
"""Experiment: remove lambda_rho0 as a shooting variable and fix it from
H(0)=0, dropping the S(tf) residual (3 unknowns, 3 terminal residuals).

Evidence for `review-sigmoid-primer.md` section 3.1. The substituted
Hamiltonian is not conserved at finite epsilon, so H(0)=0 does not imply
H(tf)=0 and the free final time is unconstrained: the 3x3 system has a
1-parameter family of roots and the final time runs away along the epsilon
ladder. The S(tf) residual of the original four-unknown formulation is
essential.

Run from the repository root with ``PYTHONPATH=experiments``::

    python3 experiments/sigmoid_primer_review/h0_lambda_rho_fixed.py
"""

import math
import sys
from pathlib import Path
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sigmoid_primer as sp
from single_stage_research import Case

Array = NDArray[np.float64]


class Record(TypedDict):
    ok: bool
    solver_ok: bool
    message: str
    nfev: int
    res: float
    tf: float
    fuel: float
    qmin: float
    qmax: float
    S0: float
    Sf: float
    H0: float
    Hf: float
    Hdrift: float
    crossings: list[float]
    params: list[float]


CASES = sp.kerbin_cases()
EPS_SCHEDULE = sp.EPS_SCHEDULE


def lambda_rho_fixed(case: Case, parameters: Array, epsilon: float) -> float:
    alpha = float(parameters[0])
    lambda_mass = float(parameters[2])
    rho, ur, ut, mass = (float(value) for value in case.x0)
    p_r, p_t = math.cos(alpha), math.sin(alpha)
    primer_length = math.hypot(p_r, p_t)
    switch0 = primer_length / mass + lambda_mass / case.stage.kappa
    q0 = float(expit(switch0 / epsilon))
    gravity_radial = ut * ut / rho - 1.0 / rho**2
    return float(
        (p_r * gravity_radial - p_t * ur * ut / rho + case.stage.gamma * q0 * switch0)
        / ur
    )


def three_residual(case: Case, parameters: Array, epsilon: float) -> Array:
    try:
        full = np.array(
            [
                parameters[0],
                parameters[1],
                lambda_rho_fixed(case, parameters, epsilon),
                parameters[2],
            ]
        )
        solution = sp.integrate_sigmoid(case, full, epsilon)
        final = np.asarray(solution.y[:, -1], dtype=float)
        return sp.terminal_residual(final[:4])
    except (FloatingPointError, ValueError, ZeroDivisionError):
        return np.full(3, 1e3)


def solve_at_eps(case: Case, parameters: Array, epsilon: float) -> tuple[Record, Array]:
    lower = np.array([-math.pi, 1e-5, -100.0])
    upper = np.array([math.pi, case.first_arc_limit, 100.0])

    def residual_function(p: Array) -> Array:
        return three_residual(case, p, epsilon)

    result = least_squares(
        residual_function,
        parameters,
        bounds=(lower, upper),
        x_scale="jac",
        ftol=2e-10,
        xtol=2e-10,
        gtol=2e-10,
        max_nfev=1000,
    )
    p = np.asarray(result.x, dtype=float)
    res = three_residual(case, p, epsilon)
    full: Array = np.array(
        [p[0], p[1], lambda_rho_fixed(case, p, epsilon), p[2]], dtype=float
    )
    record: Record = {
        "ok": bool(result.success and np.linalg.norm(res) < 2e-6),
        "solver_ok": bool(result.success),
        "message": result.message,
        "nfev": int(result.nfev),
        "res": float(np.linalg.norm(res)),
        "tf": float(full[1]),
        "fuel": float("nan"),
        "qmin": float("nan"),
        "qmax": float("nan"),
        "S0": float("nan"),
        "Sf": float("nan"),
        "H0": float("nan"),
        "Hf": float("nan"),
        "Hdrift": float("nan"),
        "crossings": [],
        "params": full.tolist(),
    }
    try:
        sol = sp.integrate_sigmoid(case, full, epsilon, dense_output=True)
        if sol.sol is None:
            raise ValueError("dense output was not created")
        samples = np.linspace(0.0, full[1], 1601)
        joints = np.asarray(sol.sol(samples).T)
        switch = np.array([sp.switching_function(j, case.stage.kappa) for j in joints])
        throttle = expit(switch / epsilon)
        hams = np.array(
            [
                sp.primer_hamiltonian(j, case.stage.gamma, case.stage.kappa, epsilon)
                for j in joints
            ]
        )
        record["fuel"] = float(1.0 - joints[-1][3])
        record["qmin"] = float(throttle.min())
        record["qmax"] = float(throttle.max())
        record["S0"] = float(switch[0])
        record["Sf"] = float(switch[-1])
        record["H0"] = float(hams[0])
        record["Hf"] = float(hams[-1])
        record["Hdrift"] = float(hams[-1] - hams[0])
        record["crossings"] = [
            float(c) for c in sp.switch_crossings(samples, switch, full[1])
        ]
    except (FloatingPointError, ValueError, ZeroDivisionError):
        record["message"] += " (diagnostic integration failed)"
    return record, full


def run_case(case: Case) -> list[Record]:
    guess = sp.initial_sigmoid_guess(case)
    parameters = np.array([guess[0], guess[1], guess[3]])
    print(f"--- {case.name} (3-unknown H(0)-fixed variant) ---")
    records: list[Record] = []
    for eps in EPS_SCHEDULE:
        record, full = solve_at_eps(case, parameters, eps)
        records.append(record)
        print(
            f"  eps={eps:7.0e} ok={record['ok']!s:5s} nfev={record['nfev']:4d} "
            f"res={record['res']:.2e} tf={record['tf']:.9f} fuel={record['fuel']:.9f} "
            f"qmin={record['qmin']:.2e} qmax={record['qmax']:.2e} "
            f"S0={record['S0']:+.2e} Sf={record['Sf']:+.2e} "
            f"Hdrift={record['Hdrift']:.2e} "
            f"crossings={[f'{c:.4f}' for c in record['crossings']]}"
        )
        parameters = np.array([full[0], full[1], full[3]])
    return records


def run_case_safe(case: Case) -> None:
    try:
        run_case(case)
    except (FloatingPointError, ValueError, ZeroDivisionError) as error:
        print(f"--- {case.name}: FAILED {error}")


if __name__ == "__main__":
    for case in CASES:
        run_case_safe(case)
