#!/usr/bin/env python3
"""Sensitivity mapping around the fixed-sequence oracle at eps=1e-4.

Evidence for `review-sigmoid-primer.md` section 3.2. Prints the sigmoid-system
residual of the oracle parameters across eps, then solves from seeds perturbed
around the oracle on a (delta_lambda_eta, delta_tf) grid and in each single
parameter direction, classifying every result as distinct-burn or smeared.
Every seed in the explored neighbourhood converges to the identical smeared
root, so no seed distance from the oracle yields a distinct second burn at
eps=1e-4.

Run from the repository root with ``PYTHONPATH=experiments``::

    python3 experiments/sigmoid_primer_review/sensitivity_grid.py
"""

import math
import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sigmoid_primer as sp

Array = NDArray[np.float64]

CASE = sp.kerbin_cases()[0]
ORACLE = np.array(
    [1.357422910819774, 0.648432960860141, -1.120006399999580, -1.373896477245226]
)
EPS = 1e-4


def classify(p: Array) -> tuple[bool, float, float, float, float]:
    res = sp.sigmoid_residual(CASE, p, EPS)
    sol = sp.integrate_sigmoid(CASE, p, EPS, dense_output=True)
    if sol.sol is None:
        raise ValueError("dense output was not created")
    samples = np.linspace(0.0, p[1], 1601)
    joints = np.asarray(sol.sol(samples).T)
    switch = np.array([sp.switching_function(j, CASE.stage.kappa) for j in joints])
    throttle = expit(switch / EPS)
    crossings = sp.switch_crossings(samples, switch, p[1])
    interior = [c for c in crossings if abs(c - p[1]) > 1e-6]
    distinct = bool(
        len(interior) >= 2
        and abs(interior[-1] - 0.5958) < 0.05
        and p[1] < 0.68
        and throttle.min() < 0.02
    )
    return (
        distinct,
        float(np.linalg.norm(res)),
        float(p[1]),
        float(1.0 - joints[-1][3]),
        float(throttle.min()),
    )


def solve_from(seed: Array, max_nfev: int = 2000) -> tuple[Array, int, bool]:
    lower = np.array([-math.pi, 1e-5, -100.0, -100.0])
    upper = np.array([math.pi, CASE.first_arc_limit, 100.0, 100.0])

    def residual_function(parameters: Array) -> Array:
        return sp.sigmoid_residual(CASE, parameters, EPS)

    result = least_squares(
        residual_function,
        seed,
        bounds=(lower, upper),
        x_scale="jac",
        ftol=2e-10,
        xtol=2e-10,
        gtol=2e-10,
        max_nfev=max_nfev,
    )
    return np.asarray(result.x, dtype=float), int(result.nfev), bool(result.success)


def safe_report(seed: Array) -> str:
    try:
        p, nfev, ok = solve_from(seed)
    except (FloatingPointError, ValueError, ZeroDivisionError) as error:
        return f"SOLVE FAILED {error}"
    try:
        distinct, res, tf, fuel, qmin = classify(p)
    except (FloatingPointError, ValueError, ZeroDivisionError) as error:
        return (
            f"DIAGNOSTIC FAILED {error} (solve: nfev={nfev} ok={ok} p={np.round(p, 6)})"
        )
    return (
        f"{'DISTINCT' if distinct else 'smeared '} "
        f"tf={tf:.6f} fuel={fuel:.9f} qmin={qmin:.2e} res={res:.2e} nfev={nfev}"
    )


if __name__ == "__main__":
    print("sigmoid-system residual at ORACLE across eps:")
    for eps in (1e-2, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6):
        res = sp.sigmoid_residual(CASE, ORACLE, eps)
        print(
            f"  eps={eps:.1e} |res|={np.linalg.norm(res):.3e} comps={np.round(res, 6)}"
        )

    print()
    print("grid scan over (d_lambda_eta, d_tf):")
    for d_le in (-4e-3, -2e-3, -1e-3, -5e-4, 0.0, 5e-4, 1e-3):
        for d_tf in (-0.02, 0.0, 0.02, 0.04):
            seed = ORACLE + np.array([0.0, d_tf, 0.0, d_le])
            print(f"  d_le={d_le:+.0e} d_tf={d_tf:+.2f} -> {safe_report(seed)}")

    print()
    print("single-parameter perturbations at eps=1e-4:")
    for name, idx, deltas in (
        ("alpha0", 0, [-1e-4, -1e-3, -1e-2, 1e-2, 1e-3, 1e-4]),
        ("tf", 1, [-1e-4, -1e-3, -1e-2, 1e-2, 1e-3, 1e-4]),
        ("lam_rho0", 2, [-1e-4, -1e-3, -1e-2, 1e-2, 1e-3, 1e-4]),
        ("lam_eta0", 3, [-1e-4, -1e-3, -1e-2, 1e-2, 1e-3, 1e-4]),
    ):
        for d in deltas:
            seed = ORACLE.copy()
            seed[idx] += d
            print(f"  {name:10s} {d:+.0e} -> {safe_report(seed)}")
