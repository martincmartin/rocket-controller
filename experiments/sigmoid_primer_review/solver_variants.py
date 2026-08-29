#!/usr/bin/env python3
"""Solver-variant comparison for the sigmoid shooting system at eps=1e-4,
seeded from the fixed-sequence oracle parameters.

Evidence for `review-sigmoid-primer.md` section 3.3. Every variant lands on
the same smeared root; `lm` needs only 7 evaluations where `trf` needs 383,
`root` adds nothing over `least_squares`, `df-sane` fails, and SLSQP
minimizing final mass subject to the terminal constraints is 15-30x slower
with no benefit. The 438-evaluation count reported in the study is mostly
`trf` step-acceptance churn, not intrinsic difficulty.

Run from the repository root with ``PYTHONPATH=experiments``::

    python3 experiments/sigmoid_primer_review/solver_variants.py
"""

import math
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import NonlinearConstraint, least_squares, minimize, root
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sigmoid_primer as sp

Array = NDArray[np.float64]

CASE = sp.kerbin_cases()[0]
ORACLE = np.array(
    [1.357422910819774, 0.648432960860141, -1.120006399999580, -1.373896477245226]
)
EPS = 1e-4


def classify(p: Array) -> str:
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
    distinct = (
        len(interior) >= 2
        and abs(interior[-1] - 0.5958) < 0.05
        and p[1] < 0.68
        and throttle.min() < 0.02
    )
    return (
        f"tf={p[1]:.9f} fuel={1 - joints[-1][3]:.9f} qmin={throttle.min():.2e} "
        f"crosses={[f'{c:.5f}' for c in crossings]} "
        f"res={np.linalg.norm(res):.2e} "
        f"{'DISTINCT' if distinct else 'smeared/other'}"
    )


def run(tag: str, solver: Callable[..., Any], **kwargs: Any) -> Array | None:
    t0 = time.time()
    try:
        result = solver(**kwargs)
    except Exception as error:
        print(f"{tag:30s} EXCEPTION {error}")
        return None
    if hasattr(result, "x"):
        p: Array = np.asarray(result.x, dtype=float)
        print(
            f"{tag:30s} ok={getattr(result, 'success', '?')} "
            f"{str(getattr(result, 'message', ''))[:22]:22s} "
            f"nfev={getattr(result, 'nfev', '?'):>5} "
            f"{classify(p)}  [{time.time() - t0:.1f}s]"
        )
        return p
    print(f"{tag:30s} {result} [{time.time() - t0:.1f}s]")
    return None


def terminal_state(p: Array) -> Array:
    try:
        sol = sp.integrate_sigmoid(CASE, p, EPS)
        return np.asarray(sol.y[:, -1], dtype=float)
    except (FloatingPointError, ValueError, ZeroDivisionError):
        return np.full(8, np.nan)


def mass_objective(p: Array) -> float:
    return float(terminal_state(p)[3])


def terminal_constraints(p: Array) -> Array:
    state = terminal_state(p)
    return np.array([state[0] - 1.0, state[1], state[2] - 1.0])


if __name__ == "__main__":
    lower = np.array([-math.pi, 1e-5, -100.0, -100.0])
    upper = np.array([math.pi, CASE.first_arc_limit, 100.0, 100.0])

    print("=== trf baseline (bounds, x_scale=jac) ===")
    run(
        "trf-baseline",
        least_squares,
        fun=lambda p: sp.sigmoid_residual(CASE, p, EPS),
        x0=ORACLE,
        bounds=(lower, upper),
        x_scale="jac",
        ftol=2e-10,
        xtol=2e-10,
        gtol=2e-10,
        max_nfev=1000,
    )
    print("=== lm (no bounds) ===")
    run(
        "lm",
        least_squares,
        fun=lambda p: sp.sigmoid_residual(CASE, p, EPS),
        x0=ORACLE,
        method="lm",
        ftol=2e-10,
        xtol=2e-10,
        gtol=2e-10,
        max_nfev=1000,
    )
    print("=== trf diff_step=1e-6 ===")
    run(
        "trf-diff1e-6",
        least_squares,
        fun=lambda p: sp.sigmoid_residual(CASE, p, EPS),
        x0=ORACLE,
        bounds=(lower, upper),
        x_scale="jac",
        diff_step=1e-6,
        ftol=2e-10,
        xtol=2e-10,
        gtol=2e-10,
        max_nfev=1000,
    )
    print("=== trf diff_step=1e-7 ===")
    run(
        "trf-diff1e-7",
        least_squares,
        fun=lambda p: sp.sigmoid_residual(CASE, p, EPS),
        x0=ORACLE,
        bounds=(lower, upper),
        x_scale="jac",
        diff_step=1e-7,
        ftol=2e-10,
        xtol=2e-10,
        gtol=2e-10,
        max_nfev=1000,
    )
    print("=== trf 3-point ===")
    run(
        "trf-3point",
        least_squares,
        fun=lambda p: sp.sigmoid_residual(CASE, p, EPS),
        x0=ORACLE,
        bounds=(lower, upper),
        x_scale="jac",
        jac="3-point",
        ftol=2e-10,
        xtol=2e-10,
        gtol=2e-10,
        max_nfev=1000,
    )
    print("=== root hybr ===")
    run(
        "root-hybr",
        root,
        fun=lambda p: sp.sigmoid_residual(CASE, p, EPS),
        x0=ORACLE,
        method="hybr",
    )
    print("=== root lm ===")
    run(
        "root-lm",
        root,
        fun=lambda p: sp.sigmoid_residual(CASE, p, EPS),
        x0=ORACLE,
        method="lm",
    )
    print("=== root df-sane ===")
    run(
        "root-df-sane",
        root,
        fun=lambda p: sp.sigmoid_residual(CASE, p, EPS),
        x0=ORACLE,
        method="df-sane",
    )
    print("=== SLSQP: min final mass s.t. 3 terminal residuals == 0 ===")
    run(
        "slsqp-mass",
        minimize,
        fun=mass_objective,
        x0=ORACLE,
        method="SLSQP",
        bounds=list(zip(lower, upper, strict=True)),
        constraints=NonlinearConstraint(terminal_constraints, 0.0, 0.0),
        options={"ftol": 2e-12, "maxiter": 2000},
    )
