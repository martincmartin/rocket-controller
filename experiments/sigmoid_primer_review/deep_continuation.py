#!/usr/bin/env python3
"""Epsilon continuation below the study's 1e-4 floor, with a large evaluation
budget.

Evidence for `review-sigmoid-primer.md` sections 3.3 and 3.4: below 1e-4 the
finite-difference solver drifts back toward the sharp solution (fuel returns
to the oracle value) but stalls at residual ~1e-3..1e-5 without ever
converging tightly. This motivates the adaptive-epsilon ladder and the sharp
event-based polish in `production_solver.py`.

Run from the repository root with ``PYTHONPATH=experiments``::

    python3 experiments/sigmoid_primer_review/deep_continuation.py
"""

import math
import sys
from pathlib import Path
from typing import Any, Literal

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
SMEARED = np.array(
    [1.3557489599336592, 0.7149047496714878, -1.1228459941245825, -1.3730707279997247]
)


def solve_from(
    seed: Array,
    epsilon: float,
    tag: str,
    max_nfev: int = 4000,
    diff_step: float | None = None,
    jac: Literal["2-point", "3-point", "cs"] = "2-point",
) -> Array:
    lower = np.array([-math.pi, 1e-5, -100.0, -100.0])
    upper = np.array([math.pi, CASE.first_arc_limit, 100.0, 100.0])
    kw: dict[str, Any] = {"diff_step": diff_step} if diff_step is not None else {}

    def residual_function(parameters: Array) -> Array:
        return sp.sigmoid_residual(CASE, parameters, epsilon)

    result = least_squares(
        residual_function,
        seed,
        bounds=(lower, upper),
        x_scale="jac",
        jac=jac,
        ftol=2e-12,
        xtol=2e-12,
        gtol=2e-12,
        max_nfev=max_nfev,
        **kw,
    )
    p: Array = np.asarray(result.x, dtype=float)
    res = sp.sigmoid_residual(CASE, p, epsilon)
    sol = sp.integrate_sigmoid(CASE, p, epsilon, dense_output=True)
    if sol.sol is None:
        raise ValueError("dense output was not created")
    samples = np.linspace(0.0, p[1], 1601)
    joints = np.asarray(sol.sol(samples).T)
    switch = np.array([sp.switching_function(j, CASE.stage.kappa) for j in joints])
    throttle = expit(switch / epsilon)
    crossings = sp.switch_crossings(samples, switch, p[1])
    print(
        f"{tag:22s} eps={epsilon:.1e} ok={result.success} {result.message[:30]:30s} "
        f"nfev={result.nfev:4d} res={np.linalg.norm(res):.3e} "
        f"tf={p[1]:.9f} fuel={1 - joints[-1][3]:.9f} "
        f"qmin={throttle.min():.3e} crosses={[f'{c:.5f}' for c in crossings]}"
    )
    return p


if __name__ == "__main__":
    print("=== deep continuation from smeared ===")
    seed = SMEARED.copy()
    for eps in (3e-5, 1e-5, 3e-6, 1e-6, 3e-7):
        seed = solve_from(seed, eps, "deep-cont", max_nfev=4000)
    print("=== deep continuation from oracle ===")
    seed = ORACLE.copy()
    for eps in (3e-6, 1e-6, 3e-7):
        seed = solve_from(seed, eps, "deep-oracle", max_nfev=4000)
