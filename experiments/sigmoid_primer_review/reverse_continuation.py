#!/usr/bin/env python3
"""Reverse epsilon continuation for the `kerbin-example` sigmoid system.

Evidence for `review-sigmoid-primer.md` section 3.2: even tracking the
sharp-root family upward from eps=1e-6 (where the fixed-sequence oracle is a
near-root) collapses onto the same smeared root at eps=1e-4. This shows the
smeared root is the only accessible root of the sigmoid shooting system at
eps=1e-4, not an artifact of the downward continuation path.

Run from the repository root with ``PYTHONPATH=experiments``::

    python3 experiments/sigmoid_primer_review/reverse_continuation.py
"""

import math
import sys
from pathlib import Path
from typing import Any

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


def solve_from(
    seed: Array,
    epsilon: float,
    tag: str,
    max_nfev: int = 1000,
    **kwargs: Any,
) -> Array:
    lower = np.array([-math.pi, 1e-5, -100.0, -100.0])
    upper = np.array([math.pi, CASE.first_arc_limit, 100.0, 100.0])

    def residual_function(parameters: Array) -> Array:
        return sp.sigmoid_residual(CASE, parameters, epsilon)

    result = least_squares(
        residual_function,
        seed,
        bounds=(lower, upper),
        x_scale="jac",
        ftol=2e-10,
        xtol=2e-10,
        gtol=2e-10,
        max_nfev=max_nfev,
        **kwargs,
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
        f"qmin={throttle.min():.3e} crosses={[f'{c:.6f}' for c in crossings]}"
    )
    return p


if __name__ == "__main__":
    print("=== reverse continuation from oracle ===")
    seed = ORACLE.copy()
    for eps in (1e-6, 3e-6, 1e-5, 3e-5, 1e-4):
        seed = solve_from(seed, eps, "sharp-track", max_nfev=2000)
