#!/usr/bin/env python3
"""Multistart basin search at eps=1e-4 and finite-difference step sweep.

Evidence for `review-sigmoid-primer.md` sections 3.2 and 3.3: 48 random seeds
in a broad box around the oracle all converge to the identical smeared root
(or fail inside the integrator), so the smeared root is the only accessible
root of the sigmoid shooting system at eps=1e-4. The diff_step sweep shows
that larger finite-difference steps recover the sharp structure at small eps
but never converge tightly, while the default (tiny) step lands on the
smeared root.

Run from the repository root with ``PYTHONPATH=experiments``::

    python3 experiments/sigmoid_primer_review/multistart_basin.py
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
EPS = 1e-4


def classify(p: Array, epsilon: float = EPS) -> str:
    res = sp.sigmoid_residual(CASE, p, epsilon)
    sol = sp.integrate_sigmoid(CASE, p, epsilon, dense_output=True)
    if sol.sol is None:
        raise ValueError("dense output was not created")
    samples = np.linspace(0.0, p[1], 1601)
    joints = np.asarray(sol.sol(samples).T)
    switch = np.array([sp.switching_function(j, CASE.stage.kappa) for j in joints])
    throttle = expit(switch / epsilon)
    crossings = sp.switch_crossings(samples, switch, p[1])
    interior = [c for c in crossings if abs(c - p[1]) > 1e-6]
    distinct = (
        len(interior) >= 2
        and abs(interior[-1] - 0.5958) < 0.05
        and p[1] < 0.68
        and throttle.min() < 0.02
    )
    return (
        f"tf={p[1]:.6f} fuel={1 - joints[-1][3]:.9f} qmin={throttle.min():.2e} "
        f"crosses={len(crossings)} res={np.linalg.norm(res):.2e} "
        f"{'DISTINCT' if distinct else 'smeared'}"
    )


def solve(
    seed: Array, epsilon: float, max_nfev: int = 800, **kw: Any
) -> tuple[Array, int]:
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
        **kw,
    )
    parameters: Array = np.asarray(result.x, dtype=float)
    return parameters, int(result.nfev)


if __name__ == "__main__":
    print("=== multistart at eps=1e-4, box around oracle ===")
    rng = np.random.default_rng(20260826)
    lo = np.array([1.15, 0.55, -1.6, -1.65])
    hi = np.array([1.55, 0.95, -0.7, -1.15])
    for _ in range(48):
        seed = rng.uniform(lo, hi)
        try:
            p, nfev = solve(seed, EPS)
            print(f"  seed={np.round(seed, 4)} -> {classify(p)} nfev={nfev}")
        except (ValueError, FloatingPointError) as error:
            print(f"  seed={np.round(seed, 4)} -> FAILED {error}")

    print()
    print("=== diff_step sweep at eps=1e-4 and 1e-5 from ORACLE ===")
    for eps in (1e-4, 3e-5, 1e-5):
        for ds in (None, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8):
            kw = {} if ds is None else {"diff_step": ds}
            try:
                p, nfev = solve(ORACLE, eps, max_nfev=1500, **kw)
                print(
                    f"  eps={eps:.0e} diff_step={ds!s:5s} -> {classify(p, eps)} "
                    f"nfev={nfev}"
                )
            except (ValueError, FloatingPointError) as error:
                print(f"  eps={eps:.0e} diff_step={ds!s:5s} -> FAILED {error}")
