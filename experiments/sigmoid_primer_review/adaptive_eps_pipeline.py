#!/usr/bin/env python3
"""Adaptive-eps sigmoid continuation followed by a sharp event-based polish,
for all three Kerbin cases.

Prototype behind `review-sigmoid-primer.md` sections 3.4 and 3.6. Epsilon is
divided by 3 after each solve until the rendered control satisfies
q_min < 0.01 and q_max > 0.99, then the hard-switch shooter is seeded with
the sigmoid parameters. The production version of this pipeline is
`production_solver.py` in this directory.

Run from the repository root with ``PYTHONPATH=experiments``::

    python3 experiments/sigmoid_primer_review/adaptive_eps_pipeline.py
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
from single_stage_research import Case, implicit_propagate, implicit_residual

Array = NDArray[np.float64]

CASES = sp.kerbin_cases()


def sigmoid_solve(
    case: Case, seed: Array, epsilon: float, max_nfev: int = 800
) -> tuple[Array, Any]:
    lower = np.array([-math.pi, 1e-5, -100.0, -100.0])
    upper = np.array([math.pi, case.first_arc_limit, 100.0, 100.0])

    def residual_function(parameters: Array) -> Array:
        return sp.sigmoid_residual(case, parameters, epsilon)

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
    parameters: Array = np.asarray(result.x, dtype=float)
    return parameters, result


def throttle_diagnostics(
    case: Case, p: Array, epsilon: float
) -> tuple[float, float, tuple[float, ...], float]:
    sol = sp.integrate_sigmoid(case, p, epsilon, dense_output=True)
    if sol.sol is None:
        raise ValueError("dense output was not created")
    samples = np.linspace(0.0, p[1], 1601)
    joints = np.asarray(sol.sol(samples).T)
    switch = np.array([sp.switching_function(j, case.stage.kappa) for j in joints])
    throttle = expit(switch / epsilon)
    crossings = sp.switch_crossings(samples, switch, p[1])
    return (
        float(throttle.min()),
        float(throttle.max()),
        crossings,
        float(1.0 - joints[-1][3]),
    )


def adaptive_continuation(
    case: Case, eps_min: float = 3e-7
) -> tuple[Array, float, int, list[tuple[float, int, float, float, float, float]], str]:
    seed = sp.initial_sigmoid_guess(case)
    eps = 1.0
    total_nfev = 0
    history: list[tuple[float, int, float, float, float, float]] = []
    while eps >= eps_min:
        p, result = sigmoid_solve(case, seed, eps)
        total_nfev += result.nfev
        qmin, qmax, _crossings, fuel = throttle_diagnostics(case, p, eps)
        history.append((eps, int(result.nfev), qmin, qmax, float(p[1]), fuel))
        seed = p
        # render-quality criterion: coasts off and burns on
        if qmin < 0.01 and qmax > 0.99:
            return p, eps, total_nfev, history, "render-ok"
        eps /= 3.0
    return seed, eps, total_nfev, history, "eps-floor"


def sharp_polish(
    case: Case, seed: Array, max_nfev: int = 6000
) -> tuple[Array, Any, float, float, list[float], float]:
    lower = np.array([-math.pi, 1e-5, -100.0, -100.0])
    upper = np.array([math.pi, case.first_arc_limit, 100.0, 100.0])

    def residual_function(parameters: Array) -> Array:
        return implicit_residual(case, parameters)

    result = least_squares(
        residual_function,
        seed,
        bounds=(lower, upper),
        x_scale="jac",
        ftol=2e-12,
        xtol=2e-12,
        gtol=2e-12,
        max_nfev=max_nfev,
    )
    z = np.asarray(result.x)
    _state, sw, _ev, fj, tt, _arcs = implicit_propagate(case, z)
    res = implicit_residual(case, z)
    return z, result, float(np.linalg.norm(res)), float(1.0 - fj[3]), sw, tt


if __name__ == "__main__":
    for case in CASES:
        p, eps, nfev, history, status = adaptive_continuation(case)
        print(
            f"--- {case.name}: adaptive stop eps={eps:.2e} status={status} nfev={nfev}"
        )
        for eps_h, n, qmin, qmax, tf, fuel in history:
            print(
                f"    eps={eps_h:7.0e} nfev={n:4d} qmin={qmin:.2e} qmax={qmax:.2e} "
                f"tf={tf:.6f} fuel={fuel:.9f}"
            )
        z, result, res, fuel, sw, tt = sharp_polish(case, p)
        print(
            f"    SHARP: ok={result.success} nfev={result.nfev} res={res:.2e} "
            f"fuel={fuel:.9f} tf={z[1]:.9f} switches={[f'{s:.5f}' for s in sw]} "
            f"thrust={tt:.9f}"
        )
        if case.name == "kerbin-example":
            print(
                "    oracle: fuel=0.458673471 tf=0.648432961 "
                "switches=[0.25891, 0.59580]"
            )
        if case.name == "kerbin-first-example":
            print("    oracle: fuel=0.417870055 tf=0.522074908")
        if case.name == "kerbin-coast-first-example":
            print("    direct: fuel=0.339814106 tf=0.510560727")
