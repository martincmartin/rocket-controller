#!/usr/bin/env python3
"""Two tests: (a) tighter eps=3e-4 warm start before the 1e-4 step;
(b) lm on a deep epsilon schedule from the oracle seed.

Evidence for `review-sigmoid-primer.md` sections 3.3 and 3.5: converging the
earlier epsilon more tightly changes nothing at eps=1e-4 (438 evaluations
either way), while `lm` reaches tf=0.64848 (the oracle is 0.64843) in 28
evaluations at eps=1e-6, far cheaper than `trf` on the same schedule.

Run from the repository root with ``PYTHONPATH=experiments``::

    python3 experiments/sigmoid_primer_review/warm_start_lm.py
"""

import math
import sys
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sigmoid_primer as sp

Array = NDArray[np.float64]

CASE = sp.kerbin_cases()[0]
ORACLE = np.array(
    [1.357422910819774, 0.648432960860141, -1.120006399999580, -1.373896477245226]
)


def solve(
    seed: Array,
    epsilon: float,
    method: Literal["trf", "lm"] = "trf",
    max_nfev: int = 1000,
    **kw: Any,
) -> tuple[Array, Any]:
    lower = np.array([-math.pi, 1e-5, -100.0, -100.0])
    upper = np.array([math.pi, CASE.first_arc_limit, 100.0, 100.0])
    kwargs: dict[str, Any] = dict(
        fun=lambda p: sp.sigmoid_residual(CASE, p, epsilon),
        x0=seed,
        method=method,
        x_scale="jac",
        ftol=kw.pop("ftol", 2e-10),
        xtol=kw.pop("xtol", 2e-10),
        gtol=kw.pop("gtol", 2e-10),
        max_nfev=max_nfev,
        **kw,
    )
    if method != "lm":
        kwargs["bounds"] = (lower, upper)
    result = least_squares(**kwargs)
    return np.asarray(result.x), result


def show(tag: str, p: Array, result: Any, epsilon: float) -> None:
    res = sp.sigmoid_residual(CASE, p, epsilon)
    print(
        f"{tag:26s} eps={epsilon:.0e} ok={result.success} nfev={result.nfev:4d} "
        f"res={np.linalg.norm(res):.2e} tf={p[1]:.9f}"
    )


def lm_step(seed: Array, eps: float) -> Array | None:
    try:
        result_seed, result = solve(seed, eps, method="lm", max_nfev=2000)
        show("lm-oracle", result_seed, result, eps)
        return result_seed
    except (ValueError, FloatingPointError) as error:
        print(f"lm-oracle eps={eps:.0e} FAILED {error}")
        return None


if __name__ == "__main__":
    # (a) replicate baseline continuation path for kerbin-example down to
    #     3e-4, then two variants of the 1e-4 step
    print("=== (a) warm-start quality at eps=1e-4 ===")
    seed = sp.initial_sigmoid_guess(CASE)
    for eps in (1.0, 3e-1, 1e-1, 3e-2, 1e-2, 3e-3, 1e-3):
        seed, result = solve(seed, eps)
    seed, result = solve(seed, 3e-4)
    p_loose = seed
    show("baseline path @3e-4", seed, result, 3e-4)
    p1, r1 = solve(p_loose, 1e-4)
    show("1e-4 from loose 3e-4", p1, r1, 1e-4)

    seed, result = solve(
        p_loose, 3e-4, ftol=1e-13, xtol=1e-13, gtol=1e-13, max_nfev=2000
    )
    p_tight = seed
    show("tight path @3e-4", seed, result, 3e-4)
    p2, r2 = solve(p_tight, 1e-4)
    show("1e-4 from tight 3e-4", p2, r2, 1e-4)

    print("=== (b) LM on deep schedule from oracle ===")
    seed = ORACLE.copy()
    for eps in (1e-4, 3e-5, 1e-5, 3e-6, 1e-6):
        result_seed = lm_step(seed, eps)
        if result_seed is None:
            break
        seed = result_seed
