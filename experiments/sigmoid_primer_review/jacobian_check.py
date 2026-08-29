#!/usr/bin/env python3
"""Verify the analytic Jacobian prototype against central finite differences.

Companion to `analytic_jacobian.py` (review section 3.3). Prints the relative
error between the variational-sensitivities Jacobian and a central-difference
Jacobian of the same residual at the oracle and smeared parameters, for
eps=1e-4 and eps=1e-5. The relative errors are ~1e-4..2e-2 and are dominated
by the finite-difference side (the ODE tolerance is 2e-10), which is exactly
the noise the analytic Jacobian exists to avoid.

Run from the repository root with ``PYTHONPATH=experiments``::

    python3 experiments/sigmoid_primer_review/jacobian_check.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sigmoid_primer as sp
from analytic_jacobian import ORACLE, SMEARED, residual_with_jac

CASE = sp.kerbin_cases()[0]
POINTS = {"oracle": ORACLE, "smeared": SMEARED}


if __name__ == "__main__":
    for tag, point in POINTS.items():
        for eps in (1e-4, 1e-5):
            _residual, jacobian = residual_with_jac(CASE, point, eps)
            finite_difference = np.zeros_like(jacobian)
            step = 1e-6
            for column in range(4):
                plus = point.copy()
                plus[column] += step
                rp, _ = residual_with_jac(CASE, plus, eps)
                minus = point.copy()
                minus[column] -= step
                rm, _ = residual_with_jac(CASE, minus, eps)
                finite_difference[:, column] = (rp - rm) / (2.0 * step)
            error = np.linalg.norm(finite_difference - jacobian) / max(
                np.linalg.norm(finite_difference), 1e-30
            )
            print(f"{tag} eps={eps:.0e} jac rel err vs FD(1e-6) = {error:.3e}")
            print("  jac=\n", np.round(jacobian, 3))
            print("  fd =\n", np.round(finite_difference, 3))
