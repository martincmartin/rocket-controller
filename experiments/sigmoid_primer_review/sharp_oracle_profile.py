#!/usr/bin/env python3
"""Propagate the fixed-sequence `kerbin-example` oracle through the exact
hard-switch dynamics and print the switching-function profile per arc.

Evidence for `review-sigmoid-primer.md` section 2: the optimal trajectory is
nearly singular. S(0) = +2.27e-3, the coast dips only to -5.75e-5, and the
second burn peaks at only +1.56e-6, so eps=1e-4 cannot render distinct
burn/coast intervals for this case.

Run from the repository root with ``PYTHONPATH=experiments``::

    python3 experiments/sigmoid_primer_review/sharp_oracle_profile.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sigmoid_primer as sp
from single_stage_research import implicit_propagate, switch_function

CASE = sp.kerbin_cases()[0]
ORACLE = np.array(
    [1.357422910819774, 0.648432960860141, -1.120006399999580, -1.373896477245226]
)


if __name__ == "__main__":
    state, switch_times, events, final_joint, thrust_time, arcs = implicit_propagate(
        CASE, ORACLE
    )
    print("sharp switch times:", switch_times)
    print("sharp thrust time:", thrust_time, "fuel:", 1 - final_joint[3])
    print("sharp terminal state:", final_joint[:4])
    print(
        "final state residual:",
        np.array([final_joint[0] - 1, final_joint[1], final_joint[2] - 1]),
    )
    costate = np.array([ORACLE[2], np.cos(ORACLE[0]), np.sin(ORACLE[0]), ORACLE[3]])
    print(
        "sharp S(0):",
        switch_function(np.concatenate((CASE.x0, costate)), CASE.stage.kappa),
    )
    print("final S:", switch_function(final_joint, CASE.stage.kappa))
    print("arcs:")
    for arc in arcs:
        phi = np.array(
            [
                switch_function(arc.joint[:, i], CASE.stage.kappa)
                for i in range(arc.joint.shape[1])
            ]
        )
        print(
            f"  t=[{arc.start:.6f},{arc.end:.6f}] q={arc.throttle} "
            f"S=({phi.min():.3e},{phi.max():.3e})"
        )
