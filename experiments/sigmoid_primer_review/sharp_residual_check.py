#!/usr/bin/env python3
"""Evaluate the hard-switch (sharp) shooting system at the sigmoid smeared
root parameters and at the fixed-sequence oracle parameters.

Evidence for `review-sigmoid-primer.md` section 2: the smeared root found by
the eps=1e-4 sigmoid continuation does not satisfy the sharp system (its hard
switch propagation misses the target), so it is a sigmoid-only root, not a
limit of the sharp problem.

Run from the repository root with ``PYTHONPATH=experiments``::

    python3 experiments/sigmoid_primer_review/sharp_residual_check.py
"""

import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sigmoid_primer as sp
from single_stage_research import implicit_propagate, switch_function

Array = NDArray[np.float64]

CASE = sp.kerbin_cases()[0]
ORACLE = np.array(
    [1.357422910819774, 0.648432960860141, -1.120006399999580, -1.373896477245226]
)
SMEARED = np.array(
    [1.3557489599336592, 0.7149047496714878, -1.1228459941245825, -1.3730707279997247]
)


def report(tag: str, parameters: Array) -> None:
    print(f"=== sharp hard-switch system residual at {tag} ===")
    _state, switches, _events, final_joint, thrust_time, arcs = implicit_propagate(
        CASE, parameters
    )
    print("switches:", switches)
    print("thrust time:", thrust_time, "fuel:", 1 - final_joint[3])
    print(
        "terminal residual:",
        np.array([final_joint[0] - 1, final_joint[1], final_joint[2] - 1]),
    )
    costate = np.array(
        [
            parameters[2],
            np.cos(parameters[0]),
            np.sin(parameters[0]),
            parameters[3],
        ]
    )
    print(
        "S(0):",
        switch_function(np.concatenate((CASE.x0, costate)), CASE.stage.kappa),
    )
    print("S(tf):", switch_function(final_joint, CASE.stage.kappa))
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
    print()


if __name__ == "__main__":
    report("smeared params", SMEARED)
    report("ORACLE params (sanity)", ORACLE)
