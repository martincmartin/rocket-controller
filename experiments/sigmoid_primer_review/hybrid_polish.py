#!/usr/bin/env python3
"""Hybrid polish test: seed the hard-switch event-based shooter from sigmoid
continuation results.

Evidence for `review-sigmoid-primer.md` section 3.6. The structure-free sharp
shooter (`single_stage_research.implicit_solve_from_initial`) uses the same
four shooting variables as the sigmoid system and re-derives the burn/coast
partition from switching-function crossings, so it can be seeded directly
with the sigmoid parameters -- including the smeared root. From the smeared
root it recovers the correct on/off/on sequence even though its default
500-evaluation budget leaves the final residual unpolished.

Run from the repository root with ``PYTHONPATH=experiments``::

    python3 experiments/sigmoid_primer_review/hybrid_polish.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sigmoid_primer as sp
from single_stage_research import (
    implicit_propagate,
    implicit_solve_from_initial,
)

CASE = sp.kerbin_cases()[0]
ORACLE = np.array(
    [1.357422910819774, 0.648432960860141, -1.120006399999580, -1.373896477245226]
)
SMEARED = np.array(
    [1.3557489599336592, 0.7149047496714878, -1.1228459941245825, -1.3730707279997247]
)
ROUGH_1E5 = np.array([1.35, 0.673443573, -1.12, -1.373])


def polish(tag: str, seed: np.ndarray) -> None:
    try:
        result = implicit_solve_from_initial(CASE, seed)
        _state, switches, _events, final_joint, thrust_time, _arcs = implicit_propagate(
            CASE, result.z
        )
        print(
            f"sharp-from-{tag}: ok={result.success} {result.message[:34]:34s} "
            f"fuel={1 - final_joint[3]:.9f} res={np.linalg.norm(result.residual):.2e} "
            f"z={np.round(result.z, 6)} switches={[f'{s:.5f}' for s in switches]} "
            f"thrust={thrust_time:.6f}"
        )
    except (ValueError, FloatingPointError) as error:
        print(f"sharp-from-{tag}: FAILED {error}")


if __name__ == "__main__":
    polish("oracle", ORACLE)
    polish("smeared-root-1e-4", SMEARED)
    polish("rough-1e-5", ROUGH_1E5)
