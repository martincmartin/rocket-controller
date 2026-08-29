#!/usr/bin/env python3
"""Benchmark the analytic-seed-only fixed-sequence single-stage solve."""

from __future__ import annotations

import argparse
import math
from time import perf_counter

import numpy as np
from initial_guess_comparison import (
    impulse_timing_estimate,
    initial_parameters,
    kerbin_case_from_vectors,
)
from single_stage_research import make_cases
from two_arc_shooting import solve_from_initial


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=5)
    arguments = parser.parse_args()
    first = kerbin_case_from_vectors(
        "kerbin-first-example",
        np.array([428392.15435586, -1053.61873734, -455905.93323801]),
        np.array([1.03031015e3, -9.32270447e-1, -1.19588146e2]),
        initial_mass=13057.14453125,
        max_burn_seconds=150.0,
    )
    cases = [make_cases()[2], first]
    for case in cases:
        timings: list[float] = []
        result: dict[str, object] = {}
        for _ in range(arguments.repetitions):
            started = perf_counter()
            timing = impulse_timing_estimate(case)
            seed = initial_parameters(case, math.atan2(case.x0[2], case.x0[1]), timing)
            result = solve_from_initial(
                case,
                seed,
                final_time_active=timing["final_time"] >= case.first_arc_limit - 1e-6,
            )
            timings.append(perf_counter() - started)
        print(
            f"{case.name:22s} mean={np.mean(timings):.6f}s "
            f"min={np.min(timings):.6f}s max={np.max(timings):.6f}s "
            f"accepted={result['success']!s:5s} "
            f"residual={result['residual_norm']:.3e}"
        )


if __name__ == "__main__":
    main()
