#!/usr/bin/env python3
"""Check single-stage boundary states that should not enter ordinary shooting."""

from __future__ import annotations

import argparse
import json

import numpy as np
from single_stage_research import Stage, integrate_polar, norm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="")
    arguments = parser.parse_args()
    stage = Stage(gamma=2.0, kappa=1.4, max_burn=0.5)
    exact_circular = np.array([1.0, 0.0, 1.0, 1.0])
    coast_final = integrate_polar(exact_circular, 3.0, stage, 0.0, 0.0)
    near_radial = np.array([0.8, 0.25, 0.05, 1.0])
    near_radial_final = integrate_polar(near_radial, 0.1, stage, 0.0, 0.0)
    records = [
        {
            "case": "exact-circular",
            "terminal_residual_norm": norm(coast_final[:3] - exact_circular[:3]),
            "mass_change": float(coast_final[3] - exact_circular[3]),
            "expected_zero_burn_fuel": 0.0,
        },
        {
            "case": "near-radial-coast",
            "initial_tangential_speed": float(near_radial[2]),
            "final_state": near_radial_final.tolist(),
            "finite": bool(np.all(np.isfinite(near_radial_final))),
        },
    ]
    print(json.dumps(records, indent=2))
    if arguments.output:
        with open(arguments.output, "w", encoding="utf-8") as output_file:
            json.dump(records, output_file, indent=2)
            output_file.write("\n")


if __name__ == "__main__":
    main()
