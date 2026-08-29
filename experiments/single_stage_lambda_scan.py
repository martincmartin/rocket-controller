#!/usr/bin/env python3
"""Deterministic scan of the implicit primer mass-costate initialization.

The other implicit parameters are initialized from the informed explicit-coast
estimate.  Only the initial mass costate is multiplied by the listed factors,
as required by the research protocol.
"""

from __future__ import annotations

import argparse
import json

from single_stage_research import (
    Case,
    direct_solve,
    explicit_solve,
    implicit_initial_guess,
    implicit_solve_from_initial,
    make_cases,
    norm,
)


def scan_case(case: Case, factors: list[float]) -> list[dict[str, object]]:
    direct = None
    for intervals in (4, 8, 16):
        direct = direct_solve(case, intervals, direct)
    if direct is None or not direct.success:
        raise ValueError(f"direct seed failed for {case.name}")
    explicit = explicit_solve(case, direct)
    base = implicit_initial_guess(case, explicit)
    records: list[dict[str, object]] = []
    for factor in factors:
        initial = base.copy()
        initial[3] *= factor
        result = implicit_solve_from_initial(case, initial)
        records.append(
            {
                "factor": factor,
                "success": result.success,
                "message": result.message,
                "parameters": result.z.tolist(),
                "residual_norm": norm(result.residual),
                "fuel_used": case.stage.gamma * result.burn_time / case.stage.kappa,
                "burn_time": result.burn_time,
                "switch_times": result.switch_times,
                "event_count": result.event_count,
                "final_state": result.x_final.tolist(),
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--output", type=str, default="")
    arguments = parser.parse_args()
    factors = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0]
    output = {
        case.name: scan_case(case, factors) for case in make_cases()[: arguments.limit]
    }
    print(json.dumps(output, indent=2))
    if arguments.output:
        with open(arguments.output, "w", encoding="utf-8") as output_file:
            json.dump(output, output_file, indent=2)
            output_file.write("\n")


if __name__ == "__main__":
    main()
