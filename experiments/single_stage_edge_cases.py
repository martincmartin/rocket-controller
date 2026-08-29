#!/usr/bin/env python3
"""Probe single-stage edge regimes without changing the production simulator."""

from __future__ import annotations

import argparse
import json
import math

from single_stage_research import (
    Case,
    DirectResult,
    direct_solve,
    explicit_solve,
    implicit_solve,
    norm,
)
from two_burn_reference import solve_two_burn_multistart


def edge_cases() -> list[Case]:
    from single_stage_research import Stage, synthetic_case

    return [
        synthetic_case(
            "near-target",
            Stage(gamma=1.8, kappa=1.35, max_burn=0.6),
            rp=0.78,
            ra=0.96,
            true_anomaly=math.radians(100.0),
        ),
        synthetic_case(
            "low-thrust",
            Stage(gamma=0.5, kappa=1.35, max_burn=1.5),
            rp=0.62,
            ra=0.86,
            true_anomaly=math.radians(118.0),
        ),
        synthetic_case(
            "very-high-thrust",
            Stage(gamma=10.0, kappa=1.35, max_burn=0.1),
            rp=0.66,
            ra=0.88,
            true_anomaly=math.radians(128.0),
        ),
        synthetic_case(
            "early-outbound",
            Stage(gamma=2.0, kappa=1.35, max_burn=0.6),
            rp=0.55,
            ra=0.90,
            true_anomaly=math.radians(80.0),
        ),
        synthetic_case(
            "near-apoapsis",
            Stage(gamma=2.0, kappa=1.35, max_burn=0.6),
            rp=0.65,
            ra=0.90,
            true_anomaly=math.radians(160.0),
        ),
    ]


def solve_case(case: Case) -> dict[str, object]:
    direct: DirectResult | None = None
    for mesh in (4, 8, 16):
        direct = direct_solve(case, mesh, direct)
    explicit = (
        None if direct is None or not direct.success else explicit_solve(case, direct)
    )
    primer = None if explicit is None else implicit_solve(case, explicit)
    primer_source = "default implicit" if primer is not None else "no one-burn seed"
    direct_for_two = direct if direct is not None and direct.success else None
    result = solve_two_burn_multistart(case, direct_for_two, 8)
    two_burn: dict[str, object] = {
        "success": result.success,
        "fuel": result.fuel,
        "residual_norm": norm(result.residual),
        "final_time": float(sum(result.parameters[:4])),
    }
    if primer is not None and not primer.success:
        primer_source = "direct two-burn fallback"
    return {
        "case": case.name,
        "time_to_apoapsis": case.time_to_apoapsis,
        "first_arc_limit": case.first_arc_limit,
        "direct_one_burn_fuel": None
        if direct is None
        else case.stage.gamma * direct.burn_time / case.stage.kappa,
        "explicit_sign_valid": bool(
            False
            if explicit is None
            else __import__(
                "single_stage_research", fromlist=["explicit_switch_diagnostics"]
            ).explicit_switch_diagnostics(case, explicit)["satisfies_candidate_sign"]
        ),
        "implicit": {
            "success": False if primer is None else primer.success,
            "source": primer_source,
            "fuel": None
            if primer is None
            else case.stage.gamma * primer.burn_time / case.stage.kappa,
            "residual_norm": None if primer is None else norm(primer.residual),
            "switch_times": [] if primer is None else primer.switch_times,
        },
        "two_burn": two_burn,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="")
    arguments = parser.parse_args()
    records = [solve_case(case) for case in edge_cases()]
    print(json.dumps(records, indent=2))
    if arguments.output:
        with open(arguments.output, "w", encoding="utf-8") as output_file:
            json.dump(records, output_file, indent=2)
            output_file.write("\n")


if __name__ == "__main__":
    main()
