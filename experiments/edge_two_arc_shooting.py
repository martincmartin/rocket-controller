#!/usr/bin/env python3
"""Run fixed two-arc primer shooting on the single-stage edge cases."""

from __future__ import annotations

import argparse
import json

from single_stage_edge_cases import edge_cases
from single_stage_research import direct_solve, norm
from two_arc_shooting import solve
from two_burn_reference import solve_two_burn_multistart


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="")
    arguments = parser.parse_args()
    records: list[dict[str, object]] = []
    for case in edge_cases():
        direct = None
        for mesh in (4, 8, 16):
            direct = direct_solve(case, mesh, direct)
        direct_for_two = direct if direct is not None and direct.success else None
        direct_two_burn = solve_two_burn_multistart(case, direct_for_two, 8)
        shooting = solve(case, direct_two_burn)
        print(
            f"{case.name:22s} direct={direct_two_burn.fuel:.9f} "
            f"shooting={shooting['fuel']:.9f} "
            f"ok={shooting['success']!s:5s} "
            f"res={shooting['residual_norm']:.3e}"
        )
        records.append(
            {
                "case": case.name,
                "direct_two_burn_fuel": direct_two_burn.fuel,
                "direct_two_burn_residual": norm(direct_two_burn.residual),
                "shooting": shooting,
            }
        )
    print(json.dumps(records, indent=2))
    if arguments.output:
        with open(arguments.output, "w", encoding="utf-8") as output_file:
            json.dump(records, output_file, indent=2)
            output_file.write("\n")


if __name__ == "__main__":
    main()
