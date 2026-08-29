#!/usr/bin/env python3
"""Measure how the selected two-burn solution changes as the time cap grows."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace

import numpy as np
from initial_guess_comparison import kerbin_case_from_vectors
from single_stage_research import Case, DirectResult, direct_solve, make_cases
from two_arc_shooting import solve
from two_burn_reference import solve_two_burn_multistart


def supplied_first_kerbin() -> Case:
    return kerbin_case_from_vectors(
        "kerbin-first-example",
        np.array([428392.15435586, -1053.61873734, -455905.93323801]),
        np.array([1.03031015e3, -9.32270447e-1, -1.19588146e2]),
        initial_mass=13057.14453125,
        max_burn_seconds=150.0,
    )


def restricted_direct(case: Case) -> DirectResult:
    direct = None
    for mesh in (4, 8, 16):
        direct = direct_solve(case, mesh, direct)
    if direct is None or not direct.success:
        raise ValueError(f"restricted direct seed failed for {case.name}")
    return direct


def sweep_case(case: Case, cap_offsets: list[float]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for offset in cap_offsets:
        capped = replace(case, first_arc_limit=case.first_arc_limit + offset)
        direct = restricted_direct(capped)
        two_burn = solve_two_burn_multistart(capped, direct, 8)
        fixed = solve(capped, two_burn)
        records.append(
            {
                "cap_offset": offset,
                "cap": capped.first_arc_limit,
                "direct_fuel": two_burn.fuel,
                "fixed_fuel": fixed["fuel"],
                "final_time": fixed["final_time"],
                "active": fixed["final_time_active"],
                "residual_norm": fixed["residual_norm"],
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="")
    arguments = parser.parse_args()
    cases = [make_cases()[3], make_cases()[2], supplied_first_kerbin()]
    offsets = [0.0, 0.25, 0.5, 1.0]
    output: dict[str, list[dict[str, object]]] = {}
    for case in cases:
        output[case.name] = sweep_case(case, offsets)
        for record in output[case.name]:
            print(
                f"{case.name:22s} offset={record['cap_offset']:.2f} "
                f"fuel={record['fixed_fuel']:.9f} "
                f"tf={record['final_time']:.9f} "
                f"active={record['active']!s:5s} "
                f"res={record['residual_norm']:.3e}"
            )
    print(json.dumps(output, indent=2))
    if arguments.output:
        with open(arguments.output, "w", encoding="utf-8") as output_file:
            json.dump(output, output_file, indent=2)
            output_file.write("\n")


if __name__ == "__main__":
    main()
