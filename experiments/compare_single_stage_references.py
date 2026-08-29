#!/usr/bin/env python3
"""Compare restricted direct, primer, and relaxed direct single-stage solves."""

from __future__ import annotations

import argparse
import json
from contextlib import suppress

from single_stage_research import (
    Case,
    DirectResult,
    PrimerResult,
    RelaxedResult,
    direct_solve,
    explicit_solve,
    implicit_solve,
    make_cases,
    norm,
    primer_seeded_relaxed_solve,
    relaxed_direct_solve,
)


def solve_case(
    case: Case, direct_meshes: list[int], relaxed_meshes: list[int]
) -> dict[str, object]:
    direct_results: list[DirectResult] = []
    previous_direct: DirectResult | None = None
    for mesh in direct_meshes:
        previous_direct = direct_solve(case, mesh, previous_direct)
        direct_results.append(previous_direct)
    direct = direct_results[-1]
    if not direct.success:
        raise ValueError(f"direct solve failed for {case.name}")

    explicit = explicit_solve(case, direct)
    implicit: PrimerResult | None = None
    relaxed_results: list[RelaxedResult] = []
    with suppress(FloatingPointError, ValueError, ZeroDivisionError):
        implicit = implicit_solve(case, explicit)
    if implicit is not None and implicit.success:
        previous_relaxed: RelaxedResult | None = None
        for mesh in relaxed_meshes:
            if previous_relaxed is None:
                previous_relaxed = primer_seeded_relaxed_solve(case, implicit, mesh)
            else:
                previous_relaxed = relaxed_direct_solve(
                    case, mesh, previous=previous_relaxed
                )
            relaxed_results.append(previous_relaxed)

    return {
        "case": case.name,
        "stage": {
            "gamma": case.stage.gamma,
            "kappa": case.stage.kappa,
            "max_burn": case.stage.max_burn,
        },
        "direct_meshes": [
            {
                "mesh": result.n_intervals,
                "success": result.success,
                "coast_time": result.coast_time,
                "burn_time": result.burn_time,
                "fuel": case.stage.gamma * result.burn_time / case.stage.kappa,
                "residual_norm": norm(result.residual),
            }
            for result in direct_results
        ],
        "explicit": {
            "success": explicit.success,
            "parameters": explicit.z.tolist(),
            "fuel": case.stage.gamma * explicit.burn_time / case.stage.kappa,
            "residual_norm": norm(explicit.residual),
            "switch_diagnostics": explicit_switch_diagnostics(case, explicit),
        },
        "implicit": None
        if implicit is None
        else {
            "success": implicit.success,
            "parameters": implicit.z.tolist(),
            "fuel": case.stage.gamma * implicit.burn_time / case.stage.kappa,
            "residual_norm": norm(implicit.residual),
            "switch_times": implicit.switch_times,
            "event_count": implicit.event_count,
            "arcs": _implicit_arc_records(case, implicit),
        },
        "relaxed_meshes": [
            {
                "mesh": result.n_intervals,
                "success": result.success,
                "horizon": result.horizon,
                "fuel": result.objective,
                "residual_norm": norm(result.residual),
                "throttle_min": float(result.throttles.min()),
                "throttle_max": float(result.throttles.max()),
            }
            for result in relaxed_results
        ],
    }


def explicit_switch_diagnostics(
    case: Case, primer: PrimerResult
) -> dict[str, float | bool]:
    # Imported lazily to keep the public list above focused on solver helpers.
    from single_stage_research import explicit_switch_diagnostics as diagnostics

    return diagnostics(case, primer)


def _implicit_arc_records(case: Case, primer: PrimerResult) -> list[dict[str, float]]:
    from single_stage_research import implicit_propagate, switch_function

    _state, _switches, _events, _final, _thrust, arcs = implicit_propagate(
        case, primer.z
    )
    records: list[dict[str, float]] = []
    for arc in arcs:
        phi = [
            switch_function(arc.joint[:, index], case.stage.kappa)
            for index in range(arc.joint.shape[1])
        ]
        records.append(
            {
                "start": arc.start,
                "end": arc.end,
                "throttle": arc.throttle,
                "phi_min": float(min(phi)),
                "phi_max": float(max(phi)),
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--output", type=str, default="")
    arguments = parser.parse_args()
    records = [
        solve_case(case, [4, 8, 16, 32], [16, 32, 48])
        for case in make_cases()[: arguments.limit]
    ]
    print(json.dumps(records, indent=2))
    if arguments.output:
        with open(arguments.output, "w", encoding="utf-8") as output_file:
            json.dump(records, output_file, indent=2)
            output_file.write("\n")


if __name__ == "__main__":
    main()
