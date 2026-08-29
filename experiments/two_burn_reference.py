#!/usr/bin/env python3
"""Independent direct reference for a two-burn single-stage transfer."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import NonlinearConstraint, minimize
from single_stage_research import (
    Case,
    DirectResult,
    PrimerArc,
    PrimerResult,
    Stage,
    direct_solve,
    estimated_burn,
    explicit_solve,
    implicit_solve,
    implicit_solve_from_initial,
    make_cases,
    norm,
)

Array = NDArray[np.float64]


@dataclass
class TwoBurnResult:
    success: bool
    message: str
    parameters: Array
    residual: Array
    fuel: float
    final_state: Array
    n_angles: int


def _research_helpers() -> tuple[
    Callable[
        [Case, Array], tuple[Array, list[float], int, Array, float, list[PrimerArc]]
    ],
    Callable[[Array, float, Stage, float, float], Array],
]:
    from single_stage_research import implicit_propagate, integrate_polar

    return implicit_propagate, integrate_polar


def propagate_two_burn(case: Case, parameters: Array, n_angles: int) -> Array:
    _implicit_propagate, integrate_polar = _research_helpers()
    start, burn_one, coast_gap, burn_two = parameters[:4]
    angles_one = parameters[4 : 4 + n_angles]
    angles_two = parameters[4 + n_angles :]
    state = integrate_polar(case.x0, start, case.stage, 0.0, 0.0)
    for angle in angles_one:
        state = integrate_polar(
            state, burn_one / n_angles, case.stage, 1.0, float(angle)
        )
    state = integrate_polar(state, coast_gap, case.stage, 0.0, 0.0)
    for angle in angles_two:
        state = integrate_polar(
            state, burn_two / n_angles, case.stage, 1.0, float(angle)
        )
    return state


def primer_seed(case: Case, primer: PrimerResult, n_angles: int) -> Array:
    implicit_propagate, _integrate_polar = _research_helpers()
    _state, switches, _events, _final, _thrust, arcs = implicit_propagate(
        case, primer.z
    )
    if len(arcs) != 3 or not (
        arcs[0].throttle > 0.5 and arcs[1].throttle < 0.5 and arcs[2].throttle > 0.5
    ):
        raise ValueError("primer result is not a two-burn trajectory")
    angles: list[float] = []
    for arc in (arcs[0], arcs[2]):
        centers = np.linspace(arc.start, arc.end, n_angles, endpoint=False)
        centers += 0.5 * (arc.end - arc.start) / n_angles
        for center in centers:
            p_r = float(np.interp(center, arc.times, arc.joint[5]))
            p_t = float(np.interp(center, arc.times, arc.joint[6]))
            angles.append(math.atan2(p_t, p_r))
    return np.concatenate(
        (
            [0.0, switches[0], switches[1] - switches[0], primer.z[1] - switches[1]],
            angles,
        ),
        dtype=float,
    )


def one_burn_seed(case: Case, direct: DirectResult, n_angles: int) -> Array:
    """Split a restricted one-burn solution into a direct two-burn seed."""
    old_centers = (np.arange(direct.n_intervals) + 0.5) / direct.n_intervals
    new_centers = (np.arange(n_angles) + 0.5) / n_angles
    angles = np.interp(
        new_centers,
        old_centers,
        direct.angles,
        left=direct.angles[0],
        right=direct.angles[-1],
    )
    start = direct.coast_time
    burn_one = 0.45 * direct.burn_time
    burn_two = 0.45 * direct.burn_time
    available_gap = case.first_arc_limit - start - burn_one - burn_two
    gap = max(0.0, min(case.time_to_apoapsis - start - burn_one, available_gap))
    return np.concatenate(
        ([start, burn_one, gap, burn_two], angles, angles), dtype=float
    )


def solve_two_burn(
    case: Case,
    primer: PrimerResult | None,
    n_angles: int,
    initial: Array | None = None,
) -> TwoBurnResult:
    if initial is None:
        if primer is None:
            raise ValueError("a primer or direct seed is required")
        initial = primer_seed(case, primer, n_angles)
    lower = np.concatenate(
        (
            [0.0, 1e-8, 0.0, 1e-8],
            np.full(2 * n_angles, -math.pi),
        )
    )
    upper = np.concatenate(
        (
            [
                case.time_to_apoapsis,
                case.stage.max_burn,
                case.first_arc_limit,
                case.stage.max_burn,
            ],
            np.full(2 * n_angles, math.pi),
        )
    )

    def terminal_residual(parameters: Array) -> Array:
        try:
            state = propagate_two_burn(case, parameters, n_angles)
            return np.array([state[0] - 1.0, state[1], state[2] - 1.0])
        except (FloatingPointError, ValueError, ZeroDivisionError):
            return np.full(3, 1e3)

    def time_constraint(parameters: Array) -> Array:
        return np.array(
            [
                case.first_arc_limit
                - parameters[0]
                - parameters[1]
                - parameters[2]
                - parameters[3]
            ],
            dtype=float,
        )

    def objective(parameters: Array) -> float:
        return float(
            case.stage.gamma * (parameters[1] + parameters[3]) / case.stage.kappa
        )

    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=list(zip(lower, upper, strict=True)),
        constraints=(
            NonlinearConstraint(terminal_residual, 0.0, 0.0),
            NonlinearConstraint(time_constraint, 0.0, np.inf),
        ),
        options={"ftol": 2e-11, "maxiter": 800, "disp": False},
    )
    state = propagate_two_burn(case, result.x, n_angles)
    residual = terminal_residual(result.x)
    return TwoBurnResult(
        success=bool(result.success and norm(residual) < 2e-6),
        message=str(result.message),
        parameters=np.asarray(result.x, dtype=float),
        residual=residual,
        fuel=float(result.fun),
        final_state=state,
        n_angles=n_angles,
    )


def solve_two_burn_from_direct(
    case: Case, direct: DirectResult, n_angles: int
) -> TwoBurnResult:
    return solve_two_burn(
        case,
        primer=None,
        n_angles=n_angles,
        initial=one_burn_seed(case, direct, n_angles),
    )


def informed_two_burn_seeds(
    case: Case, direct: DirectResult | None, n_angles: int
) -> list[np.ndarray]:
    """Return deterministic vis-viva two-impulse seeds."""
    seeds: list[np.ndarray] = []
    if direct is not None:
        seeds.append(one_burn_seed(case, direct, n_angles))
    burn_estimate = estimated_burn(case)
    starts = (0.0,) if direct is None else (0.0, direct.coast_time)
    for start in starts:
        for ratio in (0.2, 0.4, 0.6, 0.8):
            for factor in (1.0, 1.2):
                total_burn = min(
                    case.stage.max_burn * 0.9,
                    max(1e-5, burn_estimate * factor),
                )
                burn_one = total_burn * ratio
                burn_two = total_burn - burn_one
                available_gap = case.first_arc_limit - start - total_burn
                gap = max(
                    0.0,
                    min(case.time_to_apoapsis - start - burn_one, available_gap),
                )
                seeds.append(
                    np.concatenate(
                        (
                            [start, burn_one, gap, burn_two],
                            np.full(2 * n_angles, math.pi / 2.0),
                        ),
                        dtype=float,
                    )
                )
    return seeds


def solve_two_burn_multistart(
    case: Case, direct: DirectResult | None, n_angles: int
) -> TwoBurnResult:
    """Choose the best accepted result from deterministic informed seeds."""
    results = [
        solve_two_burn(case, primer=None, n_angles=n_angles, initial=seed)
        for seed in informed_two_burn_seeds(case, direct, n_angles)
    ]
    accepted = [result for result in results if result.success]
    if accepted:
        return min(accepted, key=lambda result: result.fuel)
    return min(results, key=lambda result: norm(result.residual))


def get_primer(case: Case) -> PrimerResult:
    direct = get_restricted_direct(case)
    if direct is None or not direct.success:
        raise ValueError(f"direct seed failed for {case.name}")
    explicit = explicit_solve(case, direct)
    primer = implicit_solve(case, explicit)
    if primer.success:
        return primer
    base = __import__(
        "single_stage_research", fromlist=["implicit_initial_guess"]
    ).implicit_initial_guess(case, explicit)
    for factor in (0.25, 0.5, 0.75, 1.25, 1.5, 2.0, 3.0, 4.0):
        adjusted = base.copy()
        adjusted[3] *= factor
        candidate = implicit_solve_from_initial(case, adjusted)
        if candidate.success:
            return candidate
    raise ValueError(f"implicit primer solve failed for {case.name}")


def get_restricted_direct(case: Case) -> DirectResult:
    direct: DirectResult | None = None
    for mesh in (4, 8, 16):
        direct = direct_solve(case, mesh, direct)
    if direct is None:
        raise ValueError(f"direct seed failed for {case.name}")
    return direct


def get_two_burn_reference(case: Case, n_angles: int) -> TwoBurnResult:
    direct = get_restricted_direct(case)
    try:
        primer = get_primer(case)
    except (FloatingPointError, ValueError, ZeroDivisionError):
        return solve_two_burn_multistart(case, direct, n_angles)
    return solve_two_burn(case, primer, n_angles)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--output", type=str, default="")
    arguments = parser.parse_args()
    records: list[dict[str, object]] = []
    for case in make_cases()[: arguments.limit]:
        direct = get_restricted_direct(case)
        try:
            primer = get_primer(case)
        except (FloatingPointError, ValueError, ZeroDivisionError):
            primer = None
        results = [
            solve_two_burn(case, primer, n)
            if primer is not None
            else solve_two_burn_multistart(case, direct, n)
            for n in (2, 4, 8)
        ]
        for result in results:
            print(
                f"{case.name:22s} two-burn N={result.n_angles:2d} "
                f"ok={result.success!s:5s} fuel={result.fuel:.9f} "
                f"res={norm(result.residual):.3e}"
            )
        records.append(
            {
                "case": case.name,
                "primer_fuel": None
                if primer is None
                else case.stage.gamma * primer.burn_time / case.stage.kappa,
                "primer_switch_times": None if primer is None else primer.switch_times,
                "meshes": [
                    {
                        "n": result.n_angles,
                        "success": result.success,
                        "message": result.message,
                        "fuel": result.fuel,
                        "residual_norm": norm(result.residual),
                        "parameters": result.parameters.tolist(),
                    }
                    for result in results
                ],
            }
        )
    print(json.dumps(records, indent=2))
    if arguments.output:
        with open(arguments.output, "w", encoding="utf-8") as output_file:
            json.dump(records, output_file, indent=2)
            output_file.write("\n")


if __name__ == "__main__":
    main()
