#!/usr/bin/env python3
"""Production-style single-stage ascent solver: sigmoid-primer continuation
with adaptive epsilon, finished by a hard-switch event-based polish.

This is the "final suggestion" pipeline of `review-sigmoid-primer.md`
(sections 3.4 and 3.6), assembled into one solver:

1. Sigmoid continuation with adaptive epsilon.
   - The first solve (epsilon = 1) uses the bounded `trf` method from the
     analytic initial guess in `sigmoid_primer.initial_sigmoid_guess`.
   - Subsequent epsilon values use the unconstrained `lm` method (much
     cheaper), with an automatic `trf` fallback when `lm` leaves the bounds
     or fails to improve the residual.
   - After each solve epsilon is divided by 3, until the *rendered* control
     is bang-bang (q_min < 0.01 and q_max > 0.99) or the epsilon floor is
     reached. This replaces the fixed schedule and adapts automatically to
     each case: healthy cases stop early, near-singular cases keep going.
2. Sharp polish. The structure-free hard-switch shooter
   (`single_stage_research.implicit_propagate` with the same four shooting
   variables) is seeded with the sigmoid parameters. It re-derives the
   burn/coast partition from the switching function and returns exact
   event times, so a smeared sigmoid endpoint still yields the correct
   structure. The polish uses `lm` first and falls back to bounded `trf`
   when `lm` leaves the bounds.

Solver details:
- All solves use finite differences. No analytic Jacobian.
- The sigmoid integrator tries DOP853 first and falls back to LSODA when
  the transition layer makes the explicit integrator fail.
- Accept/reject per step: a solver result is adopted only when it does not
  worsen the shooting residual at that epsilon.

Run from the repository root with ``PYTHONPATH=experiments``::

    python3 experiments/sigmoid_primer_review/production_solver.py

Use ``--case NAME`` to run a single case and ``--output FILE`` to write the
full JSON record.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypedDict, cast

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from scipy.integrate._ivp.ivp import OdeResult
from scipy.optimize import least_squares
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sigmoid_primer as sp
from single_stage_research import (
    Case,
    implicit_propagate,
    implicit_residual,
    norm,
    switch_function,
)

Array = NDArray[np.float64]
IntegrationMethod = Literal["DOP853", "LSODA"]


class ThrottleProfile(TypedDict):
    q_min: float
    q_max: float
    crossings: list[float]
    fuel_fraction: float


class Reference(TypedDict, total=False):
    fuel: float
    tf: float
    switches: list[float]


EPS_START = 1.0
EPS_FACTOR = 3.0
EPS_FLOOR = 1e-7
MAX_EPS_STEPS = 40
RENDER_Q_MIN = 0.01
RENDER_Q_MAX = 0.99
RESIDUAL_SUCCESS = 2e-6
SIGMOID_NFEV = 1000
POLISH_NFEV = 5000
SIGMOID_TOLERANCES = {"ftol": 2e-10, "xtol": 2e-10, "gtol": 2e-10}
POLISH_TOLERANCES = {"ftol": 2e-12, "xtol": 2e-12, "gtol": 2e-12}

REFERENCES: dict[str, Reference] = {
    "kerbin-example": {
        "fuel": 0.458673471,
        "tf": 0.648432961,
        "switches": [0.25891, 0.59580],
    },
    "kerbin-first-example": {"fuel": 0.417870055, "tf": 0.522074908},
    "kerbin-coast-first-example": {"fuel": 0.339814106, "tf": 0.510560727},
}

# Solver marker codes in the epsilon history records.
FIRST_STEP = 0
LM_STEP = 1
TRF_STEP = 2
INHERITED = 3


@dataclass
class SolverOutcome:
    case_name: str
    success: bool
    message: str
    phase: str
    parameters: list[float]
    residual: list[float]
    switch_times: list[float]
    thrust_time: float
    fuel_fraction: float
    final_state: list[float]
    arcs: list[dict[str, float]]
    eps_stop: float
    sigmoid_nfev: int
    polish_nfev: int
    eps_history: list[dict[str, float]]
    wall_seconds: float
    reference: Reference = field(default_factory=lambda: cast(Reference, {}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "case": self.case_name,
            "success": self.success,
            "message": self.message,
            "phase": self.phase,
            "parameters": self.parameters,
            "residual": self.residual,
            "residual_norm": norm(np.asarray(self.residual, dtype=float)),
            "switch_times": self.switch_times,
            "thrust_time": self.thrust_time,
            "fuel_fraction": self.fuel_fraction,
            "final_state": self.final_state,
            "arcs": self.arcs,
            "eps_stop": self.eps_stop,
            "sigmoid_nfev": self.sigmoid_nfev,
            "polish_nfev": self.polish_nfev,
            "eps_history": self.eps_history,
            "wall_seconds": self.wall_seconds,
            "reference": self.reference,
        }


def sigmoid_bounds(case: Case) -> tuple[Array, Array]:
    lower = np.array([-math.pi, 1e-5, -100.0, -100.0], dtype=float)
    upper = np.array([math.pi, case.first_arc_limit, 100.0, 100.0], dtype=float)
    return lower, upper


def in_bounds(parameters: Array, lower: Array, upper: Array) -> bool:
    return bool(
        np.all(np.isfinite(parameters))
        and np.all(parameters >= lower)
        and np.all(parameters <= upper)
    )


def integrate_sigmoid_robust(
    case: Case, parameters: Array, epsilon: float, *, dense_output: bool = False
) -> OdeResult:
    """Integrate the sigmoid system, falling back to LSODA when DOP853 fails."""
    joint0 = sp.initial_joint(case, parameters)

    def try_method(
        method: IntegrationMethod,
    ) -> tuple[bool, OdeResult | None, str]:
        try:
            solution = solve_ivp(
                sp.sigmoid_primer_rhs,
                (0.0, float(parameters[1])),
                joint0,
                method=method,
                args=(case.stage.gamma, case.stage.kappa, epsilon),
                rtol=sp.RTOL,
                atol=sp.ATOL,
                dense_output=dense_output,
            )
            if solution.success and np.all(np.isfinite(solution.y[:, -1])):
                return True, solution, ""
            return False, None, solution.message
        except (FloatingPointError, ValueError, ZeroDivisionError) as error:
            return False, None, str(error)

    failures: list[str] = []
    methods: tuple[IntegrationMethod, ...] = ("DOP853", "LSODA")
    for method in methods:
        success, solution, failure = try_method(method)
        if success and solution is not None:
            return solution
        failures.append(failure)
    raise ValueError("sigmoid integration failed: " + "; ".join(failures))


def rendered_throttle(case: Case, parameters: Array, epsilon: float) -> ThrottleProfile:
    solution = integrate_sigmoid_robust(case, parameters, epsilon, dense_output=True)
    if solution.sol is None:
        raise ValueError("dense output was not created")
    samples = np.linspace(0.0, float(parameters[1]), 1601)
    joints = np.asarray(solution.sol(samples).T, dtype=float)
    switch = np.array([sp.switching_function(j, case.stage.kappa) for j in joints])
    throttle = expit(switch / epsilon)
    return {
        "q_min": float(np.min(throttle)),
        "q_max": float(np.max(throttle)),
        "crossings": [
            float(c) for c in sp.switch_crossings(samples, switch, parameters[1])
        ],
        "fuel_fraction": float(1.0 - joints[-1, 3]),
    }


def sigmoid_solve(
    case: Case,
    seed: Array,
    epsilon: float,
    method: Literal["trf", "lm"],
    max_nfev: int,
) -> tuple[Array, Any, float]:
    kwargs: dict[str, Any] = {
        "fun": lambda p: sp.sigmoid_residual(case, p, epsilon),
        "x0": seed,
        "method": method,
        "x_scale": "jac",
        "max_nfev": max_nfev,
        **SIGMOID_TOLERANCES,
    }
    if method != "lm":
        kwargs["bounds"] = sigmoid_bounds(case)
    result = least_squares(**kwargs)
    parameters = np.asarray(result.x, dtype=float)
    residual = sp.sigmoid_residual(case, parameters, epsilon)
    return parameters, result, float(np.linalg.norm(residual))


def render_ok(profile: ThrottleProfile) -> bool:
    return bool(
        float(profile["q_min"]) < RENDER_Q_MIN
        and float(profile["q_max"]) > RENDER_Q_MAX
    )


def sigmoid_continuation(
    case: Case,
) -> tuple[Array, float, list[dict[str, float]], int, str]:
    lower, upper = sigmoid_bounds(case)
    try:
        seed = sp.initial_sigmoid_guess(case)
    except (FloatingPointError, ValueError, ZeroDivisionError):
        seed = np.array(
            [math.pi / 2.0, max(case.time_to_apoapsis, 1e-3), 0.0, -case.stage.kappa],
            dtype=float,
        )
    parameters: Array = seed.copy()
    epsilon = EPS_START
    total_nfev = 0
    history: list[dict[str, float]] = []
    stop_reason = "epsilon-floor"

    parameters, result, residual = sigmoid_solve(
        case, parameters, epsilon, "trf", SIGMOID_NFEV
    )
    total_nfev += int(result.nfev)
    profile = rendered_throttle(case, parameters, epsilon)
    history.append(
        {
            "epsilon": epsilon,
            "nfev": float(result.nfev),
            "residual": residual,
            "q_min": float(profile["q_min"]),
            "q_max": float(profile["q_max"]),
            "solver": float(FIRST_STEP),
        }
    )
    if render_ok(profile):
        stop_reason = "render-ok"

    while (
        not render_ok(profile) and epsilon > EPS_FLOOR and len(history) < MAX_EPS_STEPS
    ):
        epsilon /= EPS_FACTOR
        inherited_residual = norm(sp.sigmoid_residual(case, parameters, epsilon))
        chosen: tuple[Array, float, int, int] | None = None

        p_lm, result_lm, res_lm = sigmoid_solve(
            case, parameters, epsilon, "lm", SIGMOID_NFEV
        )
        total_nfev += int(result_lm.nfev)
        if in_bounds(p_lm, lower, upper) and res_lm <= inherited_residual * 1.01:
            chosen = (p_lm, res_lm, int(result_lm.nfev), LM_STEP)
        else:
            p_trf, result_trf, res_trf = sigmoid_solve(
                case, parameters, epsilon, "trf", SIGMOID_NFEV
            )
            total_nfev += int(result_trf.nfev)
            if in_bounds(p_trf, lower, upper) and res_trf <= inherited_residual * 1.01:
                chosen = (p_trf, res_trf, int(result_trf.nfev), TRF_STEP)

        if chosen is None:
            history.append(
                {
                    "epsilon": epsilon,
                    "nfev": 0.0,
                    "residual": inherited_residual,
                    "q_min": float("nan"),
                    "q_max": float("nan"),
                    "solver": float(INHERITED),
                }
            )
            continue
        parameters, residual, step_nfev, marker = chosen
        profile = rendered_throttle(case, parameters, epsilon)
        history.append(
            {
                "epsilon": epsilon,
                "nfev": float(step_nfev),
                "residual": residual,
                "q_min": float(profile["q_min"]),
                "q_max": float(profile["q_max"]),
                "solver": float(marker),
            }
        )
        if render_ok(profile):
            stop_reason = "render-ok"

    return parameters, epsilon, history, total_nfev, stop_reason


def sharp_solve(
    case: Case,
    seed: Array,
    method: Literal["trf", "lm"],
    max_nfev: int,
) -> tuple[Array, Any, float]:
    kwargs: dict[str, Any] = {
        "fun": lambda p: implicit_residual(case, p),
        "x0": seed,
        "method": method,
        "x_scale": "jac",
        "max_nfev": max_nfev,
        **POLISH_TOLERANCES,
    }
    if method != "lm":
        kwargs["bounds"] = sigmoid_bounds(case)
    result = least_squares(**kwargs)
    parameters = np.asarray(result.x, dtype=float)
    residual = implicit_residual(case, parameters)
    return parameters, result, float(norm(residual))


def sharp_polish(
    case: Case, seed: Array
) -> tuple[
    Array, Any, Array, list[float], float, float, list[dict[str, float]], list[float]
]:
    """Polish with both lm and trf and keep the smaller-residual result.

    `lm` is usually the better and cheaper choice (review section 3.3), but
    on some cases it can stop in a spurious local minimum of the squared
    residual; `trf` recovers those cases. Picking by final residual handles
    both outcomes.
    """
    lower, upper = sigmoid_bounds(case)
    candidates: list[tuple[float, Array, Any, str]] = []
    p_lm, result_lm, res_lm = sharp_solve(case, seed, "lm", POLISH_NFEV)
    if in_bounds(p_lm, lower, upper):
        candidates.append((res_lm, p_lm, result_lm, "lm"))
    p_trf, result_trf, res_trf = sharp_solve(case, seed, "trf", POLISH_NFEV)
    candidates.append((res_trf, p_trf, result_trf, "trf"))
    _, parameters, result, method = min(candidates, key=lambda item: item[0])
    result.message = f"{method}: {result.message}"
    residual = implicit_residual(case, parameters)
    _state, switch_times, _events, final_joint, thrust_time, arcs = implicit_propagate(
        case, parameters
    )
    arc_records = [
        {
            "start": float(arc.start),
            "end": float(arc.end),
            "throttle": float(arc.throttle),
            "phi_min": float(
                np.min(
                    [
                        switch_function(arc.joint[:, i], case.stage.kappa)
                        for i in range(arc.joint.shape[1])
                    ]
                )
            ),
            "phi_max": float(
                np.max(
                    [
                        switch_function(arc.joint[:, i], case.stage.kappa)
                        for i in range(arc.joint.shape[1])
                    ]
                )
            ),
        }
        for arc in arcs
    ]
    return (
        parameters,
        result,
        residual,
        [float(t) for t in switch_times],
        float(thrust_time),
        float(1.0 - final_joint[3]),
        arc_records,
        final_joint[:4].tolist(),
    )


def solve(case: Case) -> SolverOutcome:
    started = time.monotonic()
    try:
        sigmoid_parameters, eps_stop, history, sigmoid_nfev, stop_reason = (
            sigmoid_continuation(case)
        )
    except (FloatingPointError, ValueError, ZeroDivisionError) as error:
        return SolverOutcome(
            case_name=case.name,
            success=False,
            message=f"sigmoid continuation failed: {error}",
            phase="sigmoid",
            parameters=[],
            residual=[],
            switch_times=[],
            thrust_time=float("nan"),
            fuel_fraction=float("nan"),
            final_state=[],
            arcs=[],
            eps_stop=EPS_START,
            sigmoid_nfev=0,
            polish_nfev=0,
            eps_history=[],
            wall_seconds=time.monotonic() - started,
        )
    try:
        (
            polish_parameters,
            polish_result,
            polish_residual,
            switch_times,
            thrust_time,
            fuel,
            arcs,
            final_state,
        ) = sharp_polish(case, sigmoid_parameters)
        polish_nfev = int(polish_result.nfev)
        polish_residual_norm = norm(polish_residual)
        success = bool(
            polish_result.success and polish_residual_norm < RESIDUAL_SUCCESS
        )
        message = (
            f"sharp polish residual {polish_residual_norm:.2e} "
            f"(sigmoid eps stopped at {eps_stop:.2e}, {stop_reason})"
        )
        if not success:
            message = (
                f"sharp polish did not fully converge: {polish_result.message} "
                f"(residual {polish_residual_norm:.2e}; sigmoid eps stopped at "
                f"{eps_stop:.2e}, {stop_reason})"
            )
        return SolverOutcome(
            case_name=case.name,
            success=success,
            message=message,
            phase="sharp",
            parameters=polish_parameters.tolist(),
            residual=polish_residual.tolist(),
            switch_times=switch_times,
            thrust_time=thrust_time,
            fuel_fraction=fuel,
            final_state=final_state,
            arcs=arcs,
            eps_stop=eps_stop,
            sigmoid_nfev=sigmoid_nfev,
            polish_nfev=polish_nfev,
            eps_history=history,
            wall_seconds=time.monotonic() - started,
            reference=REFERENCES.get(case.name, {}),
        )
    except (FloatingPointError, ValueError, ZeroDivisionError) as error:
        sigmoid_residual = sp.sigmoid_residual(case, sigmoid_parameters, eps_stop)
        return SolverOutcome(
            case_name=case.name,
            success=False,
            message=f"sharp polish failed ({error}); keeping sigmoid result",
            phase="sigmoid",
            parameters=sigmoid_parameters.tolist(),
            residual=sigmoid_residual.tolist(),
            switch_times=[],
            thrust_time=float("nan"),
            fuel_fraction=float("nan"),
            final_state=[],
            arcs=[],
            eps_stop=eps_stop,
            sigmoid_nfev=sigmoid_nfev,
            polish_nfev=0,
            eps_history=history,
            wall_seconds=time.monotonic() - started,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=str, action="append", default=[])
    parser.add_argument("--output", type=str, default="")
    arguments = parser.parse_args()

    cases = sp.kerbin_cases()
    if arguments.case:
        by_name = {case.name: case for case in cases}
        cases = [by_name[name] for name in arguments.case if name in by_name]
        if not cases:
            raise SystemExit("no requested case matched a Kerbin case")

    outcomes = [solve(case) for case in cases]
    for outcome in outcomes:
        status = "OK  " if outcome.success else "FAIL"
        final_time = outcome.parameters[1] if outcome.parameters else float("nan")
        print(
            f"{outcome.case_name:28s} {status} phase={outcome.phase:7s} "
            f"fuel={outcome.fuel_fraction:.9f} "
            f"res={norm(np.asarray(outcome.residual)):.2e} "
            f"tf={final_time:.9f} switches={len(outcome.switch_times)} "
            f"eps_stop={outcome.eps_stop:.2e} "
            f"nfev={outcome.sigmoid_nfev + outcome.polish_nfev} "
            f"[{outcome.wall_seconds:.1f}s]"
        )
        print(f"    {outcome.message}")
    if arguments.output:
        with open(arguments.output, "w", encoding="utf-8") as output_file:
            json.dump([o.to_dict() for o in outcomes], output_file, indent=2)
            output_file.write("\n")


if __name__ == "__main__":
    main()
