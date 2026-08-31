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
     is bang-bang (q_min < 0.01 and q_max > 0.99) or the `EPS_HANDOFF`
     (1e-3) or epsilon floor is reached. This replaces the fixed schedule and
     adapts automatically to each case: healthy cases stop early, and
     near-singular cases hand off to the sharp solver instead of grinding
     down to a tiny epsilon.
2. Sharp polish. The structure-free hard-switch shooter
   (`single_stage_research.implicit_propagate` with the same four shooting
   variables) is seeded with the sigmoid parameters. It re-derives the
   burn/coast partition from the switching function and returns exact
   event times, so a smeared sigmoid endpoint still yields the correct
   structure. Its fourth residual is S(tf)=0 (the terminal switching
   function) instead of the Hamiltonian transversality: at the target the
   two agree, but S(tf)=0 also pins the final time to the end of thrust
   instead of admitting a fuel-neutral trailing coast. The polish runs `lm`
   first and skips the `trf` pass when `lm` gets within
   `POLISH_SKIP_TRF_RES` of a root; otherwise `trf` runs and the
   smaller-residual result is kept.

Solver details:
- All solves use finite differences. No analytic Jacobian.
- The sigmoid integrator tries DOP853 first and falls back to LSODA when
  the transition layer makes the explicit integrator fail.
- Accept/reject per step: a solver result is adopted only when it does not
  worsen the shooting residual at that epsilon.
- Fuel guard (C3): if the sharp polish lands on a higher-fuel branch than
  the sigmoid hand-off point (by more than `FUEL_GUARD_TOL`), the ladder is
  continued to the render criterion and the lower-fuel result is kept.

Run from the repository root with ``PYTHONPATH=experiments``::

    python3 experiments/sigmoid_primer_review/production_solver.py

Use ``--case NAME`` to run a single case, ``--output FILE`` to write the
full JSON record, and ``--offline`` for the oracle-generation mode (accuracy
over speed: bigger evaluation budgets, tighter solver and integration
tolerances, no early hand-off, and repeated polish passes).
"""

from __future__ import annotations

import argparse
import json
import math
import os
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
EPS_HANDOFF = 1e-3
FUEL_GUARD_TOL = 5e-4
MAX_EPS_STEPS = 40
RENDER_Q_MIN = 0.01
RENDER_Q_MAX = 0.99
RESIDUAL_SUCCESS = 2e-6
SIGMOID_NFEV = 300
POLISH_NFEV = 3000
POLISH_SKIP_TRF_RES = 1e-4
SIGMOID_TOLERANCES = {"ftol": 2e-10, "xtol": 2e-10, "gtol": 2e-10}
POLISH_TOLERANCES = {"ftol": 2e-12, "xtol": 2e-12, "gtol": 2e-12}

# Offline (oracle-generation) settings: accuracy over speed.
SIGMOID_NFEV_OFFLINE = 2000
POLISH_NFEV_OFFLINE = 10000
POLISH_PASSES_OFFLINE = 3
RTOL_OFFLINE = 1e-12
ATOL_OFFLINE = 1e-14
OFFLINE_TOLERANCES = {"ftol": 1e-14, "xtol": 1e-14, "gtol": 1e-14}


@dataclass(frozen=True)
class Settings:
    eps_handoff: float
    sigmoid_nfev: int
    polish_nfev: int
    polish_passes: int
    rtol: float
    atol: float
    sigmoid_tolerances: dict[str, float]
    polish_tolerances: dict[str, float]


ONLINE = Settings(
    eps_handoff=EPS_HANDOFF,
    sigmoid_nfev=SIGMOID_NFEV,
    polish_nfev=POLISH_NFEV,
    polish_passes=1,
    rtol=sp.RTOL,
    atol=sp.ATOL,
    sigmoid_tolerances=SIGMOID_TOLERANCES,
    polish_tolerances=POLISH_TOLERANCES,
)

OFFLINE = Settings(
    eps_handoff=EPS_FLOOR,
    sigmoid_nfev=SIGMOID_NFEV_OFFLINE,
    polish_nfev=POLISH_NFEV_OFFLINE,
    polish_passes=POLISH_PASSES_OFFLINE,
    rtol=RTOL_OFFLINE,
    atol=ATOL_OFFLINE,
    sigmoid_tolerances=OFFLINE_TOLERANCES,
    polish_tolerances=OFFLINE_TOLERANCES,
)

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
    user_cpu_seconds: float
    system_cpu_seconds: float
    polish_wall_seconds: float
    polish_cpu_seconds: float
    polish_lm_nfev: int
    polish_trf_nfev: int
    polish_method: str
    mode: str
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
            "user_cpu_seconds": self.user_cpu_seconds,
            "system_cpu_seconds": self.system_cpu_seconds,
            "polish_wall_seconds": self.polish_wall_seconds,
            "polish_cpu_seconds": self.polish_cpu_seconds,
            "polish_lm_nfev": self.polish_lm_nfev,
            "polish_trf_nfev": self.polish_trf_nfev,
            "polish_method": self.polish_method,
            "mode": self.mode,
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
    case: Case,
    parameters: Array,
    epsilon: float,
    *,
    dense_output: bool = False,
    rtol: float = sp.RTOL,
    atol: float = sp.ATOL,
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
                rtol=rtol,
                atol=atol,
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


def rendered_throttle(
    case: Case,
    parameters: Array,
    epsilon: float,
    *,
    rtol: float = sp.RTOL,
    atol: float = sp.ATOL,
) -> ThrottleProfile:
    solution = integrate_sigmoid_robust(
        case, parameters, epsilon, dense_output=True, rtol=rtol, atol=atol
    )
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


def sigmoid_residual_tight(
    case: Case, parameters: Array, epsilon: float, rtol: float, atol: float
) -> Array:
    """Sigmoid shooting residual with selectable integration tolerances."""
    try:
        solution = integrate_sigmoid_robust(
            case, parameters, epsilon, rtol=rtol, atol=atol
        )
        final = np.asarray(solution.y[:, -1], dtype=float)
        return np.concatenate(
            (
                sp.terminal_residual(final[:4]),
                [sp.switching_function(final, case.stage.kappa)],
            )
        )
    except (FloatingPointError, ValueError, ZeroDivisionError):
        return np.full(4, 1e3)


def sigmoid_solve(
    case: Case,
    seed: Array,
    epsilon: float,
    method: Literal["trf", "lm"],
    max_nfev: int,
    tolerances: dict[str, float],
    rtol: float,
    atol: float,
) -> tuple[Array, Any, float]:
    kwargs: dict[str, Any] = {
        "fun": lambda p: sigmoid_residual_tight(case, p, epsilon, rtol, atol),
        "x0": seed,
        "method": method,
        "x_scale": "jac",
        "max_nfev": max_nfev,
        **tolerances,
    }
    if method != "lm":
        kwargs["bounds"] = sigmoid_bounds(case)
    result = least_squares(**kwargs)
    parameters = np.asarray(result.x, dtype=float)
    residual = sigmoid_residual_tight(case, parameters, epsilon, rtol, atol)
    return parameters, result, float(np.linalg.norm(residual))


def render_ok(profile: ThrottleProfile) -> bool:
    return bool(
        float(profile["q_min"]) < RENDER_Q_MIN
        and float(profile["q_max"]) > RENDER_Q_MAX
    )


def sigmoid_continuation(
    case: Case,
    *,
    stop_eps: float = EPS_FLOOR,
    seed: Array | None = None,
    start_eps: float = EPS_START,
    max_nfev: int = SIGMOID_NFEV,
    tolerances: dict[str, float] | None = None,
    rtol: float = sp.RTOL,
    atol: float = sp.ATOL,
) -> tuple[Array, float, list[dict[str, float]], int, str, float]:
    if tolerances is None:
        tolerances = SIGMOID_TOLERANCES
    lower, upper = sigmoid_bounds(case)
    if seed is None:
        try:
            seed = sp.initial_sigmoid_guess(case)
        except (FloatingPointError, ValueError, ZeroDivisionError):
            seed = np.array(
                [
                    math.pi / 2.0,
                    max(case.time_to_apoapsis, 1e-3),
                    0.0,
                    -case.stage.kappa,
                ],
                dtype=float,
            )
    parameters: Array = np.asarray(seed, dtype=float).copy()
    epsilon = start_eps
    total_nfev = 0
    history: list[dict[str, float]] = []

    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    parameters, result, residual = sigmoid_solve(
        case, parameters, epsilon, "trf", max_nfev, tolerances, rtol, atol
    )
    first_wall = time.perf_counter() - wall_start
    first_cpu = time.process_time() - cpu_start
    first_nfev = int(result.nfev)
    total_nfev += first_nfev
    profile = rendered_throttle(case, parameters, epsilon, rtol=rtol, atol=atol)
    history.append(
        {
            "epsilon": epsilon,
            "nfev": float(first_nfev),
            "total_nfev": float(first_nfev),
            "residual": residual,
            "q_min": float(profile["q_min"]),
            "q_max": float(profile["q_max"]),
            "solver": float(FIRST_STEP),
            "wall_s": first_wall,
            "cpu_s": first_cpu,
        }
    )

    while (
        not render_ok(profile) and epsilon > stop_eps and len(history) < MAX_EPS_STEPS
    ):
        epsilon /= EPS_FACTOR
        inherited_residual = norm(
            sigmoid_residual_tight(case, parameters, epsilon, rtol, atol)
        )
        chosen: tuple[Array, float, int, int] | None = None
        step_wall = 0.0
        step_cpu = 0.0
        step_nfev_total = 0

        wall_start = time.perf_counter()
        cpu_start = time.process_time()
        p_lm, result_lm, res_lm = sigmoid_solve(
            case, parameters, epsilon, "lm", max_nfev, tolerances, rtol, atol
        )
        step_wall += time.perf_counter() - wall_start
        step_cpu += time.process_time() - cpu_start
        step_nfev_total += int(result_lm.nfev)
        total_nfev += int(result_lm.nfev)
        if in_bounds(p_lm, lower, upper) and res_lm <= inherited_residual * 1.01:
            chosen = (p_lm, res_lm, int(result_lm.nfev), LM_STEP)
        else:
            wall_start = time.perf_counter()
            cpu_start = time.process_time()
            p_trf, result_trf, res_trf = sigmoid_solve(
                case, parameters, epsilon, "trf", max_nfev, tolerances, rtol, atol
            )
            step_wall += time.perf_counter() - wall_start
            step_cpu += time.process_time() - cpu_start
            step_nfev_total += int(result_trf.nfev)
            total_nfev += int(result_trf.nfev)
            if in_bounds(p_trf, lower, upper) and res_trf <= inherited_residual * 1.01:
                chosen = (p_trf, res_trf, int(result_trf.nfev), TRF_STEP)

        if chosen is None:
            profile = rendered_throttle(case, parameters, epsilon, rtol=rtol, atol=atol)
            history.append(
                {
                    "epsilon": epsilon,
                    "nfev": 0.0,
                    "total_nfev": float(step_nfev_total),
                    "residual": inherited_residual,
                    "q_min": float(profile["q_min"]),
                    "q_max": float(profile["q_max"]),
                    "solver": float(INHERITED),
                    "wall_s": step_wall,
                    "cpu_s": step_cpu,
                }
            )
            continue
        parameters, residual, step_nfev, marker = chosen
        profile = rendered_throttle(case, parameters, epsilon, rtol=rtol, atol=atol)
        history.append(
            {
                "epsilon": epsilon,
                "nfev": float(step_nfev),
                "total_nfev": float(step_nfev_total),
                "residual": residual,
                "q_min": float(profile["q_min"]),
                "q_max": float(profile["q_max"]),
                "solver": float(marker),
                "wall_s": step_wall,
                "cpu_s": step_cpu,
            }
        )

    if render_ok(profile):
        stop_reason = "render-ok"
    elif epsilon <= stop_eps:
        stop_reason = "handoff" if stop_eps > EPS_FLOOR else "epsilon-floor"
    else:
        stop_reason = "max-steps"
    return (
        parameters,
        epsilon,
        history,
        total_nfev,
        stop_reason,
        float(profile["fuel_fraction"]),
    )


def sharp_residual(case: Case, parameters: Array) -> Array:
    """Sharp shooting residual: terminal orbit conditions plus S(tf)=0.

    At the target the non-thrust part of the Hamiltonian vanishes, so
    S(tf)=0 is equivalent to the free-final-time transversality H(tf)=0 when
    the trajectory ends with thrust on. Unlike the Hamiltonian condition it
    also pins tf to the end of thrust: H(tf)=0 is satisfied trivially on any
    trailing coast (q=0) of the target orbit, which is the wandering-tf
    degeneracy; S(tf)=0 is not, so the reported final time is exactly the
    last switching-function crossing.
    """
    try:
        _state, _switch_times, _events, final_joint, _thrust, _arcs = (
            implicit_propagate(case, parameters)
        )
        return np.array(
            [
                final_joint[0] - 1.0,
                final_joint[1],
                final_joint[2] - 1.0,
                switch_function(final_joint, case.stage.kappa),
            ],
            dtype=float,
        )
    except (FloatingPointError, ValueError, ZeroDivisionError):
        return np.full(4, 1e3)


def sharp_solve(
    case: Case,
    seed: Array,
    method: Literal["trf", "lm"],
    max_nfev: int,
    tolerances: dict[str, float],
) -> tuple[Array, Any, float]:
    kwargs: dict[str, Any] = {
        "fun": lambda p: sharp_residual(case, p),
        "x0": seed,
        "method": method,
        "x_scale": "jac",
        "max_nfev": max_nfev,
        **tolerances,
    }
    if method != "lm":
        kwargs["bounds"] = sigmoid_bounds(case)
    result = least_squares(**kwargs)
    parameters = np.asarray(result.x, dtype=float)
    residual = sharp_residual(case, parameters)
    return parameters, result, float(norm(residual))


def sharp_polish(
    case: Case,
    seed: Array,
    max_nfev: int = POLISH_NFEV,
    tolerances: dict[str, float] | None = None,
) -> tuple[
    Array,
    Any,
    Array,
    list[float],
    float,
    float,
    list[dict[str, float]],
    list[float],
    int,
    int,
    str,
]:
    """Polish with lm first and trf only when lm does not get close enough.

    `lm` is usually the better and cheaper choice (review section 3.3). When
    its final residual is below `POLISH_SKIP_TRF_RES` the `trf` pass is
    skipped entirely; otherwise `trf` runs too (it recovers the cases where
    `lm` stops in a spurious local minimum of the squared residual, e.g.
    `kerbin-coast-first-example`) and the smaller-residual result is kept.
    """
    if tolerances is None:
        tolerances = POLISH_TOLERANCES
    lower, upper = sigmoid_bounds(case)
    p_lm, result_lm, res_lm = sharp_solve(case, seed, "lm", max_nfev, tolerances)
    trf_nfev = 0
    lm_acceptable = in_bounds(p_lm, lower, upper) and res_lm < POLISH_SKIP_TRF_RES
    if lm_acceptable:
        parameters, result, method = p_lm, result_lm, "lm"
    else:
        candidates: list[tuple[float, Array, Any, str]] = []
        if in_bounds(p_lm, lower, upper):
            candidates.append((res_lm, p_lm, result_lm, "lm"))
        p_trf, result_trf, res_trf = sharp_solve(
            case, seed, "trf", max_nfev, tolerances
        )
        trf_nfev = int(result_trf.nfev)
        candidates.append((res_trf, p_trf, result_trf, "trf"))
        _, parameters, result, method = min(candidates, key=lambda item: item[0])
    result.message = f"{method}: {result.message}"
    residual = sharp_residual(case, parameters)
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
        int(result_lm.nfev),
        trf_nfev,
        method,
    )


@dataclass
class PolishOutcome:
    parameters: Array
    result: Any
    residual: Array
    residual_norm: float
    switch_times: list[float]
    thrust_time: float
    fuel_fraction: float
    final_state: list[float]
    arcs: list[dict[str, float]]
    lm_nfev: int
    trf_nfev: int
    method: str
    wall_seconds: float
    cpu_seconds: float


def polish(case: Case, seed: Array, settings: Settings = ONLINE) -> PolishOutcome:
    """Run sharp-polish passes, re-seeding from the best result each time.

    The online mode runs a single pass; the offline mode runs several
    (POLISH_PASSES_OFFLINE) and stops early when a pass no longer improves
    the residual. The returned outcome carries the totals of all passes.
    """
    total_wall = 0.0
    total_cpu = 0.0
    lm_nfev_total = 0
    trf_nfev_total = 0
    best: PolishOutcome | None = None
    current_seed = seed
    for _ in range(settings.polish_passes):
        wall_start = time.perf_counter()
        cpu_start = time.process_time()
        (
            parameters,
            result,
            residual,
            switch_times,
            thrust_time,
            fuel,
            arcs,
            final_state,
            lm_nfev,
            trf_nfev,
            method,
        ) = sharp_polish(
            case, current_seed, settings.polish_nfev, settings.polish_tolerances
        )
        total_wall += time.perf_counter() - wall_start
        total_cpu += time.process_time() - cpu_start
        lm_nfev_total += lm_nfev
        trf_nfev_total += trf_nfev
        candidate = PolishOutcome(
            parameters=parameters,
            result=result,
            residual=residual,
            residual_norm=norm(residual),
            switch_times=switch_times,
            thrust_time=thrust_time,
            fuel_fraction=fuel,
            final_state=final_state,
            arcs=arcs,
            lm_nfev=lm_nfev,
            trf_nfev=trf_nfev,
            method=method,
            wall_seconds=total_wall,
            cpu_seconds=total_cpu,
        )
        if best is None or candidate.residual_norm < best.residual_norm:
            best = candidate
            current_seed = candidate.parameters
        else:
            break
    if best is None:
        raise ValueError("sharp polish produced no result")
    best.lm_nfev = lm_nfev_total
    best.trf_nfev = trf_nfev_total
    best.wall_seconds = total_wall
    best.cpu_seconds = total_cpu
    return best


def solve(case: Case, offline: bool = False) -> SolverOutcome:
    settings = OFFLINE if offline else ONLINE
    mode_name = "offline" if offline else "online"
    wall_started = time.monotonic()
    times_started = os.times()

    def cpu_accounting() -> tuple[float, float, float]:
        wall = time.monotonic() - wall_started
        times_ended = os.times()
        return (
            wall,
            times_ended.user - times_started.user,
            times_ended.system - times_started.system,
        )

    def failure_outcome(
        message: str,
        *,
        phase: str,
        parameters: list[float],
        residual: list[float],
        eps_stop: float,
        sigmoid_nfev: int,
        history: list[dict[str, float]],
    ) -> SolverOutcome:
        wall, user_cpu, system_cpu = cpu_accounting()
        return SolverOutcome(
            case_name=case.name,
            success=False,
            message=message,
            phase=phase,
            parameters=parameters,
            residual=residual,
            switch_times=[],
            thrust_time=float("nan"),
            fuel_fraction=float("nan"),
            final_state=[],
            arcs=[],
            eps_stop=eps_stop,
            sigmoid_nfev=sigmoid_nfev,
            polish_nfev=0,
            eps_history=history,
            wall_seconds=wall,
            user_cpu_seconds=user_cpu,
            system_cpu_seconds=system_cpu,
            polish_wall_seconds=0.0,
            polish_cpu_seconds=0.0,
            polish_lm_nfev=0,
            polish_trf_nfev=0,
            polish_method="none",
            mode=mode_name,
        )

    try:
        handoff_params, eps_stop, history, sigmoid_nfev, stop_reason, endpoint_fuel = (
            sigmoid_continuation(
                case,
                stop_eps=settings.eps_handoff,
                max_nfev=settings.sigmoid_nfev,
                tolerances=settings.sigmoid_tolerances,
                rtol=settings.rtol,
                atol=settings.atol,
            )
        )
    except (FloatingPointError, ValueError, ZeroDivisionError) as error:
        return failure_outcome(
            f"sigmoid continuation failed: {error}",
            phase="sigmoid",
            parameters=[],
            residual=[],
            eps_stop=EPS_START,
            sigmoid_nfev=0,
            history=[],
        )

    polish_lm_nfev = 0
    polish_trf_nfev = 0
    polish_wall_seconds = 0.0
    polish_cpu_seconds = 0.0
    guard_fell_back = False
    try:
        chosen = polish(case, handoff_params, settings)
        polish_lm_nfev += chosen.lm_nfev
        polish_trf_nfev += chosen.trf_nfev
        polish_wall_seconds += chosen.wall_seconds
        polish_cpu_seconds += chosen.cpu_seconds

        # C3 guard: if the sharp polish landed on a higher-fuel branch than
        # the sigmoid hand-off point, continue the ladder to the render
        # criterion and polish again, then keep the lower-fuel result.
        if chosen.fuel_fraction > endpoint_fuel + FUEL_GUARD_TOL:
            guard_fell_back = True
            deep_params, deep_eps, deep_history, deep_nfev, deep_reason, _ = (
                sigmoid_continuation(
                    case,
                    stop_eps=EPS_FLOOR,
                    seed=handoff_params,
                    start_eps=eps_stop / EPS_FACTOR,
                    max_nfev=settings.sigmoid_nfev,
                    tolerances=settings.sigmoid_tolerances,
                    rtol=settings.rtol,
                    atol=settings.atol,
                )
            )
            deep = polish(case, deep_params, settings)
            polish_lm_nfev += deep.lm_nfev
            polish_trf_nfev += deep.trf_nfev
            polish_wall_seconds += deep.wall_seconds
            polish_cpu_seconds += deep.cpu_seconds
            sigmoid_nfev += deep_nfev
            history = history + deep_history
            if deep.fuel_fraction < chosen.fuel_fraction:
                chosen = deep
            eps_stop = deep_eps
            stop_reason = deep_reason

        polish_nfev = polish_lm_nfev + polish_trf_nfev
        success = bool(
            chosen.result.success and chosen.residual_norm < RESIDUAL_SUCCESS
        )
        guard_note = "; guard fell back to full ladder" if guard_fell_back else ""
        message = (
            f"sharp polish residual {chosen.residual_norm:.2e} "
            f"(sigmoid eps stopped at {eps_stop:.2e}, {stop_reason}{guard_note})"
        )
        if not success:
            message = (
                f"sharp polish did not fully converge: {chosen.result.message} "
                f"(residual {chosen.residual_norm:.2e}; sigmoid eps stopped at "
                f"{eps_stop:.2e}, {stop_reason}{guard_note})"
            )
        wall, user_cpu, system_cpu = cpu_accounting()
        return SolverOutcome(
            case_name=case.name,
            success=success,
            message=message,
            phase="sharp",
            parameters=chosen.parameters.tolist(),
            residual=chosen.residual.tolist(),
            switch_times=chosen.switch_times,
            thrust_time=chosen.thrust_time,
            fuel_fraction=chosen.fuel_fraction,
            final_state=chosen.final_state,
            arcs=chosen.arcs,
            eps_stop=eps_stop,
            sigmoid_nfev=sigmoid_nfev,
            polish_nfev=polish_nfev,
            eps_history=history,
            wall_seconds=wall,
            user_cpu_seconds=user_cpu,
            system_cpu_seconds=system_cpu,
            polish_wall_seconds=polish_wall_seconds,
            polish_cpu_seconds=polish_cpu_seconds,
            polish_lm_nfev=polish_lm_nfev,
            polish_trf_nfev=polish_trf_nfev,
            polish_method=chosen.method,
            mode=mode_name,
            reference=REFERENCES.get(case.name, {}),
        )
    except (FloatingPointError, ValueError, ZeroDivisionError) as error:
        sigmoid_residual = sp.sigmoid_residual(case, handoff_params, eps_stop)
        return failure_outcome(
            f"sharp polish failed ({error}); keeping sigmoid result",
            phase="sigmoid",
            parameters=handoff_params.tolist(),
            residual=sigmoid_residual.tolist(),
            eps_stop=eps_stop,
            sigmoid_nfev=sigmoid_nfev,
            history=history,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=str, action="append", default=[])
    parser.add_argument("--output", type=str, default="")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="oracle mode: accuracy over speed (bigger budgets, tighter "
        "tolerances, no early hand-off, re-polish passes)",
    )
    arguments = parser.parse_args()

    cases = sp.kerbin_cases()
    if arguments.case:
        by_name = {case.name: case for case in cases}
        cases = [by_name[name] for name in arguments.case if name in by_name]
        if not cases:
            raise SystemExit("no requested case matched a Kerbin case")

    outcomes = [solve(case, offline=arguments.offline) for case in cases]
    for outcome in outcomes:
        status = "OK  " if outcome.success else "FAIL"
        final_time = outcome.parameters[1] if outcome.parameters else float("nan")
        print(
            f"{outcome.case_name:28s} {status} [{outcome.mode:7s}] "
            f"fuel={outcome.fuel_fraction:.9f} "
            f"res={norm(np.asarray(outcome.residual)):.2e} "
            f"tf={final_time:.9f} switches={len(outcome.switch_times)} "
            f"eps_stop={outcome.eps_stop:.2e} "
            f"nfev={outcome.sigmoid_nfev + outcome.polish_nfev} "
            f"(sig={outcome.sigmoid_nfev} pol_lm={outcome.polish_lm_nfev} "
            f"pol_trf={outcome.polish_trf_nfev}->{outcome.polish_method}) "
            f"[{outcome.wall_seconds:.1f}s wall / {outcome.user_cpu_seconds:.1f}s user "
            f"/ {outcome.system_cpu_seconds:.1f}s sys]"
        )
        print(f"    {outcome.message}")
    if arguments.output:
        with open(arguments.output, "w", encoding="utf-8") as output_file:
            json.dump([o.to_dict() for o in outcomes], output_file, indent=2)
            output_file.write("\n")


if __name__ == "__main__":
    main()
