#!/usr/bin/env python3
"""Generalized on/off/on primer shooting with scheduled staging.

The powered arcs use active-burn durations.  A stage boundary can therefore
split either powered arc.  The boundary applies a fixed dry-mass jettison and
inserts a fixed ballistic staging gap before the next stage resumes thrust.

Run from the repository root with::

    PYTHONPATH=experiments python3 experiments/multi_stage_primer.py
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import NonlinearConstraint, least_squares, minimize
from staged_two_arc_research import (
    NormalizedStage,
    PhysicalStage,
    PrimerArc,
    hamiltonian,
    integrate_primer_arc,
    integrate_state,
    physical_stage_to_normalized,
    switch_function,
)

Array = NDArray[np.float64]


@dataclass(frozen=True)
class MultiStageCase:
    name: str
    mu: float
    target_radius: float
    velocity_scale: float
    time_scale: float
    initial_state: Array
    stages: tuple[NormalizedStage, ...]
    staging_gap: float
    staging_gap_seconds: float
    initial_time_to_apoapsis_seconds: float
    final_time_cap: float


@dataclass
class BurnTracker:
    stage_index: int = 0
    active_used: float = 0.0
    propellant_used: float = 0.0
    transition_count: int = 0
    staging_time: float = 0.0


@dataclass(frozen=True)
class PrimerJunction:
    source: str
    from_stage_index: int
    to_stage_index: int
    from_stage: str
    to_stage: str
    before_jump: Array
    after_jump: Array


@dataclass(frozen=True)
class BoundaryPoint:
    joint: Array
    stage: NormalizedStage
    stage_index: int


@dataclass(frozen=True)
class PrimerPropagation:
    final_joint: Array
    phases: tuple[PrimerArc, ...]
    junctions: tuple[PrimerJunction, ...]
    first_switch: BoundaryPoint
    coast_end: BoundaryPoint
    total_time: float
    propellant_used: float
    transition_count: int


@dataclass(frozen=True)
class DirectPropagation:
    final_state: Array
    tracker: BurnTracker
    total_time: float
    transition_sources: tuple[str, ...]


@dataclass(frozen=True)
class DirectResult:
    success: bool
    message: str
    parameters: Array
    final_state: Array
    residual: Array
    propellant_used: float
    total_time: float
    angle_intervals: int
    attempts: int
    transition_source: str


@dataclass(frozen=True)
class PrimerResult:
    success: bool
    message: str
    parameters: Array
    final_joint: Array
    residual: Array
    propagation: PrimerPropagation
    attempts: int


def norm(vector: Array) -> float:
    return float(np.linalg.norm(vector))


def physical_fixture_stages(case_name: str) -> tuple[PhysicalStage, PhysicalStage]:
    if case_name == "kerbin-stage-data":
        return (
            PhysicalStage(
                name='LV-T45 "Swivel"',
                exhaust_velocity=3138.1279999999997,
                thrust=215_000.0,
                full_burn_time=59.0500960010656,
                start_mass=13_885.650390625,
            ),
            PhysicalStage(
                name='LV-909 "Terrier"',
                exhaust_velocity=3383.2942499999995,
                thrust=60_000.0,
                full_burn_time=112.77647255563578,
                start_mass=4_449.999407536325,
            ),
        )
    if case_name == "kerbin-first-example":
        return (
            PhysicalStage(
                name="Swivel",
                exhaust_velocity=320.0 * 9.80665,
                thrust=215_000.0,
                full_burn_time=46.95725973451462,
                start_mass=13_057.14453125,
            ),
            PhysicalStage(
                name="Terrier",
                exhaust_velocity=345.0 * 9.80665,
                thrust=60_000.0,
                full_burn_time=112.77647255563578,
                start_mass=4_450.0,
            ),
        )
    raise ValueError(f"unknown case {case_name}")


def make_case(case_name: str) -> MultiStageCase:
    mu = 3.5316e12
    target_radius = 680_000.0
    velocity_scale = math.sqrt(mu / target_radius)
    time_scale = target_radius / velocity_scale
    if case_name == "kerbin-stage-data":
        position = np.array(
            [424370.58766631, -1093.08696926, -470992.64951719], dtype=float
        )
        velocity = np.array([723.81414935, -1.20334290, -122.60883836], dtype=float)
        time_to_apoapsis = 72.12194913376851
    elif case_name == "kerbin-first-example":
        position = np.array(
            [428392.15435586, -1053.61873734, -455905.93323801], dtype=float
        )
        velocity = np.array(
            [1.03031015e03, -9.32270447e-01, -1.19588146e02], dtype=float
        )
        time_to_apoapsis = 103.31401749403551
    else:
        raise ValueError(f"unknown case {case_name}")

    radius = norm(position)
    radial_hat = position / radius
    radial_velocity = float(np.dot(velocity, radial_hat))
    tangential_vector = velocity - radial_velocity * radial_hat
    tangential_hat = tangential_vector / norm(tangential_vector)
    tangential_velocity = float(np.dot(velocity, tangential_hat))
    initial_state = np.array(
        [
            radius / target_radius,
            radial_velocity / velocity_scale,
            tangential_velocity / velocity_scale,
            1.0,
        ],
        dtype=float,
    )

    physical_stages = physical_fixture_stages(case_name)
    mass_scale = physical_stages[0].start_mass
    stages = tuple(
        physical_stage_to_normalized(
            stage,
            mass_scale,
            velocity_scale,
            target_radius,
            time_scale,
        )
        for stage in physical_stages
    )
    for previous, following in pairwise(stages):
        if not following.start_mass < previous.end_mass:
            raise ValueError("stage start mass must be below previous end mass")
    energy = (
        0.5 * (initial_state[1] ** 2 + initial_state[2] ** 2) - 1.0 / initial_state[0]
    )
    semi_major_axis = -1.0 / (2.0 * energy)
    final_time_cap = float(
        time_to_apoapsis / time_scale + 0.75 * (2.0 * math.pi * semi_major_axis**1.5)
    )
    return MultiStageCase(
        name=case_name,
        mu=mu,
        target_radius=target_radius,
        velocity_scale=velocity_scale,
        time_scale=time_scale,
        initial_state=initial_state,
        stages=stages,
        staging_gap=2.0 / time_scale,
        staging_gap_seconds=2.0,
        initial_time_to_apoapsis_seconds=time_to_apoapsis,
        final_time_cap=final_time_cap,
    )


def jettison_mass(case: MultiStageCase, stage_index: int) -> float:
    current = case.stages[stage_index]
    following = case.stages[stage_index + 1]
    return current.end_mass - following.start_mass


def record_transition(
    case: MultiStageCase,
    tracker: BurnTracker,
    state: Array,
    source: str,
) -> tuple[Array, PrimerJunction]:
    stage_index = tracker.stage_index
    if stage_index + 1 >= len(case.stages):
        raise ValueError("no stage remains after the exhausted stage")
    current = case.stages[stage_index]
    following = case.stages[stage_index + 1]
    before_jump = state.copy()
    after_jump = state.copy()
    after_jump[3] -= jettison_mass(case, stage_index)
    if abs(after_jump[3] - following.start_mass) > 2e-9:
        raise ValueError("jettison does not reach the following stage mass")
    tracker.stage_index += 1
    tracker.active_used = 0.0
    tracker.transition_count += 1
    tracker.staging_time += case.staging_gap
    return after_jump, PrimerJunction(
        source=source,
        from_stage_index=stage_index,
        to_stage_index=stage_index + 1,
        from_stage=current.name,
        to_stage=following.name,
        before_jump=before_jump,
        after_jump=after_jump.copy(),
    )


def consume_primer_burn(
    case: MultiStageCase,
    joint: Array,
    duration: float,
    tracker: BurnTracker,
    phases: list[PrimerArc],
    junctions: list[PrimerJunction],
    source: str,
    transition_at_endpoint: bool,
) -> tuple[Array, BoundaryPoint]:
    remaining = duration
    endpoint: BoundaryPoint | None = None
    tolerance = 1e-11
    while remaining > tolerance:
        stage = case.stages[tracker.stage_index]
        capacity = stage.full_burn_time - tracker.active_used
        if capacity < -tolerance:
            raise ValueError("stage active-burn budget became negative")
        if capacity <= tolerance:
            if tracker.stage_index + 1 >= len(case.stages):
                raise ValueError("powered schedule exceeds final-stage capacity")
            joint, junction = record_transition(case, tracker, joint, source)
            junctions.append(junction)
            continue
        segment_duration = min(remaining, capacity)
        joint, phase = integrate_primer_arc(
            joint,
            segment_duration,
            stage,
            1.0,
            f"{source}: {stage.name}",
        )
        phases.append(phase)
        tracker.active_used += segment_duration
        tracker.propellant_used += stage.gamma / stage.kappa * segment_duration
        remaining -= segment_duration
        if remaining <= tolerance:
            remaining = 0.0
        capacity_exhausted = stage.full_burn_time - tracker.active_used <= tolerance
        if not capacity_exhausted:
            endpoint = BoundaryPoint(joint.copy(), stage, tracker.stage_index)
            break
        if remaining > tolerance:
            if tracker.stage_index + 1 >= len(case.stages):
                raise ValueError("powered schedule exceeds final-stage capacity")
            joint, junction = record_transition(case, tracker, joint, source)
            junctions.append(junction)
            gap_stage = case.stages[tracker.stage_index - 1]
            joint, gap_phase = integrate_primer_arc(
                joint,
                case.staging_gap,
                gap_stage,
                0.0,
                f"staging gap after {gap_stage.name}",
            )
            phases.append(gap_phase)
            endpoint = None
            continue
        if transition_at_endpoint:
            if tracker.stage_index + 1 >= len(case.stages):
                endpoint = BoundaryPoint(joint.copy(), stage, tracker.stage_index)
                break
            endpoint = BoundaryPoint(joint.copy(), stage, tracker.stage_index)
            joint, junction = record_transition(case, tracker, joint, source)
            junctions.append(junction)
            joint, gap_phase = integrate_primer_arc(
                joint,
                case.staging_gap,
                stage,
                0.0,
                f"staging gap after {stage.name}",
            )
            phases.append(gap_phase)
        else:
            endpoint = BoundaryPoint(joint.copy(), stage, tracker.stage_index)
    if endpoint is None:
        endpoint = BoundaryPoint(
            joint.copy(), case.stages[tracker.stage_index], tracker.stage_index
        )
    return joint, endpoint


def consume_direct_burn(
    case: MultiStageCase,
    state: Array,
    duration: float,
    alpha: float,
    tracker: BurnTracker,
    transition_sources: list[str],
    source: str,
    transition_at_endpoint: bool,
) -> Array:
    remaining = duration
    tolerance = 1e-11
    while remaining > tolerance:
        stage = case.stages[tracker.stage_index]
        capacity = stage.full_burn_time - tracker.active_used
        if capacity < -tolerance:
            raise ValueError("stage active-burn budget became negative")
        if capacity <= tolerance:
            if tracker.stage_index + 1 >= len(case.stages):
                raise ValueError("powered schedule exceeds final-stage capacity")
            state, _junction = record_transition(case, tracker, state, source)
            state = integrate_state(state, case.staging_gap, stage, 0.0, 0.0)
            transition_sources.append(source)
            continue
        segment_duration = min(remaining, capacity)
        state = integrate_state(state, segment_duration, stage, 1.0, alpha)
        tracker.active_used += segment_duration
        tracker.propellant_used += stage.gamma / stage.kappa * segment_duration
        remaining -= segment_duration
        if remaining <= tolerance:
            remaining = 0.0
        capacity_exhausted = stage.full_burn_time - tracker.active_used <= tolerance
        if not capacity_exhausted:
            break
        if remaining > tolerance or transition_at_endpoint:
            if tracker.stage_index + 1 >= len(case.stages):
                if remaining > tolerance:
                    raise ValueError("powered schedule exceeds final-stage capacity")
                break
            state, _junction = record_transition(case, tracker, state, source)
            state = integrate_state(state, case.staging_gap, stage, 0.0, 0.0)
            transition_sources.append(source)
    return state


def propagate_primer(case: MultiStageCase, parameters: Array) -> PrimerPropagation:
    (
        alpha,
        burn_one,
        coast,
        burn_two,
        lambda_rho,
        lambda_mass,
    ) = parameters
    initial_joint = np.concatenate(
        (
            case.initial_state,
            [lambda_rho, math.cos(alpha), math.sin(alpha), lambda_mass],
        ),
    )
    tracker = BurnTracker()
    phases: list[PrimerArc] = []
    junctions: list[PrimerJunction] = []
    joint, first_switch = consume_primer_burn(
        case,
        initial_joint,
        float(burn_one),
        tracker,
        phases,
        junctions,
        "burn 1",
        transition_at_endpoint=True,
    )
    coast_stage = case.stages[tracker.stage_index]
    joint, coast_phase = integrate_primer_arc(
        joint,
        float(coast),
        coast_stage,
        0.0,
        "planned coast",
    )
    phases.append(coast_phase)
    coast_end = BoundaryPoint(joint.copy(), coast_stage, tracker.stage_index)
    joint, _second_endpoint = consume_primer_burn(
        case,
        joint,
        float(burn_two),
        tracker,
        phases,
        junctions,
        "burn 2",
        transition_at_endpoint=False,
    )
    return PrimerPropagation(
        final_joint=joint,
        phases=tuple(phases),
        junctions=tuple(junctions),
        first_switch=first_switch,
        coast_end=coast_end,
        total_time=sum(phase.duration for phase in phases),
        propellant_used=tracker.propellant_used,
        transition_count=tracker.transition_count,
    )


def primer_residual(
    case: MultiStageCase, parameters: Array, final_time_active: bool = False
) -> Array:
    try:
        propagation = propagate_primer(case, parameters)
        if propagation.transition_count != len(case.stages) - 1:
            return np.full(6, 1e3)
        final = propagation.final_joint
        final_stage = propagation.phases[-1].stage
        junction = propagation.junctions[0]
        lambda_burn = stage_event_clock_lambda(case, propagation)
        first_switch_value = propagation.first_switch.stage.gamma * switch_function(
            propagation.first_switch.joint, propagation.first_switch.stage
        ) - (
            lambda_burn
            if propagation.first_switch.stage_index == junction.from_stage_index
            else 0.0
        )
        coast_switch_value = propagation.coast_end.stage.gamma * switch_function(
            propagation.coast_end.joint, propagation.coast_end.stage
        ) - (
            lambda_burn
            if propagation.coast_end.stage_index == junction.from_stage_index
            else 0.0
        )
        residual = [
            final[0] - 1.0,
            final[1],
            final[2] - 1.0,
            first_switch_value,
            coast_switch_value,
        ]
        if final_time_active:
            residual.append(propagation.total_time - case.final_time_cap)
        else:
            residual.append(hamiltonian(final, final_stage, 1.0))
        return np.asarray(residual, dtype=float)
    except (FloatingPointError, IndexError, ValueError, ZeroDivisionError):
        return np.full(6, 1e3)


def stage_event_clock_lambda(
    case: MultiStageCase, propagation: PrimerPropagation
) -> float:
    junction = propagation.junctions[0]
    stage = case.stages[junction.from_stage_index]
    return float(
        hamiltonian(junction.after_jump, stage, 0.0)
        - hamiltonian(junction.before_jump, stage, 1.0)
    )


def direct_propagate(
    case: MultiStageCase,
    parameters: Array,
    angle_intervals: int,
) -> DirectPropagation:
    burn_one, coast, burn_two = (float(value) for value in parameters[:3])
    angles_one = parameters[3 : 3 + angle_intervals]
    angles_two = parameters[3 + angle_intervals :]
    tracker = BurnTracker()
    transition_sources: list[str] = []
    state = case.initial_state.copy()
    for angle in angles_one:
        state = consume_direct_burn(
            case,
            state,
            burn_one / angle_intervals,
            float(angle),
            tracker,
            transition_sources,
            "burn 1",
            transition_at_endpoint=True,
        )
    coast_stage = case.stages[tracker.stage_index]
    state = integrate_state(state, coast, coast_stage, 0.0, 0.0)
    for index, angle in enumerate(angles_two):
        state = consume_direct_burn(
            case,
            state,
            burn_two / angle_intervals,
            float(angle),
            tracker,
            transition_sources,
            "burn 2",
            transition_at_endpoint=index + 1 < angle_intervals,
        )
    return DirectPropagation(
        final_state=state,
        tracker=tracker,
        total_time=burn_one + coast + burn_two + tracker.staging_time,
        transition_sources=tuple(transition_sources),
    )


def direct_residual(
    case: MultiStageCase, parameters: Array, angle_intervals: int
) -> Array:
    try:
        propagation = direct_propagate(case, parameters, angle_intervals)
        if propagation.tracker.transition_count != len(case.stages) - 1:
            return np.full(3, 1e3)
        final = propagation.final_state
        return np.array([final[0] - 1.0, final[1], final[2] - 1.0])
    except (FloatingPointError, ValueError, ZeroDivisionError):
        return np.full(3, 1e3)


def direct_initial_guesses(case: MultiStageCase, angle_intervals: int) -> list[Array]:
    velocity_angle = math.atan2(case.initial_state[2], case.initial_state[1])
    first_patterns = (
        np.full(angle_intervals, velocity_angle),
        np.full(angle_intervals, math.pi / 2.0),
        np.linspace(velocity_angle, math.pi / 2.0, angle_intervals),
        np.linspace(1.0, 1.7, angle_intervals),
    )
    second_patterns = (
        np.full(angle_intervals, math.pi / 2.0),
        np.full(angle_intervals, 1.35),
        np.full(angle_intervals, 1.7),
        np.linspace(1.2, 1.75, angle_intervals),
    )
    first_capacity = case.stages[0].full_burn_time
    second_capacity = case.stages[1].full_burn_time
    coast_to_apoapsis = max(
        0.0,
        case.initial_time_to_apoapsis_seconds / case.time_scale - first_capacity,
    )
    guesses: list[Array] = []
    for index, first in enumerate(first_patterns):
        second = second_patterns[index]
        for first_fraction in (0.5, 0.8, 1.0):
            for second_fraction in (0.3, 0.5, 0.7):
                burn_one = first_fraction * first_capacity
                burn_two = second_fraction * second_capacity
                guesses.append(
                    np.concatenate(
                        (
                            [burn_one, coast_to_apoapsis, burn_two],
                            first,
                            second,
                        ),
                        dtype=float,
                    )
                )
    return guesses


def direct_objective(
    case: MultiStageCase, parameters: Array, angle_intervals: int
) -> float:
    try:
        propagation = direct_propagate(case, parameters, angle_intervals)
    except (FloatingPointError, ValueError, ZeroDivisionError):
        return 1e3
    if propagation.tracker.transition_count != len(case.stages) - 1:
        return 1e3
    return propagation.tracker.propellant_used


def solve_direct(
    case: MultiStageCase, angle_intervals: int, max_attempts: int
) -> DirectResult:
    lower = np.concatenate(
        (
            [1e-8, 0.0, 1e-8],
            np.full(2 * angle_intervals, -math.pi),
        ),
        dtype=float,
    )
    upper = np.concatenate(
        (
            [
                sum(stage.full_burn_time for stage in case.stages),
                case.final_time_cap,
                sum(stage.full_burn_time for stage in case.stages),
            ],
            np.full(2 * angle_intervals, math.pi),
        ),
        dtype=float,
    )
    attempts: list[tuple[Any, Array, DirectPropagation, Array]] = []
    for initial in direct_initial_guesses(case, angle_intervals)[:max_attempts]:

        def terminal_constraint(parameters: Array) -> Array:
            return direct_residual(case, parameters, angle_intervals)

        def time_constraint(parameters: Array) -> Array:
            try:
                return np.array(
                    [
                        case.final_time_cap
                        - direct_propagate(case, parameters, angle_intervals).total_time
                    ]
                )
            except (FloatingPointError, ValueError, ZeroDivisionError):
                return np.array([-1e3])

        def objective(parameters: Array) -> float:
            return direct_objective(case, parameters, angle_intervals)

        result = minimize(
            objective,
            initial,
            method="SLSQP",
            bounds=list(zip(lower, upper, strict=True)),
            constraints=(
                NonlinearConstraint(terminal_constraint, 0.0, 0.0),
                NonlinearConstraint(time_constraint, 0.0, np.inf),
            ),
            options={"ftol": 2e-11, "maxiter": 1000, "disp": False},
        )
        parameters = np.asarray(result.x, dtype=float)
        propagation = direct_propagate(case, parameters, angle_intervals)
        residual = direct_residual(case, parameters, angle_intervals)
        attempts.append((result, parameters, propagation, residual))
    accepted = [
        index
        for index, item in enumerate(attempts)
        if bool(attempts[index][0].success)
        and norm(attempts[index][3]) < 2e-6
        and attempts[index][2].tracker.transition_count == len(case.stages) - 1
        and attempts[index][2].total_time <= case.final_time_cap + 1e-8
    ]
    candidates = accepted or list(range(len(attempts)))
    chosen_index = min(
        candidates,
        key=lambda index: (
            attempts[index][2].tracker.propellant_used
            if index in accepted
            else norm(attempts[index][3])
        ),
    )
    result, parameters, propagation, residual = attempts[chosen_index]
    transition_source = (
        propagation.transition_sources[0] if propagation.transition_sources else "none"
    )
    return DirectResult(
        success=bool(
            result.success
            and norm(residual) < 2e-6
            and propagation.tracker.transition_count == len(case.stages) - 1
            and propagation.total_time <= case.final_time_cap + 1e-8
        ),
        message=str(result.message),
        parameters=parameters,
        final_state=propagation.final_state,
        residual=residual,
        propellant_used=propagation.tracker.propellant_used,
        total_time=propagation.total_time,
        angle_intervals=angle_intervals,
        attempts=len(attempts),
        transition_source=transition_source,
    )


def primer_initial_guesses(case: MultiStageCase, direct: DirectResult) -> list[Array]:
    alpha = float(direct.parameters[3])
    burn_one, coast, burn_two = direct.parameters[:3]
    guesses: list[Array] = []
    for alpha_offset in (0.0, -0.15, 0.15):
        for mass_factor in (0.5, 1.0, 1.5, 2.0):
            lambda_mass = -mass_factor * case.stages[-1].kappa
            rho = float(case.initial_state[0])
            radial_velocity = float(case.initial_state[1])
            tangential_velocity = float(case.initial_state[2])
            p_r = math.cos(alpha + alpha_offset)
            p_t = math.sin(alpha + alpha_offset)
            gravity_radial = tangential_velocity**2 / rho - 1.0 / rho**2
            lambda_rho = (
                p_r * gravity_radial
                - p_t * radial_velocity * tangential_velocity / rho
                + case.stages[0].gamma * (1.0 + lambda_mass / case.stages[0].kappa)
            ) / radial_velocity
            guesses.append(
                np.array(
                    [
                        alpha + alpha_offset,
                        burn_one,
                        coast,
                        burn_two,
                        lambda_rho,
                        lambda_mass,
                    ],
                    dtype=float,
                )
            )
    return guesses


def solve_primer(
    case: MultiStageCase,
    direct: DirectResult,
    max_attempts: int,
    final_time_active: bool,
) -> PrimerResult:
    total_capacity = sum(stage.full_burn_time for stage in case.stages)
    lower = np.array([-math.pi, 1e-8, 0.0, 1e-8, -100.0, -100.0], dtype=float)
    upper = np.array(
        [
            math.pi,
            total_capacity,
            case.final_time_cap,
            total_capacity,
            100.0,
            100.0,
        ],
        dtype=float,
    )
    attempts: list[tuple[Any, Array, Array, PrimerPropagation]] = []
    for initial in primer_initial_guesses(case, direct)[:max_attempts]:

        def residual_function(parameters: Array) -> Array:
            return primer_residual(case, parameters, final_time_active)

        result = least_squares(
            residual_function,
            initial,
            bounds=(lower, upper),
            x_scale="jac",
            ftol=2e-12,
            xtol=2e-12,
            gtol=2e-12,
            max_nfev=1800,
        )
        parameters = np.asarray(result.x, dtype=float)
        residual = primer_residual(case, parameters, final_time_active)
        propagation = propagate_primer(case, parameters)
        attempts.append((result, parameters, residual, propagation))
    accepted = [
        index
        for index, item in enumerate(attempts)
        if bool(item[0].success)
        and norm(item[2]) < 2e-6
        and item[3].transition_count == len(case.stages) - 1
        and (not final_time_active or item[3].total_time <= case.final_time_cap + 1e-8)
    ]
    candidates = accepted or list(range(len(attempts)))
    chosen_index = min(
        candidates,
        key=lambda index: (
            attempts[index][3].propellant_used
            if index in accepted
            else norm(attempts[index][2])
        ),
    )
    result, parameters, residual, propagation = attempts[chosen_index]
    return PrimerResult(
        success=bool(
            result.success
            and norm(residual) < 2e-6
            and propagation.transition_count == len(case.stages) - 1
        ),
        message=str(result.message),
        parameters=parameters,
        final_joint=propagation.final_joint,
        residual=residual,
        propagation=propagation,
        attempts=len(attempts),
    )


def phase_diagnostics(
    case: MultiStageCase, propagation: PrimerPropagation, lambda_burn: float
) -> list[dict[str, float | str]]:
    records: list[dict[str, float | str]] = []
    for phase in propagation.phases:
        h_values = np.array(
            [
                hamiltonian(sample, phase.stage, phase.throttle)
                for sample in phase.samples
            ],
            dtype=float,
        )
        phi_values = np.array(
            [switch_function(sample, phase.stage) for sample in phase.samples],
            dtype=float,
        )
        stage_index = case.stages.index(phase.stage)
        augmented_switch_values = phase.stage.gamma * phi_values - (
            lambda_burn if stage_index == 0 else 0.0
        )
        records.append(
            {
                "name": phase.name,
                "stage": phase.stage.name,
                "throttle": phase.throttle,
                "duration": phase.duration,
                "hamiltonian_start": float(h_values[0]),
                "hamiltonian_end": float(h_values[-1]),
                "hamiltonian_drift": float(h_values[-1] - h_values[0]),
                "switch_min": float(phi_values.min()),
                "switch_max": float(phi_values.max()),
                "augmented_switch_min": float(augmented_switch_values.min()),
                "augmented_switch_max": float(augmented_switch_values.max()),
            }
        )
    records.extend(
        [
            {
                "name": f"junction after {junction.source}",
                "stage": f"{junction.from_stage} -> {junction.to_stage}",
                "mass_before": float(junction.before_jump[3]),
                "mass_after": float(junction.after_jump[3]),
                "mass_jump": float(junction.after_jump[3] - junction.before_jump[3]),
                "lambda_mass_before": float(junction.before_jump[7]),
                "lambda_mass_after": float(junction.after_jump[7]),
                "lambda_mass_jump": float(
                    junction.after_jump[7] - junction.before_jump[7]
                ),
                "hamiltonian_before": hamiltonian(
                    junction.before_jump,
                    case.stages[junction.from_stage_index],
                    1.0,
                ),
                "hamiltonian_after": hamiltonian(
                    junction.after_jump,
                    case.stages[junction.from_stage_index],
                    0.0,
                ),
                "burn_clock_lambda": lambda_burn,
                "junction_hamiltonian_residual": hamiltonian(
                    junction.before_jump,
                    case.stages[junction.from_stage_index],
                    1.0,
                )
                + lambda_burn
                - hamiltonian(
                    junction.after_jump,
                    case.stages[junction.from_stage_index],
                    0.0,
                ),
            }
            for junction in propagation.junctions
        ]
    )
    final_stage = propagation.phases[-1].stage
    final_switch = switch_function(propagation.final_joint, final_stage)
    final_hamiltonian = hamiltonian(propagation.final_joint, final_stage, 1.0)
    records.append(
        {
            "name": "terminal",
            "stage": final_stage.name,
            "switch": final_switch,
            "hamiltonian": final_hamiltonian,
            "hamiltonian_from_switch": -final_stage.gamma * final_switch,
        }
    )
    return records


def result_record(
    case: MultiStageCase,
    direct: DirectResult,
    primer: PrimerResult,
    final_time_active: bool,
) -> dict[str, Any]:
    mass_scale = physical_fixture_stages(case.name)[0].start_mass
    return {
        "case": case.name,
        "normalization": {
            "target_radius_m": case.target_radius,
            "velocity_scale_m_per_s": case.velocity_scale,
            "time_scale_s": case.time_scale,
            "initial_state": case.initial_state.tolist(),
            "initial_time_to_apoapsis_seconds": case.initial_time_to_apoapsis_seconds,
            "final_time_cap_normalized": case.final_time_cap,
        },
        "staging": {
            "gap_seconds": case.staging_gap_seconds,
            "gap_normalized": case.staging_gap,
            "stages": [
                {
                    "name": stage.name,
                    "gamma": stage.gamma,
                    "kappa": stage.kappa,
                    "full_burn_time": stage.full_burn_time,
                    "start_mass_fraction": stage.start_mass,
                    "end_mass_fraction": stage.end_mass,
                }
                for stage in case.stages
            ],
            "jettison_mass_fractions": [
                jettison_mass(case, index) for index in range(len(case.stages) - 1)
            ],
        },
        "direct_reference": {
            "success": direct.success,
            "message": direct.message,
            "attempts": direct.attempts,
            "angle_intervals_per_powered_arc": direct.angle_intervals,
            "parameters": direct.parameters.tolist(),
            "transition_source": direct.transition_source,
            "total_time_normalized": direct.total_time,
            "total_time_seconds": direct.total_time * case.time_scale,
            "propellant_fraction": direct.propellant_used,
            "final_mass_fraction": float(direct.final_state[3]),
            "final_mass_kg": float(direct.final_state[3]) * mass_scale,
            "final_state": direct.final_state.tolist(),
            "residual_norm": norm(direct.residual),
        },
        "primer_solution": {
            "success": primer.success,
            "message": primer.message,
            "attempts": primer.attempts,
            "parameters": primer.parameters.tolist(),
            "final_time_active": final_time_active,
            "total_time_normalized": primer.propagation.total_time,
            "total_time_seconds": primer.propagation.total_time * case.time_scale,
            "propellant_fraction": primer.propagation.propellant_used,
            "final_mass_fraction": float(primer.final_joint[3]),
            "final_mass_kg": float(primer.final_joint[3]) * mass_scale,
            "final_state": primer.final_joint[:4].tolist(),
            "residual_norm": norm(primer.residual),
            "phase_diagnostics": phase_diagnostics(
                case,
                primer.propagation,
                stage_event_clock_lambda(case, primer.propagation),
            ),
        },
        "comparison": {
            "propellant_difference": primer.propagation.propellant_used
            - direct.propellant_used,
            "final_mass_difference": float(
                primer.final_joint[3] - direct.final_state[3]
            ),
            "total_time_difference": primer.propagation.total_time - direct.total_time,
        },
    }


def run_case(
    case_name: str,
    angle_intervals: int,
    direct_attempts: int,
    primer_attempts: int,
) -> dict[str, Any]:
    case = make_case(case_name)
    direct = solve_direct(case, angle_intervals, direct_attempts)
    if not direct.success:
        raise RuntimeError(
            f"{case.name} direct solve failed: {direct.message}; "
            f"residual={norm(direct.residual):.3e}"
        )
    final_time_active = bool(direct.total_time >= case.final_time_cap - 1e-6)
    primer = solve_primer(
        case,
        direct,
        primer_attempts,
        final_time_active,
    )
    if not primer.success:
        raise RuntimeError(
            f"{case.name} primer solve failed: {primer.message}; "
            f"residual={norm(primer.residual):.3e}"
        )
    return result_record(case, direct, primer, final_time_active)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        choices=("kerbin-stage-data", "kerbin-first-example", "both"),
        default="both",
    )
    parser.add_argument("--angles", type=int, default=8)
    parser.add_argument("--direct-attempts", type=int, default=20)
    parser.add_argument("--primer-attempts", type=int, default=12)
    parser.add_argument("--output", type=str, default="")
    arguments = parser.parse_args()
    case_names = (
        ("kerbin-stage-data", "kerbin-first-example")
        if arguments.case == "both"
        else (arguments.case,)
    )
    records = [
        run_case(
            case_name,
            arguments.angles,
            arguments.direct_attempts,
            arguments.primer_attempts,
        )
        for case_name in case_names
    ]
    for record in records:
        direct = record["direct_reference"]
        primer = record["primer_solution"]
        print(
            f"{record['case']}: direct={direct['success']} primer={primer['success']} "
            f"direct_fuel={direct['propellant_fraction']:.9f} "
            f"primer_fuel={primer['propellant_fraction']:.9f} "
            f"direct_res={direct['residual_norm']:.3e} "
            f"primer_res={primer['residual_norm']:.3e}"
        )
    output: object = records[0] if len(records) == 1 else records
    print(json.dumps(output, indent=2))
    if arguments.output:
        with open(arguments.output, "w", encoding="utf-8") as output_file:
            json.dump(output, output_file, indent=2)
            output_file.write("\n")


if __name__ == "__main__":
    main()
