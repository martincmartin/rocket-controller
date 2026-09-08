#!/usr/bin/env python3
"""Offline single-stage trajectory oracle.

The public API accepts physical three-dimensional state vectors and a segment
list. The solver converts the state to a normalized polar problem in the local
orbital plane, then compares explicit burn families with a Lawden diagnostic.

Run the three built-in Kerbin examples with::

    python3 offline/oracle_finder.py --output oracle.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, cast

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from scipy.special import expit

Array = NDArray[np.float64]
PhaseKind = Literal["burn", "coast"]
PhaseSequence = tuple[PhaseKind, ...]

DEFAULT_MU = 3.5316e12
DEFAULT_BODY_RADIUS = 600_000.0
DEFAULT_TARGET_ALTITUDE = 80_000.0
DEFAULT_TARGET_RADIUS = DEFAULT_BODY_RADIUS + DEFAULT_TARGET_ALTITUDE
DEFAULT_MAX_BURN_TIME = 150.0

INTEGRATION_RTOL = 2e-10
INTEGRATION_ATOL = 2e-12
RESIDUAL_TOL = 3e-6
SWITCH_TOL = 5e-5
RADIAL_VELOCITY_TOL = 1e-8
LIMIT_TOLERANCE_SECONDS = 1e-6
MIN_PHASE_DURATION_SECONDS = 0.02
INVALID_RESIDUAL = 1e3
INITIAL_COAST_BOUNDARY_TAU_TOL = 1e-12
OBJECTIVE_MASS_FRACTION_TIE_TOLERANCE = 1e-12
OBJECTIVE_DELTA_V_TAU_TIE_TOLERANCE = 1e-12

SUPPORTED_LAWDEN_SEQUENCES: tuple[PhaseSequence, ...] = (
    ("burn", "coast", "burn"),
    ("coast", "burn"),
    ("burn",),
    (),
)


class SegmentLike(Protocol):
    """Physical fields required from the first rocket segment."""

    @property
    def thrust(self) -> float: ...

    @property
    def ve(self) -> float: ...

    @property
    def initial_mass(self) -> float: ...


@dataclass(frozen=True)
class ExampleSegment:
    """Small segment record used by the built-in examples."""

    thrust: float
    ve: float
    initial_mass: float
    max_burn_time: float = DEFAULT_MAX_BURN_TIME


@dataclass(frozen=True, init=False)
class OrbitalPlane:
    """Orthonormal basis for the input state orbital plane."""

    r_hat: Array
    t_hat: Array
    n_hat: Array

    def __init__(self, position: Array, velocity: Array) -> None:
        position_array = np.asarray(position, dtype=float)
        velocity_array = np.asarray(velocity, dtype=float)
        if (
            position_array.shape != (3,)
            or velocity_array.shape != (3,)
            or not np.all(np.isfinite(position_array))
            or not np.all(np.isfinite(velocity_array))
        ):
            raise ValueError("position and velocity must be finite 3D vectors")
        radius = _norm(position_array)
        if radius <= 0.0:
            raise ValueError("position must not be zero")
        r_hat = position_array / radius
        radial_speed = float(np.dot(velocity_array, r_hat))
        tangent_vector = velocity_array - radial_speed * r_hat
        tangent_speed = _norm(tangent_vector)
        if tangent_speed <= 1e-10:
            raise ValueError("velocity must define a non-zero orbital tangent")
        t_hat = tangent_vector / tangent_speed
        n_hat = np.cross(r_hat, t_hat)
        object.__setattr__(self, "r_hat", r_hat)
        object.__setattr__(self, "t_hat", t_hat)
        object.__setattr__(self, "n_hat", n_hat)

    def to_dict(self) -> dict[str, object]:
        return {
            "r_hat": self.r_hat.tolist(),
            "t_hat": self.t_hat.tolist(),
            "n_hat": self.n_hat.tolist(),
            "initial_theta_rad": 0.0,
            "mapping": (
                "r_3d = x_2d * r_hat + y_2d * t_hat; positive theta follows t_hat"
            ),
        }


@dataclass(frozen=True)
class PhysicalContext:
    position_3d_m: Array
    velocity_3d_m_per_s: Array
    mu_m3_per_s2: float
    target_radius_m: float
    velocity_scale_m_per_s: float
    time_scale_s: float
    plane: OrbitalPlane
    mass0_kg: float
    thrust_n: float
    ve_m_per_s: float
    max_burn_time_s: float


@dataclass(frozen=True)
class Stage:
    gamma: float
    kappa: float
    max_burn_tau: float


class OrbitalParameters:
    """Common two-body orbital quantities in normalized units."""

    radius: float
    velocity: Array
    energy: float
    semi_major_axis: float
    angular_momentum: float
    eccentricity: float
    eccentric_anomaly: float
    mean_anomaly: float
    mean_motion: float
    periapsis: float
    apoapsis: float

    def __init__(self, radius: float, velocity: Array) -> None:
        velocity_array = np.asarray(velocity, dtype=float)
        if velocity_array.shape != (2,) or not np.all(np.isfinite(velocity_array)):
            raise ValueError("velocity must be a finite two-element vector")
        if not math.isfinite(radius) or radius <= 0.0:
            raise ValueError("radius must be positive")
        radial_velocity, tangential_velocity = velocity_array
        self.radius = float(radius)
        self.velocity = velocity_array.copy()
        self.energy = float(
            0.5 * (radial_velocity**2 + tangential_velocity**2) - 1.0 / radius
        )
        self.angular_momentum = float(radius * tangential_velocity)
        self.eccentricity = float(
            math.sqrt(max(0.0, 1.0 + 2.0 * self.energy * self.angular_momentum**2))
        )
        if self.energy >= 0.0:
            self.semi_major_axis = math.nan
            self.eccentric_anomaly = math.nan
            self.mean_anomaly = math.nan
            self.mean_motion = math.nan
            self.periapsis = self.angular_momentum**2 / (1.0 + self.eccentricity)
            self.apoapsis = math.inf
            return

        self.semi_major_axis = -1.0 / (2.0 * self.energy)
        self.periapsis = self.semi_major_axis * (1.0 - self.eccentricity)
        self.apoapsis = self.semi_major_axis * (1.0 + self.eccentricity)
        self.mean_motion = self.semi_major_axis**-1.5
        if self.eccentricity <= 1e-12:
            self.eccentric_anomaly = 0.0
            self.mean_anomaly = 0.0
            return

        cosine_eccentric_anomaly = (
            1.0 - radius / self.semi_major_axis
        ) / self.eccentricity
        direction = 1.0 if self.angular_momentum >= 0.0 else -1.0
        sine_eccentric_anomaly = (
            direction * radius * radial_velocity / math.sqrt(self.semi_major_axis)
        ) / self.eccentricity
        self.eccentric_anomaly = math.atan2(
            sine_eccentric_anomaly, cosine_eccentric_anomaly
        ) % (2.0 * math.pi)
        self.mean_anomaly = (
            self.eccentric_anomaly
            - self.eccentricity * math.sin(self.eccentric_anomaly)
        ) % (2.0 * math.pi)

    def coast_to_apoapsis(self) -> tuple[OrbitalParameters, float]:
        if not math.isfinite(self.semi_major_axis):
            raise ValueError("cannot find an apoapsis for a non-elliptic orbit")
        if self.eccentricity <= 1e-12:
            return OrbitalParameters(self.radius, self.velocity), 0.0
        delta_mean_anomaly = (math.pi - self.mean_anomaly) % (2.0 * math.pi)
        time_to_apoapsis = delta_mean_anomaly / self.mean_motion
        apo_speed = math.sqrt(
            max(0.0, 2.0 / self.apoapsis - 1.0 / self.semi_major_axis)
        )
        apo_orbit = OrbitalParameters(
            self.apoapsis, np.array([0.0, apo_speed], dtype=float)
        )
        return apo_orbit, float(time_to_apoapsis)

    def period(self) -> float:
        if not math.isfinite(self.mean_motion):
            raise ValueError("initial orbit is not elliptic")
        return float(2.0 * math.pi / self.mean_motion)


@dataclass(frozen=True)
class Problem:
    physical: PhysicalContext
    radius: float
    velocity: Array
    stage: Stage
    initial_apoapsis: float
    time_to_apoapsis: float
    orbital_period: float
    time_limit: float

    @property
    def target_state(self) -> Array:
        return np.array([1.0, 0.0, 1.0], dtype=float)

    @property
    def min_phase_duration(self) -> float:
        return MIN_PHASE_DURATION_SECONDS / self.physical.time_scale_s

    def to_dict(self) -> dict[str, object]:
        physical = self.physical
        return {
            "position_m": physical.position_3d_m.tolist(),
            "velocity_m_per_s": physical.velocity_3d_m_per_s.tolist(),
            "mu_m3_per_s2": physical.mu_m3_per_s2,
            "target_radius_m": physical.target_radius_m,
            "velocity_scale_m_per_s": physical.velocity_scale_m_per_s,
            "time_scale_s": physical.time_scale_s,
            "initial_radius": self.radius,
            "initial_velocity": self.velocity.tolist(),
            "stage": {
                "mass0_kg": physical.mass0_kg,
                "thrust_n": physical.thrust_n,
                "ve_m_per_s": physical.ve_m_per_s,
                "gamma": self.stage.gamma,
                "kappa": self.stage.kappa,
                "max_burn_time_s": physical.max_burn_time_s,
                "max_burn_tau": self.stage.max_burn_tau,
            },
            "initial_apoapsis_m": self.initial_apoapsis * physical.target_radius_m,
            "time_to_apoapsis_s": self.time_to_apoapsis * physical.time_scale_s,
            "orbital_period_s": self.orbital_period * physical.time_scale_s,
            "time_limit_s": self.time_limit * physical.time_scale_s,
            "min_phase_duration_s": MIN_PHASE_DURATION_SECONDS,
            "min_phase_duration": self.min_phase_duration,
        }


@dataclass
class PhaseData:
    kind: PhaseKind
    tau: Array
    joint: Array
    eta: Array
    throttle: Array
    switching: Array
    alpha: Array
    acceleration: Array
    lawden: bool

    @property
    def duration_tau(self) -> float:
        return float(self.tau[-1] - self.tau[0])

    def to_dict(
        self, problem: Problem, theta0: float
    ) -> tuple[dict[str, object], float]:
        physical = problem.physical
        rho = self.joint[:, 0]
        radial_velocity = self.joint[:, 1]
        tangential_velocity = self.joint[:, 2]
        theta = _integrate_theta(self.tau, rho, tangential_velocity, theta0)

        radius = rho * physical.target_radius_m
        radial_speed = radial_velocity * physical.velocity_scale_m_per_s
        tangential_speed = tangential_velocity * physical.velocity_scale_m_per_s
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        position_2d = np.column_stack((radius * cos_theta, radius * sin_theta))
        velocity_2d = np.column_stack(
            (
                radial_speed * cos_theta - tangential_speed * sin_theta,
                radial_speed * sin_theta + tangential_speed * cos_theta,
            )
        )
        position_3d = (
            position_2d[:, 0, None] * physical.plane.r_hat
            + position_2d[:, 1, None] * physical.plane.t_hat
        )
        velocity_3d = (
            velocity_2d[:, 0, None] * physical.plane.r_hat
            + velocity_2d[:, 1, None] * physical.plane.t_hat
        )

        direction_radial_tangential = np.column_stack(
            (np.cos(self.alpha), np.sin(self.alpha))
        )
        direction_2d = np.column_stack(
            (
                np.cos(theta) * np.cos(self.alpha) - np.sin(theta) * np.sin(self.alpha),
                np.sin(theta) * np.cos(self.alpha) + np.cos(theta) * np.sin(self.alpha),
            )
        )
        direction_3d = (
            direction_2d[:, 0, None] * physical.plane.r_hat
            + direction_2d[:, 1, None] * physical.plane.t_hat
        )
        acceleration_magnitude = (
            self.acceleration * physical.velocity_scale_m_per_s / physical.time_scale_s
        )
        acceleration_2d = acceleration_magnitude[:, None] * direction_2d
        acceleration_3d = (
            acceleration_2d[:, 0, None] * physical.plane.r_hat
            + acceleration_2d[:, 1, None] * physical.plane.t_hat
        )

        if self.lawden:
            costate = self.joint[:, 4:8]
            costate_columns = [
                "lambda_rho",
                "p_r",
                "p_t",
                "lambda_mass",
            ]
        else:
            costate = self.joint[:, 3:6]
            costate_columns = ["lambda_rho", "p_r", "p_t"]

        record: dict[str, object] = {
            "kind": self.kind,
            "time_tau": self.tau.tolist(),
            "time_s": (self.tau * physical.time_scale_s).tolist(),
            "state_columns": [
                "theta_rad",
                "radius_m",
                "radial_velocity_m_per_s",
                "tangential_velocity_m_per_s",
                "mass_kg",
            ],
            "state": np.column_stack(
                (
                    theta,
                    radius,
                    radial_speed,
                    tangential_speed,
                    self.eta * physical.mass0_kg,
                )
            ).tolist(),
            "position_2d_m": position_2d.tolist(),
            "velocity_2d_m_per_s": velocity_2d.tolist(),
            "position_3d_m": position_3d.tolist(),
            "velocity_3d_m_per_s": velocity_3d.tolist(),
            "costate_columns": costate_columns,
            "costate": costate.tolist(),
            "control": {
                "throttle": self.throttle.tolist(),
                "thrust_angle_rad": self.alpha.tolist(),
                "thrust_direction_radial_tangential": (
                    direction_radial_tangential.tolist()
                ),
                "thrust_direction_2d": direction_2d.tolist(),
                "thrust_direction_3d": direction_3d.tolist(),
                "thrust_acceleration_m_per_s2": acceleration_magnitude.tolist(),
                "thrust_acceleration_2d_m_per_s2": acceleration_2d.tolist(),
                "thrust_acceleration_3d_m_per_s2": acceleration_3d.tolist(),
                "switching_function": self.switching.tolist(),
            },
        }
        return record, float(theta[-1])


@dataclass
class Candidate:
    name: str
    formulation: str
    success: bool
    message: str
    parameters: Array
    residual: Array
    final_state: Array | None
    final_eta: float | None
    phases: list[PhaseData]
    powered_tau: float
    final_tau: float
    limit_hit: bool
    sequence: PhaseSequence
    diagnostics: dict[str, object]

    @property
    def residual_norm(self) -> float:
        return float(np.linalg.norm(self.residual))

    @property
    def final_mass_fraction(self) -> float | None:
        return self.final_eta

    @property
    def fuel_fraction(self) -> float | None:
        if self.final_eta is None:
            return None
        return 1.0 - self.final_eta

    @property
    def delta_v(self) -> float | None:
        if self.final_eta is None or self.final_eta <= 0.0:
            return None
        return cast(float, self.diagnostics["kappa"]) * math.log(1.0 / self.final_eta)

    def final_mass_kg(self, problem: Problem) -> float | None:
        if self.final_mass_fraction is None:
            return None
        return problem.physical.mass0_kg * self.final_mass_fraction

    def fuel_consumed_kg(self, problem: Problem) -> float | None:
        if self.fuel_fraction is None:
            return None
        return problem.physical.mass0_kg * self.fuel_fraction

    def delta_v_m_per_s(self, problem: Problem) -> float | None:
        if self.delta_v is None:
            return None
        return problem.physical.velocity_scale_m_per_s * self.delta_v

    def summary(self, problem: Problem) -> dict[str, object]:
        return {
            "name": self.name,
            "formulation": self.formulation,
            "success": self.success,
            "message": self.message,
            "parameters": self.parameters.tolist(),
            "residual": self.residual.tolist(),
            "residual_norm": self.residual_norm,
            "final_time_tau": self.final_tau,
            "final_time_s": self.final_tau * problem.physical.time_scale_s,
            "powered_time_tau": self.powered_tau,
            "powered_time_s": self.powered_tau * problem.physical.time_scale_s,
            "final_mass_fraction": self.final_mass_fraction,
            "fuel_fraction": self.fuel_fraction,
            "delta_v": self.delta_v,
            "final_mass_kg": self.final_mass_kg(problem),
            "fuel_consumed_kg": self.fuel_consumed_kg(problem),
            "delta_v_m_per_s": self.delta_v_m_per_s(problem),
            "limit_hit": self.limit_hit,
            "phase_sequence": "/".join(self.sequence) or "empty",
            "diagnostics": _json_value(self.diagnostics),
        }

    def to_dict(self, problem: Problem, *, full: bool) -> dict[str, object]:
        record = self.summary(problem)
        record["initial_costate"] = self._initial_costate().tolist()
        if self.final_state is not None:
            record["final_state"] = self.final_state.tolist()
        if full:
            phases: list[dict[str, object]] = []
            theta = 0.0
            for phase in self.phases:
                phase_record, theta = phase.to_dict(problem, theta)
                phases.append(phase_record)
            record["phases"] = phases
        return record

    def _initial_costate(self) -> Array:
        if self.formulation.startswith("lawden"):
            return np.asarray(
                [
                    self.parameters[2],
                    math.cos(self.parameters[0]),
                    math.sin(self.parameters[0]),
                    self.parameters[3],
                ],
                dtype=float,
            )
        return np.asarray(self.parameters[:3], dtype=float)


def _candidate_is_better(candidate: Candidate, incumbent: Candidate | None) -> bool:
    if incumbent is None:
        return True
    if candidate.success != incumbent.success:
        return candidate.success
    if not candidate.success:
        return candidate.residual_norm < incumbent.residual_norm
    if candidate.final_mass_fraction is None or incumbent.final_mass_fraction is None:
        return False
    mass_difference = candidate.final_mass_fraction - incumbent.final_mass_fraction
    if abs(mass_difference) > OBJECTIVE_MASS_FRACTION_TIE_TOLERANCE:
        return mass_difference > 0.0
    candidate_delta_v = candidate.delta_v
    incumbent_delta_v = incumbent.delta_v
    if candidate_delta_v is not None and incumbent_delta_v is not None:
        delta_v_difference = candidate_delta_v - incumbent_delta_v
        if abs(delta_v_difference) > OBJECTIVE_DELTA_V_TAU_TIE_TOLERANCE:
            return delta_v_difference < 0.0
    candidate_fuel = candidate.fuel_fraction
    incumbent_fuel = incumbent.fuel_fraction
    if candidate_fuel is not None and incumbent_fuel is not None:
        return candidate_fuel < incumbent_fuel
    return False


@dataclass
class OracleResult:
    problem: Problem
    selected: Candidate
    candidates: list[Candidate]
    warnings: list[str]
    elapsed_seconds: float

    def to_dict(self) -> dict[str, object]:
        return {
            "input": self.problem.to_dict(),
            "plane": self.problem.physical.plane.to_dict(),
            "selected": self.selected.to_dict(self.problem, full=True),
            "candidates": [
                candidate.to_dict(self.problem, full=False)
                for candidate in self.candidates
            ],
            "warnings": self.warnings,
            "elapsed_seconds": self.elapsed_seconds,
        }


@dataclass(frozen=True)
class TimingSeed:
    burn_raise_apoapsis: float
    coast: float
    burn_circularize: float


def _json_value(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _phase_sequence(phases: Sequence[PhaseData]) -> PhaseSequence:
    return tuple(phase.kind for phase in phases if len(phase.tau) > 0)


def _sequence_text(sequence: PhaseSequence) -> str:
    return "/".join(sequence) or "empty"


def _mode_sequence(mode: str) -> PhaseSequence:
    if mode == "burn_coast_burn":
        return ("burn", "coast", "burn")
    if mode == "coast_burn":
        return ("coast", "burn")
    return ("burn",)


def _as_vector(value: object, name: str) -> Array:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite three-element vector")
    return np.asarray(vector, dtype=float)


def _norm(vector: Array) -> float:
    return float(np.linalg.norm(vector))


def _polar_rhs(
    _time: float,
    state: Array,
    gamma: float,
    kappa: float,
    throttle: float,
    alpha: float,
) -> Array:
    rho, radial_velocity, tangential_velocity, eta = state
    if rho <= 0.0 or eta <= 0.0:
        return np.full(4, np.nan)
    acceleration = throttle * gamma / eta
    return np.array(
        [
            radial_velocity,
            tangential_velocity**2 / rho
            - 1.0 / rho**2
            + acceleration * math.cos(alpha),
            -radial_velocity * tangential_velocity / rho
            + acceleration * math.sin(alpha),
            -throttle * gamma / kappa,
        ],
        dtype=float,
    )


def _prepare_problem(
    position: Array,
    velocity: Array,
    segments: Sequence[SegmentLike],
    mu: float,
    target_radius: float,
    max_burn_time: float,
) -> Problem:
    if not segments:
        raise ValueError("segments must contain at least one segment")
    if mu <= 0.0 or target_radius <= 0.0 or max_burn_time <= 0.0:
        raise ValueError("mu, target_radius, and max_burn_time must be positive")

    radius = _norm(position)
    if radius <= 0.0:
        raise ValueError("position must not be zero")
    plane = OrbitalPlane(position, velocity)
    radial_speed = float(np.dot(velocity, plane.r_hat))
    tangent_speed = float(np.dot(velocity, plane.t_hat))

    velocity_scale = math.sqrt(mu / target_radius)
    time_scale = target_radius / velocity_scale
    initial_radius = radius / target_radius
    initial_velocity = np.array(
        [radial_speed / velocity_scale, tangent_speed / velocity_scale],
        dtype=float,
    )
    segment = segments[0]
    thrust = float(segment.thrust)
    ve = float(segment.ve)
    mass0 = float(segment.initial_mass)
    if thrust <= 0.0 or ve <= 0.0 or mass0 <= 0.0:
        raise ValueError("the first segment must have positive thrust, ve, and mass")

    physical = PhysicalContext(
        position_3d_m=position,
        velocity_3d_m_per_s=velocity,
        mu_m3_per_s2=mu,
        target_radius_m=target_radius,
        velocity_scale_m_per_s=velocity_scale,
        time_scale_s=time_scale,
        plane=plane,
        mass0_kg=mass0,
        thrust_n=thrust,
        ve_m_per_s=ve,
        max_burn_time_s=max_burn_time,
    )
    stage = Stage(
        gamma=thrust * target_radius / (mass0 * velocity_scale**2),
        kappa=ve / velocity_scale,
        max_burn_tau=max_burn_time / time_scale,
    )
    initial_orbit = OrbitalParameters(initial_radius, initial_velocity)
    orbital_period = initial_orbit.period()
    time_to_apoapsis = initial_orbit.coast_to_apoapsis()[1]
    time_limit = time_to_apoapsis + 0.75 * orbital_period
    return Problem(
        physical=physical,
        radius=initial_radius,
        velocity=initial_velocity,
        stage=stage,
        initial_apoapsis=initial_orbit.apoapsis,
        time_to_apoapsis=time_to_apoapsis,
        orbital_period=orbital_period,
        time_limit=time_limit,
    )


def _burn_time_for_delta_v(
    delta_v: float, mass: float, thrust: float, ve: float
) -> float:
    if delta_v <= 0.0:
        return 0.0
    return mass * ve / thrust * (1.0 - math.exp(-delta_v / ve))


def _impulse_timing_seed(problem: Problem) -> TimingSeed:
    speed = _norm(problem.velocity)
    target = 1.0
    delta_v_raise_apoapsis = 0.0
    post_velocity = problem.velocity.copy()
    if problem.initial_apoapsis < target - 1e-6:
        angular_momentum_coefficient = problem.radius * problem.velocity[1] / speed
        denominator = 1.0 - (angular_momentum_coefficient / target) ** 2
        numerator = 2.0 * (1.0 / problem.radius - 1.0 / target)
        if denominator <= 0.0 or numerator <= 0.0:
            raise ValueError("could not construct the prograde impulse seed")
        post_speed = math.sqrt(numerator / denominator)
        delta_v_raise_apoapsis = max(0.0, post_speed - speed)
        post_velocity = problem.velocity * (post_speed / speed)

    mass_after_raise_apoapsis = math.exp(-delta_v_raise_apoapsis / problem.stage.kappa)
    apo_orbit, time_to_apoapsis = OrbitalParameters(
        problem.radius, post_velocity
    ).coast_to_apoapsis()
    circular_speed = math.sqrt(1.0 / apo_orbit.radius)
    delta_v_circularize = max(0.0, circular_speed - apo_orbit.velocity[1])
    burn_raise_apoapsis = _burn_time_for_delta_v(
        delta_v_raise_apoapsis, 1.0, problem.stage.gamma, problem.stage.kappa
    )
    burn_circularize = _burn_time_for_delta_v(
        delta_v_circularize,
        mass_after_raise_apoapsis,
        problem.stage.gamma,
        problem.stage.kappa,
    )
    powered = burn_raise_apoapsis + burn_circularize
    if powered > problem.stage.max_burn_tau:
        scale = problem.stage.max_burn_tau / powered
        burn_raise_apoapsis *= scale
        burn_circularize *= scale
    coast = max(
        0.0,
        time_to_apoapsis - burn_raise_apoapsis - 0.5 * burn_circularize,
    )
    if burn_raise_apoapsis + coast + burn_circularize > problem.time_limit:
        coast = max(
            0.0,
            problem.time_limit - burn_raise_apoapsis - burn_circularize,
        )
    return TimingSeed(
        burn_raise_apoapsis=burn_raise_apoapsis,
        coast=coast,
        burn_circularize=burn_circularize,
    )


def _prograde_apoapsis_angle(radius: float, velocity: Array) -> float:
    radial_velocity, tangential_velocity = velocity
    angular_momentum = radius * tangential_velocity
    eccentricity_vector = np.array(
        [
            angular_momentum * tangential_velocity - 1.0,
            -angular_momentum * radial_velocity,
        ],
        dtype=float,
    )
    eccentricity = _norm(eccentricity_vector)
    if eccentricity <= 1e-12:
        return math.pi / 2.0
    apoapsis_direction = -eccentricity_vector / eccentricity
    if angular_momentum >= 0.0:
        prograde = np.array([-apoapsis_direction[1], apoapsis_direction[0]])
    else:
        prograde = np.array([apoapsis_direction[1], -apoapsis_direction[0]])
    return math.atan2(float(prograde[1]), float(prograde[0]))


def _lambda_rho_seed(radius: float, velocity: Array, primer: Array) -> float:
    radial_velocity, tangential_velocity = velocity
    if abs(radial_velocity) <= RADIAL_VELOCITY_TOL:
        return 0.0
    gravity_term = tangential_velocity**2 / radius - 1.0 / radius**2
    value = (
        primer[0] * gravity_term
        - primer[1] * radial_velocity * tangential_velocity / radius
    ) / radial_velocity
    if not math.isfinite(value):
        return 0.0
    return float(np.clip(value, -50.0, 50.0))


def _explicit_seeds(problem: Problem, timing: TimingSeed, mode: str) -> list[Array]:
    velocity_direction = problem.velocity.copy()
    velocity_direction /= _norm(velocity_direction)
    directions = [
        velocity_direction,
        np.array([0.0, 1.0], dtype=float),
        np.array(
            [
                math.cos(_prograde_apoapsis_angle(problem.radius, problem.velocity)),
                math.sin(_prograde_apoapsis_angle(problem.radius, problem.velocity)),
            ],
            dtype=float,
        ),
    ]
    time_variants = [
        (timing.burn_raise_apoapsis, timing.coast, timing.burn_circularize),
        (
            0.75 * timing.burn_raise_apoapsis,
            timing.coast,
            1.25 * timing.burn_circularize,
        ),
        (
            1.25 * timing.burn_raise_apoapsis,
            timing.coast,
            0.75 * timing.burn_circularize,
        ),
    ]
    seeds: list[Array] = []
    for direction in directions:
        lambda_rho = _lambda_rho_seed(problem.radius, problem.velocity, direction)
        lambda_values = (
            lambda_rho * 0.5,
            lambda_rho,
            lambda_rho * 2.0,
        )
        if abs(lambda_rho) <= 1e-12:
            lambda_values = (-1.0, 0.0, 1.0)
        for lambda_value in lambda_values:
            for burn_raise_apoapsis, coast, burn_circularize in time_variants:
                if mode == "burn_coast_burn":
                    values = [
                        lambda_value,
                        float(direction[0]),
                        float(direction[1]),
                        max(problem.min_phase_duration, burn_raise_apoapsis),
                        max(problem.min_phase_duration, coast),
                        max(problem.min_phase_duration, burn_circularize),
                    ]
                elif mode == "coast_burn":
                    values = [
                        lambda_value,
                        float(direction[0]),
                        float(direction[1]),
                        max(problem.min_phase_duration, coast),
                        max(problem.min_phase_duration, burn_circularize),
                    ]
                else:
                    duration = burn_raise_apoapsis + burn_circularize
                    if duration <= 0.0:
                        duration = max(1e-4, timing.burn_circularize)
                    values = [
                        lambda_value,
                        float(direction[0]),
                        float(direction[1]),
                        max(
                            problem.min_phase_duration,
                            min(problem.stage.max_burn_tau * 0.95, duration),
                        ),
                    ]
                seeds.append(np.asarray(values, dtype=float))
    return seeds


def _prussing_rhs(
    _time: float,
    joint: Array,
    gamma: float,
    eta: float,
    throttle: float,
) -> Array:
    rho, radial_velocity, tangential_velocity, lambda_rho, p_r, p_t = joint
    if rho <= 0.0 or eta <= 0.0:
        return np.full(6, np.nan)
    primer_length = math.hypot(p_r, p_t)
    if primer_length <= 1e-14:
        return np.full(6, np.nan)
    alpha = math.atan2(p_t, p_r)
    acceleration = throttle * gamma / eta
    return np.array(
        [
            radial_velocity,
            tangential_velocity**2 / rho
            - 1.0 / rho**2
            + acceleration * math.cos(alpha),
            -radial_velocity * tangential_velocity / rho
            + acceleration * math.sin(alpha),
            p_r * (2.0 / rho**3 - tangential_velocity**2 / rho**2)
            + p_t * radial_velocity * tangential_velocity / rho**2,
            lambda_rho + p_t * tangential_velocity / rho,
            -2.0 * p_r * tangential_velocity / rho + p_t * radial_velocity / rho,
        ],
        dtype=float,
    )


def _integrate_explicit_arc(
    problem: Problem,
    joint: Array,
    start_tau: float,
    duration: float,
    eta_start: float,
    kind: PhaseKind,
) -> tuple[Array, float, PhaseData]:
    throttle = 1.0 if kind == "burn" else 0.0
    gamma = problem.stage.gamma
    mass_rate = problem.stage.gamma / problem.stage.kappa

    def rhs(time_tau: float, value: Array) -> Array:
        elapsed = time_tau - start_tau
        eta = eta_start - mass_rate * elapsed if kind == "burn" else eta_start
        return _prussing_rhs(time_tau, value, gamma, eta, throttle)

    if duration <= 0.0:
        tau = np.array([start_tau], dtype=float)
        values = np.asarray(joint, dtype=float)[None, :]
        eta = np.array([eta_start], dtype=float)
        phase = _make_explicit_phase(kind, tau, values, eta, problem)
        return joint.copy(), eta_start, phase
    solution = solve_ivp(
        rhs,
        (start_tau, start_tau + duration),
        joint,
        method="DOP853",
        rtol=INTEGRATION_RTOL,
        atol=INTEGRATION_ATOL,
        dense_output=True,
    )
    if not solution.success or not np.all(np.isfinite(solution.y[:, -1])):
        raise ValueError(solution.message)
    eta_end = eta_start - mass_rate * duration if kind == "burn" else eta_start
    if eta_end <= 0.0:
        raise ValueError("segment mass became non-positive")
    tau = np.asarray(solution.t, dtype=float)
    values = np.asarray(solution.y.T, dtype=float)
    eta = (
        eta_start - mass_rate * (tau - start_tau)
        if kind == "burn"
        else np.full(len(tau), eta_start, dtype=float)
    )
    phase = _make_explicit_phase(kind, tau, values, eta, problem)
    return np.asarray(solution.y[:, -1], dtype=float), eta_end, phase


def _make_explicit_phase(
    kind: PhaseKind,
    tau: Array,
    joint: Array,
    eta: Array,
    problem: Problem,
) -> PhaseData:
    primer_length = np.hypot(joint[:, 4], joint[:, 5])
    switching = 1.0 - primer_length
    alpha = np.arctan2(joint[:, 5], joint[:, 4])
    throttle = np.full(len(tau), 1.0 if kind == "burn" else 0.0, dtype=float)
    acceleration = throttle * problem.stage.gamma / eta
    return PhaseData(
        kind=kind,
        tau=tau,
        joint=joint,
        eta=eta,
        throttle=throttle,
        switching=np.asarray(switching, dtype=float),
        alpha=np.asarray(alpha, dtype=float),
        acceleration=np.asarray(acceleration, dtype=float),
        lawden=False,
    )


def _mode_arcs(mode: str, parameters: Array) -> list[tuple[PhaseKind, float]]:
    if mode == "burn_coast_burn":
        return [
            ("burn", float(parameters[3])),
            ("coast", float(parameters[4])),
            ("burn", float(parameters[5])),
        ]
    if mode == "coast_burn":
        return [("coast", float(parameters[3])), ("burn", float(parameters[4]))]
    return [("burn", float(parameters[3]))]


def _propagate_explicit(
    problem: Problem,
    mode: str,
    parameters: Array,
    collect: bool,
) -> tuple[Array, float, list[float], list[PhaseData], float]:
    arcs = _mode_arcs(mode, parameters)
    total_burn = sum(duration for kind, duration in arcs if kind == "burn")
    total_time = sum(duration for _kind, duration in arcs)
    if total_burn > problem.stage.max_burn_tau + 1e-9:
        raise ValueError("powered-time limit exceeded")
    if total_time > problem.time_limit + 1e-9:
        raise ValueError("first-outbound time limit exceeded")
    joint = np.array(
        [
            problem.radius,
            *problem.velocity,
            float(parameters[0]),
            float(parameters[1]),
            float(parameters[2]),
        ],
        dtype=float,
    )
    eta = 1.0
    start_tau = 0.0
    switch_ends: list[float] = []
    phases: list[PhaseData] = []
    for kind, duration in arcs:
        joint, eta, phase = _integrate_explicit_arc(
            problem, joint, start_tau, duration, eta, kind
        )
        switch_ends.append(float(phase.switching[-1]))
        if collect:
            phases.append(phase)
        start_tau += duration
    return joint, eta, switch_ends, phases, total_burn


def _explicit_residual(problem: Problem, mode: str, parameters: Array) -> Array:
    count = 3 + len(_mode_arcs(mode, parameters))
    try:
        final, _eta, switches, _phases, _burn = _propagate_explicit(
            problem, mode, parameters, collect=False
        )
        return np.asarray(
            [final[0] - 1.0, final[1], final[2] - 1.0, *switches],
            dtype=float,
        )
    except (FloatingPointError, ValueError, ZeroDivisionError):
        return np.full(count, INVALID_RESIDUAL)


def _explicit_bounds(problem: Problem, mode: str) -> tuple[Array, Array]:
    costate_lower = np.array([-100.0, -10.0, -10.0], dtype=float)
    costate_upper = np.array([100.0, 10.0, 10.0], dtype=float)
    minimum = problem.min_phase_duration
    if mode == "burn_coast_burn":
        lower_times = np.full(3, minimum, dtype=float)
        upper_times = np.array(
            [
                problem.stage.max_burn_tau,
                problem.time_limit,
                problem.stage.max_burn_tau,
            ],
            dtype=float,
        )
    elif mode == "coast_burn":
        lower_times = np.full(2, minimum, dtype=float)
        upper_times = np.array(
            [problem.time_limit, problem.stage.max_burn_tau], dtype=float
        )
    else:
        lower_times = np.array([minimum], dtype=float)
        upper_times = np.array([problem.stage.max_burn_tau], dtype=float)
    return (
        np.concatenate((costate_lower, lower_times)),
        np.concatenate((costate_upper, upper_times)),
    )


def _valid_explicit(
    problem: Problem,
    mode: str,
    parameters: Array,
    residual: Array,
    phases: list[PhaseData],
) -> bool:
    if not np.all(np.isfinite(parameters)) or not np.all(np.isfinite(residual)):
        return False
    if _norm(residual) >= RESIDUAL_TOL:
        return False
    if mode == "burn_coast_burn" and parameters[3] < problem.min_phase_duration - 1e-12:
        return False
    try:
        _final, eta, _switches, _phases, powered_tau = _propagate_explicit(
            problem, mode, parameters, collect=False
        )
    except (FloatingPointError, ValueError, ZeroDivisionError):
        return False
    if eta <= 0.0 or powered_tau > problem.stage.max_burn_tau + 1e-9:
        return False
    if any(
        duration < problem.min_phase_duration - 1e-12
        for phase in phases
        for duration in (phase.duration_tau,)
    ):
        return False
    for phase in phases:
        if phase.kind == "burn" and float(np.max(phase.switching)) > SWITCH_TOL:
            return False
        switching = phase.switching
        if (
            phase.kind == "coast"
            and phase.tau[0] <= INITIAL_COAST_BOUNDARY_TAU_TOL
            and len(switching) > 1
        ):
            # At t=0 a coast has no preceding burn boundary. Check its
            # interior samples rather than treating the initial boundary as
            # an interior switching-sign violation.
            switching = switching[1:]
        if phase.kind == "coast" and float(np.min(switching)) < -SWITCH_TOL:
            return False
    return True


def _candidate_from_explicit(
    problem: Problem,
    mode: str,
    parameters: Array,
    result_success: bool,
    result_message: str,
    nfev: int,
) -> Candidate:
    residual = _explicit_residual(problem, mode, parameters)
    try:
        final, eta, _switches, phases, powered_tau = _propagate_explicit(
            problem, mode, parameters, collect=True
        )
        final_state = np.asarray([final[0], final[1], final[2], eta], dtype=float)
        final_tau = sum(duration for _kind, duration in _mode_arcs(mode, parameters))
        valid = _valid_explicit(problem, mode, parameters, residual, phases)
    except (FloatingPointError, ValueError, ZeroDivisionError) as error:
        final = None
        eta = None
        phases = []
        powered_tau = 0.0
        final_state = None
        final_tau = 0.0
        valid = False
        result_message = f"{result_message}; propagation failed: {error}"
    limit_hit = powered_tau * problem.physical.time_scale_s >= (
        problem.physical.max_burn_time_s - LIMIT_TOLERANCE_SECONDS
    )
    diagnostics: dict[str, object] = {
        "nfev": nfev,
        "kappa": problem.stage.kappa,
        "optimizer_success": result_success,
    }
    if valid and not result_success:
        result_message = f"physical checks accepted optimizer result: {result_message}"
    return Candidate(
        name=mode,
        formulation="prussing-explicit",
        success=bool(valid),
        message=result_message,
        parameters=np.asarray(parameters, dtype=float),
        residual=np.asarray(residual, dtype=float),
        final_state=final_state,
        final_eta=None if eta is None else float(eta),
        phases=phases,
        powered_tau=float(powered_tau),
        final_tau=float(final_tau),
        limit_hit=limit_hit,
        sequence=_mode_sequence(mode),
        diagnostics=diagnostics,
    )


def _solve_explicit_mode(
    problem: Problem,
    mode: str,
    timing: TimingSeed,
) -> Candidate:
    lower, upper = _explicit_bounds(problem, mode)
    best: Candidate | None = None
    seed_results: list[dict[str, object]] = []
    for seed_index, seed in enumerate(_explicit_seeds(problem, timing, mode)):
        seed = np.clip(seed, lower + 1e-10, upper - 1e-10)
        try:
            result = cast(Any, least_squares)(
                lambda values: _explicit_residual(problem, mode, values),
                seed,
                bounds=(lower, upper),
                method="trf",
                x_scale="jac",
                ftol=2e-10,
                xtol=2e-10,
                gtol=2e-10,
                max_nfev=350,
            )
            candidate = _candidate_from_explicit(
                problem,
                mode,
                np.asarray(result.x, dtype=float),
                bool(result.success),
                str(result.message),
                int(result.nfev),
            )
        except (FloatingPointError, ValueError, ZeroDivisionError) as error:
            candidate = _candidate_from_explicit(
                problem,
                mode,
                seed,
                False,
                str(error),
                0,
            )
        seed_results.append(
            {
                "seed_index": seed_index,
                "seed": seed.tolist(),
                "optimizer_success": candidate.diagnostics.get(
                    "optimizer_success", False
                ),
                "accepted": candidate.success,
                "residual_norm": candidate.residual_norm,
                "message": candidate.message,
                "final_mass_fraction": candidate.final_mass_fraction,
                "fuel_fraction": candidate.fuel_fraction,
                "delta_v": candidate.delta_v,
            }
        )
        if _candidate_is_better(candidate, best):
            best = candidate
    if best is None:
        raise ValueError(f"no seeds were generated for {mode}")
    best.diagnostics["seed_results"] = seed_results
    return best


def _lawden_rhs(
    _time: float,
    joint: Array,
    gamma: float,
    kappa: float,
    epsilon: float,
) -> Array:
    (
        rho,
        radial_velocity,
        tangential_velocity,
        eta,
        lambda_rho,
        p_r,
        p_t,
        lambda_mass,
    ) = joint
    if rho <= 0.0 or eta <= 0.0:
        return np.full(8, np.nan)
    primer_length = math.hypot(p_r, p_t)
    if primer_length <= 1e-14:
        return np.full(8, np.nan)
    switching = primer_length / eta + lambda_mass / kappa
    throttle = float(expit(switching / epsilon))
    alpha = math.atan2(p_t, p_r)
    acceleration = throttle * gamma / eta
    return np.array(
        [
            radial_velocity,
            tangential_velocity**2 / rho
            - 1.0 / rho**2
            + acceleration * math.cos(alpha),
            -radial_velocity * tangential_velocity / rho
            + acceleration * math.sin(alpha),
            -throttle * gamma / kappa,
            p_r * (2.0 / rho**3 - tangential_velocity**2 / rho**2)
            + p_t * radial_velocity * tangential_velocity / rho**2,
            lambda_rho + p_t * tangential_velocity / rho,
            -2.0 * p_r * tangential_velocity / rho + p_t * radial_velocity / rho,
            -throttle * gamma * primer_length / eta**2,
        ],
        dtype=float,
    )


def _lawden_initial_joint(problem: Problem, parameters: Array) -> Array:
    alpha, _final_tau, lambda_rho, lambda_mass = parameters
    return np.concatenate(
        (
            np.array([problem.radius, *problem.velocity, 1.0], dtype=float),
            np.array(
                [lambda_rho, math.cos(alpha), math.sin(alpha), lambda_mass],
                dtype=float,
            ),
        )
    )


def _integrate_lawden_smooth(
    problem: Problem,
    parameters: Array,
    epsilon: float,
    dense_output: bool = False,
) -> Any:
    solution = solve_ivp(
        _lawden_rhs,
        (0.0, float(parameters[1])),
        _lawden_initial_joint(problem, parameters),
        args=(problem.stage.gamma, problem.stage.kappa, epsilon),
        method="DOP853",
        rtol=INTEGRATION_RTOL,
        atol=INTEGRATION_ATOL,
        dense_output=dense_output,
    )
    if not solution.success or not np.all(np.isfinite(solution.y[:, -1])):
        raise ValueError(solution.message)
    return solution


def _lawden_switch(joint: Array, kappa: float) -> float:
    return float(
        math.hypot(float(joint[5]), float(joint[6])) / float(joint[3])
        + float(joint[7]) / kappa
    )


def _lawden_residual(problem: Problem, parameters: Array, epsilon: float) -> Array:
    try:
        solution = _integrate_lawden_smooth(problem, parameters, epsilon)
        final = np.asarray(solution.y[:, -1], dtype=float)
        return np.array(
            [
                final[0] - 1.0,
                final[1],
                final[2] - 1.0,
                _lawden_switch(final, problem.stage.kappa),
            ],
            dtype=float,
        )
    except (FloatingPointError, ValueError, ZeroDivisionError):
        return np.full(4, INVALID_RESIDUAL)


def _lawden_profile(
    problem: Problem,
    parameters: Array,
    epsilon: float,
) -> tuple[float, float, float]:
    solution = _integrate_lawden_smooth(problem, parameters, epsilon, True)
    if solution.sol is None:
        raise ValueError("dense output was not created")
    samples = np.linspace(0.0, float(parameters[1]), 401)
    joints = np.asarray(solution.sol(samples).T, dtype=float)
    switches = np.array(
        [_lawden_switch(joint, problem.stage.kappa) for joint in joints], dtype=float
    )
    throttles = expit(switches / epsilon)
    return (
        float(np.min(throttles)),
        float(np.max(throttles)),
        float(1.0 - joints[-1, 3]),
    )


def _lawden_hard_rhs(
    _time: float,
    joint: Array,
    gamma: float,
    kappa: float,
    throttle: float,
) -> Array:
    (
        rho,
        radial_velocity,
        tangential_velocity,
        eta,
        lambda_rho,
        p_r,
        p_t,
        _lambda_mass,
    ) = joint
    if rho <= 0.0 or eta <= 0.0:
        return np.full(8, np.nan)
    primer_length = math.hypot(p_r, p_t)
    if primer_length <= 1e-14:
        return np.full(8, np.nan)
    alpha = math.atan2(p_t, p_r)
    acceleration = throttle * gamma / eta
    return np.array(
        [
            radial_velocity,
            tangential_velocity**2 / rho
            - 1.0 / rho**2
            + acceleration * math.cos(alpha),
            -radial_velocity * tangential_velocity / rho
            + acceleration * math.sin(alpha),
            -throttle * gamma / kappa,
            p_r * (2.0 / rho**3 - tangential_velocity**2 / rho**2)
            + p_t * radial_velocity * tangential_velocity / rho**2,
            lambda_rho + p_t * tangential_velocity / rho,
            -2.0 * p_r * tangential_velocity / rho + p_t * radial_velocity / rho,
            -throttle * gamma * primer_length / eta**2,
        ],
        dtype=float,
    )


def _make_lawden_phase(
    tau: Array,
    joint: Array,
    throttle: float,
    problem: Problem,
) -> PhaseData:
    eta = joint[:, 3]
    switching = np.array(
        [_lawden_switch(value, problem.stage.kappa) for value in joint], dtype=float
    )
    alpha = np.arctan2(joint[:, 6], joint[:, 5])
    throttle_values = np.full(len(tau), throttle, dtype=float)
    acceleration = throttle_values * problem.stage.gamma / eta
    return PhaseData(
        kind="burn" if throttle > 0.5 else "coast",
        tau=tau,
        joint=joint,
        eta=eta,
        throttle=throttle_values,
        switching=switching,
        alpha=np.asarray(alpha, dtype=float),
        acceleration=np.asarray(acceleration, dtype=float),
        lawden=True,
    )


def _propagate_lawden_hard(
    problem: Problem,
    parameters: Array,
    collect: bool,
) -> tuple[Array, list[PhaseData], list[float], float]:
    final_tau = float(parameters[1])
    joint = _lawden_initial_joint(problem, parameters)
    time_tau = 0.0
    throttle = 1.0 if _lawden_switch(joint, problem.stage.kappa) > 0.0 else 0.0
    phases: list[PhaseData] = []
    switches: list[float] = []
    powered_tau = 0.0
    last_kind: PhaseKind | None = None
    last_start_tau = 0.0
    last_start_joint: Array | None = None
    for _index in range(12):
        if time_tau >= final_tau - 1e-12:
            break
        segment_start = time_tau
        direction = -1.0 if throttle > 0.5 else 1.0

        def event(
            event_time: float,
            event_joint: Array,
            *_args: float,
            segment_start: float = segment_start,
            direction: float = direction,
        ) -> float:
            if event_time <= segment_start + 1e-9:
                return -direction * 1e-10
            return _lawden_switch(event_joint, problem.stage.kappa)

        event.terminal = True  # type: ignore[attr-defined]
        event.direction = direction  # type: ignore[attr-defined]
        solution = solve_ivp(
            _lawden_hard_rhs,
            (time_tau, final_tau),
            joint,
            args=(problem.stage.gamma, problem.stage.kappa, throttle),
            events=event,
            method="DOP853",
            rtol=INTEGRATION_RTOL,
            atol=INTEGRATION_ATOL,
            dense_output=True,
        )
        if not solution.success or not np.all(np.isfinite(solution.y[:, -1])):
            raise ValueError(solution.message)
        end_tau = float(solution.t[-1])
        duration = end_tau - segment_start
        powered_tau += throttle * duration
        last_kind = "burn" if throttle > 0.5 else "coast"
        last_start_tau = segment_start
        last_start_joint = joint.copy()
        if collect and duration > 0.0:
            phases.append(
                _make_lawden_phase(
                    np.asarray(solution.t, dtype=float),
                    np.asarray(solution.y.T, dtype=float),
                    throttle,
                    problem,
                )
            )
        joint = np.asarray(solution.y[:, -1], dtype=float)
        time_tau = end_tau
        if solution.t_events is None or len(solution.t_events[0]) == 0:
            break
        switches.append(time_tau)
        throttle = 1.0 - throttle
    else:
        raise ValueError("too many Lawden switching events")
    if last_kind == "coast" and last_start_joint is not None:
        # A final coast cannot change a circular target or the objective.
        joint = last_start_joint
        time_tau = last_start_tau
        if phases and phases[-1].kind == "coast":
            phases.pop()
    return joint, phases, switches, powered_tau


def _lawden_hard_residual(problem: Problem, parameters: Array) -> Array:
    try:
        final, _phases, _switches, _powered = _propagate_lawden_hard(
            problem, parameters, collect=False
        )
        return np.array(
            [
                final[0] - 1.0,
                final[1],
                final[2] - 1.0,
                _lawden_switch(final, problem.stage.kappa),
            ],
            dtype=float,
        )
    except (FloatingPointError, ValueError, ZeroDivisionError):
        return np.full(4, INVALID_RESIDUAL)


def _solve_lawden(problem: Problem, explicit: Candidate) -> Candidate:
    if explicit.success and len(explicit.parameters) >= 6:
        p_r = float(explicit.parameters[1])
        p_t = float(explicit.parameters[2])
        alpha = math.atan2(p_t, p_r)
        final_tau = explicit.final_tau
        lambda_rho = float(explicit.parameters[0])
    else:
        direction = problem.velocity / _norm(problem.velocity)
        alpha = math.atan2(float(direction[1]), float(direction[0]))
        final_tau = min(problem.time_limit * 0.8, problem.stage.max_burn_tau)
        lambda_rho = _lambda_rho_seed(problem.radius, problem.velocity, direction)
    seed = np.array(
        [alpha, max(1e-4, final_tau), lambda_rho, -problem.stage.kappa], dtype=float
    )
    lower = np.array([-math.pi, 1e-5, -100.0, -100.0], dtype=float)
    upper = np.array([math.pi, problem.time_limit, 100.0, 100.0], dtype=float)
    epsilon_values = (1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003)
    history: list[dict[str, object]] = []
    current = np.clip(seed, lower + 1e-8, upper - 1e-8)
    for epsilon in epsilon_values:
        try:
            result = cast(Any, least_squares)(
                lambda values, epsilon=epsilon: _lawden_residual(
                    problem, values, epsilon
                ),
                current,
                bounds=(lower, upper),
                method="trf",
                x_scale="jac",
                ftol=2e-9,
                xtol=2e-9,
                gtol=2e-9,
                max_nfev=180,
            )
            current = np.asarray(result.x, dtype=float)
            residual = _lawden_residual(problem, current, epsilon)
            q_min, q_max, fuel_fraction = _lawden_profile(problem, current, epsilon)
            history.append(
                {
                    "epsilon": epsilon,
                    "nfev": int(result.nfev),
                    "residual_norm": float(np.linalg.norm(residual)),
                    "q_min": q_min,
                    "q_max": q_max,
                    "fuel_fraction": fuel_fraction,
                }
            )
            if q_min < 0.01 and q_max > 0.99:
                break
        except (FloatingPointError, ValueError, ZeroDivisionError) as error:
            history.append({"epsilon": epsilon, "error": str(error)})
            break

    try:
        polish = cast(Any, least_squares)(
            lambda values: _lawden_hard_residual(problem, values),
            current,
            bounds=(lower, upper),
            method="trf",
            x_scale="jac",
            ftol=2e-9,
            xtol=2e-9,
            gtol=2e-9,
            max_nfev=300,
        )
        parameters = np.asarray(polish.x, dtype=float)
        residual = _lawden_hard_residual(problem, parameters)
        final, phases, _switches, powered_tau = _propagate_lawden_hard(
            problem, parameters, collect=True
        )
        final_eta = float(final[3])
        final_tau = float(phases[-1].tau[-1]) if phases else 0.0
        sequence = _phase_sequence(phases)
        short_phase = any(
            phase.duration_tau < problem.min_phase_duration - 1e-12 for phase in phases
        )
        unsupported_sequence = sequence not in SUPPORTED_LAWDEN_SEQUENCES
        valid = bool(
            np.linalg.norm(residual) < RESIDUAL_TOL
            and not short_phase
            and not unsupported_sequence
            and all(
                (
                    float(np.max(phase.switching)) >= -SWITCH_TOL
                    if phase.kind == "burn"
                    else float(np.min(phase.switching)) <= SWITCH_TOL
                )
                for phase in phases
            )
        )
        message = str(polish.message)
        optimizer_success = bool(polish.success)
        if unsupported_sequence:
            message = (
                f"unsupported Lawden phase sequence {_sequence_text(sequence)}; "
                + message
            )
        elif short_phase:
            message = (
                f"Lawden phase is shorter than the implementation limit; {message}"
            )
        nfev = int(polish.nfev)
    except (FloatingPointError, ValueError, ZeroDivisionError) as error:
        parameters = current
        residual = _lawden_residual(problem, current, epsilon_values[-1])
        final = None
        phases = []
        powered_tau = 0.0
        final_eta = None
        final_tau = float(current[1])
        sequence = ()
        unsupported_sequence = False
        short_phase = False
        valid = False
        message = f"hard Lawden polish failed: {error}"
        optimizer_success = False
        nfev = 0
    limit_hit = powered_tau * problem.physical.time_scale_s >= (
        problem.physical.max_burn_time_s - LIMIT_TOLERANCE_SECONDS
    )
    return Candidate(
        name="lawden",
        formulation="lawden-sigmoid-hard-polish",
        success=valid,
        message=message,
        parameters=parameters,
        residual=np.asarray(residual, dtype=float),
        final_state=(
            None
            if final is None
            else np.asarray([final[0], final[1], final[2], final[3]], dtype=float)
        ),
        final_eta=final_eta,
        phases=phases,
        powered_tau=float(powered_tau),
        final_tau=final_tau,
        limit_hit=limit_hit,
        sequence=sequence,
        diagnostics={
            "nfev": nfev,
            "kappa": problem.stage.kappa,
            "epsilon_history": history,
            "optimizer_success": optimizer_success,
            "phase_sequence": _sequence_text(sequence),
            "unsupported_sequence": unsupported_sequence,
            "short_phase": short_phase,
        },
    )


def _integrate_theta(
    tau: Array, rho: Array, tangential_velocity: Array, theta0: float
) -> Array:
    theta = np.empty(len(tau), dtype=float)
    theta[0] = theta0
    angular_rate = tangential_velocity / rho
    if len(tau) > 1:
        theta[1:] = theta0 + np.cumsum(
            0.5 * (angular_rate[:-1] + angular_rate[1:]) * np.diff(tau)
        )
    return theta


def kerbin_examples() -> dict[str, tuple[Array, Array, list[ExampleSegment]]]:
    """Return the three supplied Kerbin examples."""
    return {
        "kerbin-example": (
            np.array([424370.58766631, -1093.08696926, -470992.64951719]),
            np.array([723.81414935, -1.2033429, -122.60883836]),
            [ExampleSegment(215000.0, 3138.128, 13885.650390625)],
        ),
        "kerbin-first-example": (
            np.array([428392.15435586, -1053.61873734, -455905.93323801]),
            np.array([1030.31015, -0.932270447, -119.588146]),
            [ExampleSegment(215000.0, 3138.128, 13057.14453125)],
        ),
        "kerbin-coast-first-example": (
            np.array([433284.5917063, -704.8282711, -459791.995176]),
            np.array([1323.15860984, 11.49193645, 135.66254872]),
            [ExampleSegment(215000.0, 3138.128, 11777.2275390625)],
        ),
    }


def _empty_candidate(problem: Problem) -> Candidate:
    initial_state = np.array([problem.radius, *problem.velocity], dtype=float)
    residual = initial_state - problem.target_state
    return Candidate(
        name="empty",
        formulation="empty",
        success=True,
        message="input state already satisfies the target orbit",
        parameters=np.empty(0, dtype=float),
        residual=np.asarray(residual, dtype=float),
        final_state=np.array([*initial_state, 1.0], dtype=float),
        final_eta=1.0,
        phases=[],
        powered_tau=0.0,
        final_tau=0.0,
        limit_hit=False,
        sequence=(),
        diagnostics={
            "kappa": problem.stage.kappa,
            "phase_sequence": "empty",
        },
    )


def find_oracle(
    position: object,
    velocity: object,
    segments: Sequence[SegmentLike],
    *,
    mu: float = DEFAULT_MU,
    target_radius: float = DEFAULT_TARGET_RADIUS,
    max_burn_time: float = DEFAULT_MAX_BURN_TIME,
) -> OracleResult:
    """Find the highest-final-mass valid single-stage trajectory."""
    started = time.perf_counter()
    problem = _prepare_problem(
        _as_vector(position, "position"),
        _as_vector(velocity, "velocity"),
        segments,
        mu,
        target_radius,
        max_burn_time,
    )
    initial_state = np.array([problem.radius, *problem.velocity], dtype=float)
    if _norm(initial_state - problem.target_state) < RESIDUAL_TOL:
        empty = _empty_candidate(problem)
        return OracleResult(
            problem=problem,
            selected=empty,
            candidates=[empty],
            warnings=[],
            elapsed_seconds=time.perf_counter() - started,
        )
    timing = _impulse_timing_seed(problem)
    explicit_candidates = [
        _solve_explicit_mode(problem, "burn_coast_burn", timing),
        _solve_explicit_mode(problem, "coast_burn", timing),
        _solve_explicit_mode(problem, "burn_only", timing),
    ]
    explicit_seed = explicit_candidates[0]
    for candidate in explicit_candidates[1:]:
        if _candidate_is_better(candidate, explicit_seed):
            explicit_seed = candidate
    lawden = _solve_lawden(problem, explicit_seed)
    candidates = [lawden, *explicit_candidates]
    accepted = [candidate for candidate in candidates if candidate.success]
    if not accepted:
        details = "; ".join(
            f"{candidate.name}: residual={candidate.residual_norm:.3e}"
            for candidate in candidates
        )
        raise ValueError(f"all oracle candidates failed: {details}")
    selected = accepted[0]
    for candidate in accepted[1:]:
        if _candidate_is_better(candidate, selected):
            selected = candidate
    result_warnings: list[str] = []
    for candidate in candidates:
        if candidate.formulation.startswith("lawden") and bool(
            candidate.diagnostics.get("unsupported_sequence", False)
        ):
            message = (
                "Lawden candidate rejected unsupported phase sequence "
                f"{candidate.diagnostics.get('phase_sequence', 'unknown')}"
            )
            result_warnings.append(message)
            warnings.warn(message, RuntimeWarning, stacklevel=2)
        if candidate.limit_hit:
            message = f"{candidate.name} reached max_burn_time={max_burn_time:.6g} s"
            result_warnings.append(message)
            warnings.warn(message, RuntimeWarning, stacklevel=2)
    return OracleResult(
        problem=problem,
        selected=selected,
        candidates=candidates,
        warnings=result_warnings,
        elapsed_seconds=time.perf_counter() - started,
    )


def _run_cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example", action="append", choices=tuple(kerbin_examples()))
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--max-burn-time", type=float, default=DEFAULT_MAX_BURN_TIME)
    parser.add_argument("--target-radius", type=float, default=DEFAULT_TARGET_RADIUS)
    arguments = parser.parse_args()
    examples = kerbin_examples()
    names = arguments.example or list(examples)
    output: list[dict[str, object]] = []
    for name in names:
        position, velocity, segments = examples[name]
        result = find_oracle(
            position,
            velocity,
            segments,
            target_radius=arguments.target_radius,
            max_burn_time=arguments.max_burn_time,
        )
        selected = result.selected
        fuel = selected.fuel_consumed_kg(result.problem)
        delta_v = selected.delta_v_m_per_s(result.problem)
        if fuel is None or delta_v is None:
            raise RuntimeError(f"{name}: selected candidate has no mass metrics")
        print(
            f"{name:28s} {selected.name:18s} "
            f"fuel={fuel:.6f} kg "
            f"dv={delta_v:.6f} m/s "
            f"res={selected.residual_norm:.3e} "
            f"time={selected.final_tau * result.problem.physical.time_scale_s:.3f} s"
        )
        for message in result.warnings:
            print(f"WARNING: {message}")
        output.append({"example": name, **result.to_dict()})
    if arguments.output:
        with open(arguments.output, "w", encoding="utf-8") as output_file:
            json.dump(output, output_file, indent=2)
            output_file.write("\n")


if __name__ == "__main__":
    _run_cli()
