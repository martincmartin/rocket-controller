#!/usr/bin/env python3

"""
Trajectory planning and orbital simulation.

This module contains the physics and optimization code used to plan orbital
maneuvers. It models rocket dynamics, propagates coasting and powered flight,
and solves for guidance parameters that achieve a desired orbit.

The simulator is intentionally independent of Kerbal Space Program and kRPC.
It operates only on physical quantities (position, velocity, mass, thrust,
specific impulse, gravitational parameter, etc.) and has no knowledge of game
objects or telemetry interfaces.

Responsibilities
----------------
- Propagate ballistic and powered trajectories with SciPy ODE solvers.
- Model multi-stage rockets as a sequence of constant-thrust burn segments.
- Compute orbital elements from propagated states.
- Optimize pluggable thrust profiles for orbit insertion and circularization.

This module computes flight plans. Executing those plans in KSP is the
responsibility of the flight-control layer.
"""

import functools
import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar, cast

import numpy as np
from numpy.typing import NDArray
from pydantic import ConfigDict, validate_call
from scipy.integrate import solve_ivp
from scipy.integrate._ivp.ivp import OdeResult
from scipy.optimize import NonlinearConstraint, minimize

from timing import TimingContext

# Type alias for float64 arrays
Vector = NDArray[np.float64]

_validate = validate_call(config=ConfigDict(arbitrary_types_allowed=True))


# Making this np.inf leads to np.inf - np.inf inside scipi, which is Nan.
MAX_ERROR = 1e18


@dataclass
class RocketSegment:
    name: str
    ve: float
    thrust: float
    max_burn_time: float
    initial_mass: float
    # True if a real decouple/staging event (modeled as a coast lasting
    # Simulator.staging_duration seconds) follows this segment before the
    # next one begins. False if the next segment begins immediately (e.g.
    # one engine group in a multi-group segment ran dry while another
    # continues, with no part separation).
    last_segment_of_stage: bool


@dataclass
class OrbitalElements:
    """Orbital elements computed from position and velocity vectors."""

    semi_major_axis: float
    eccentricity: float
    eccentricity_vector: Vector
    angular_momentum: float
    specific_energy: float
    periapsis_radius: float
    apoapsis_radius: float


@dataclass
class BurnResult:
    """Final state after propagating a powered-flight profile."""

    r: Vector
    v: Vector
    mass: float
    # One entry per powered/coast integration call made during this
    # propagation, in chronological order. Each entry is a 2D array of shape
    # (N, 6) for a thrust phase (columns:
    # t, x, y, vx, vy, mass) or (N, 5) for a coast phase (columns:
    # t, x, y, vx, vy) -- i.e. solve_ivp's `t` and `y` zipped together so
    # everything about one timestep lives in one row. Column count (5 vs
    # 6) is how a consumer distinguishes a coast phase from a thrust
    # phase. These are 2D arrays, so plain np.ndarray rather than the
    # Vector alias (reserved for 1D arrays).
    phases: list[np.ndarray[Any, np.dtype[np.float64]]]


@dataclass
class ThrustProfileResult:
    """Result of evaluating a thrust profile for one optimizer candidate."""

    burn_result: BurnResult
    burn_start_time: float
    burn_stop_time: float

    @property
    def burn_time(self) -> float:
        """Elapsed time from the start to the end of the profile."""
        return self.burn_stop_time - self.burn_start_time


class ThrustProfile(ABC):
    """Strategy for parameterizing and propagating a powered-flight profile.

    A profile owns all of the parameters that affect when thrust starts and
    stops, how it is steered, and any coasts it may insert between thrust
    phases.  The outer optimizer only needs the parameter vector, its bounds,
    and the result of evaluating that vector.
    """

    name: str

    @abstractmethod
    def initial_parameters(
        self, sim: "Simulator", coast_time_guess: float, burn_time_guess: float
    ) -> Vector:
        """Return an initial optimizer vector."""

    @abstractmethod
    def parameter_bounds(
        self, sim: "Simulator", coast_bound: float
    ) -> list[tuple[float, float]]:
        """Return bounds for the optimizer vector."""

    @abstractmethod
    def parameter_names(self, sim: "Simulator") -> tuple[str, ...]:
        """Return display names in the same order as the optimizer vector."""

    @abstractmethod
    def evaluate(
        self,
        sim: "Simulator",
        coast_fn: Callable[[float], np.ndarray[Any, Any]],
        ref_angle: float,
        parameters: tuple[float, ...],
    ) -> ThrustProfileResult:
        """Evaluate one candidate profile from the pre-burn coast solution."""

    def objective(self, result: ThrustProfileResult) -> float:
        """Return the quantity minimized by the outer optimizer."""
        return result.burn_time


class LinearTangentProfile(ThrustProfile):
    """One linear-tangent steering law shared by every stage.

    Parameters are ``(burn_start_time, a_coeff, b_coeff, burn_time)``.  The
    burn time includes any mandatory staging coasts between segments.
    """

    name = "single linear-tangent"

    def initial_parameters(
        self, sim: "Simulator", coast_time_guess: float, burn_time_guess: float
    ) -> Vector:
        return np.array([coast_time_guess, 0.0, 0.0, burn_time_guess])

    def parameter_bounds(
        self, sim: "Simulator", coast_bound: float
    ) -> list[tuple[float, float]]:
        return [
            (0.0, coast_bound),
            (-5.0, 5.0),
            (-1.0, 1.0),
            (0.0, sim.total_burn_budget()),
        ]

    def parameter_names(self, sim: "Simulator") -> tuple[str, ...]:
        return ("burn start", "a coefficient", "b coefficient", "burn time")

    def evaluate(
        self,
        sim: "Simulator",
        coast_fn: Callable[[float], np.ndarray[Any, Any]],
        ref_angle: float,
        parameters: tuple[float, ...],
    ) -> ThrustProfileResult:
        burn_start_time, a_coeff, b_coeff, burn_time = parameters
        r, v = to_rv(cast(Vector, coast_fn(burn_start_time)))
        result = sim.propagate_linear_tangent(
            r, v, a_coeff, b_coeff, ref_angle, burn_time
        )
        return ThrustProfileResult(
            result,
            float(burn_start_time),
            float(burn_start_time + burn_time),
        )


class PerStageLinearTangentProfile(ThrustProfile):
    """A separate linear-tangent steering law for each stage.

    Parameters are ``(burn_start_time, a1, b1, ..., aN, bN, burn_time)``.
    ``N`` is inferred from the simulator's segment staging boundaries.
    """

    name = "linear-tangent per stage"

    @staticmethod
    def _stage_count(sim: "Simulator") -> int:
        return 1 + sum(segment.last_segment_of_stage for segment in sim.segments[:-1])

    def _unpack(
        self, sim: "Simulator", parameters: tuple[float, ...]
    ) -> tuple[float, tuple[tuple[float, float], ...], float]:
        stage_count = self._stage_count(sim)
        expected_length = 2 + 2 * stage_count
        if len(parameters) != expected_length:
            raise ValueError(
                f"expected {expected_length} profile parameters for "
                f"{stage_count} stages, got {len(parameters)}"
            )

        burn_start_time = parameters[0]
        coefficients = tuple(
            (parameters[1 + 2 * i], parameters[2 + 2 * i]) for i in range(stage_count)
        )
        burn_time = parameters[-1]
        return burn_start_time, coefficients, burn_time

    def initial_parameters(
        self, sim: "Simulator", coast_time_guess: float, burn_time_guess: float
    ) -> Vector:
        parameters: list[float] = [coast_time_guess]
        for _ in range(self._stage_count(sim)):
            parameters.extend((-0.7, 0.012))
        parameters.append(burn_time_guess)
        return np.array(parameters)

    def parameter_bounds(
        self, sim: "Simulator", coast_bound: float
    ) -> list[tuple[float, float]]:
        bounds: list[tuple[float, float]] = [(0.0, coast_bound)]
        for _ in range(self._stage_count(sim)):
            bounds.extend(((-5.0, 5.0), (-1.0, 1.0)))
        bounds.append((0.0, sim.total_burn_budget()))
        return bounds

    def parameter_names(self, sim: "Simulator") -> tuple[str, ...]:
        names: list[str] = ["burn start"]
        for stage_index in range(self._stage_count(sim)):
            names.extend(
                (
                    f"a coefficient stage {stage_index + 1}",
                    f"b coefficient stage {stage_index + 1}",
                )
            )
        names.append("burn time")
        return tuple(names)

    def evaluate(
        self,
        sim: "Simulator",
        coast_fn: Callable[[float], np.ndarray[Any, Any]],
        ref_angle: float,
        parameters: tuple[float, ...],
    ) -> ThrustProfileResult:
        burn_start_time, coefficients, burn_time = self._unpack(sim, parameters)
        r, v = to_rv(cast(Vector, coast_fn(burn_start_time)))

        def coefficients_for_segment(
            _segment_index: int, stage_index: int
        ) -> tuple[float, float]:
            return coefficients[stage_index]

        result = sim._propagate_linear_tangent_segments(
            r,
            v,
            ref_angle,
            burn_time,
            coefficients_for_segment,
        )
        return ThrustProfileResult(
            result,
            float(burn_start_time),
            float(burn_start_time + burn_time),
        )


class OrbitalPlane:
    """A fixed 2D plane within 3D space, spanned by orthonormal unit vectors
    r_hat (radial direction) and w_hat (in-plane tangential direction),
    used to represent in-plane orbital motion as 2D vectors."""

    @_validate
    def __init__(self, r: Vector, v: Vector) -> None:
        """Build the (r_hat, w_hat) basis for the orbital plane containing
        the 3D position `r` and velocity `v`."""
        # r = 0 means at the center of the body; since we're above the
        # surface, r should never be close to zero, so we can divide with
        # confidence.
        r_norm = np.linalg.norm(r)
        r_hat = r / r_norm

        v_dot_r_hat = np.dot(v, r_hat)

        w = v - v_dot_r_hat * r_hat
        w_norm = np.linalg.norm(w)
        # If r and v are nearly parallel, we can clean things up a bit by
        # doing "twice is enough" re-orthogonalization, if
        # norm(w) < 1e-4*norm(v).
        if w_norm < 1e-4 * np.linalg.norm(v):
            w = w - np.dot(w, r_hat) * r_hat
            w_norm = np.linalg.norm(w)

        # Should probably check that norm(w) isn't near zero, that happens
        # when the rocket is going straight up and velocity is parallel to
        # position.  Oh well.
        w_hat = w / w_norm

        self.r_hat = r_hat
        self.w_hat = w_hat

    def __repr__(self) -> str:
        return f"OrbitalPlane(r_hat={self.r_hat!r}, w_hat={self.w_hat!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, OrbitalPlane):
            return NotImplemented
        return bool(
            np.array_equal(self.r_hat, other.r_hat)
            and np.array_equal(self.w_hat, other.w_hat)
        )

    @_validate
    def to_plane(self, v3d: Vector) -> Vector:
        """Project a 3D vector onto this plane, returning its 2D
        (r_hat, w_hat) coordinates."""
        return np.array([np.dot(v3d, self.r_hat), np.dot(v3d, self.w_hat)])

    @_validate
    def from_plane(self, v2d: Vector) -> Vector:
        """Expand a 2D (r_hat, w_hat) vector back into 3D."""
        return cast(Vector, v2d[0] * self.r_hat + v2d[1] * self.w_hat)


@dataclass
class CircularizationPlan:
    """Everything needed to execute a planned circularization burn."""

    plane: OrbitalPlane  # orbital-plane basis (r_hat, w_hat) at planning time
    thrust_profile: ThrustProfile
    profile_parameters: tuple[float, ...]
    burn_time: float
    ref_angle: float
    coast_time: float  # predicted seconds from now until burn should start
    staging_apoapsis_altitudes: list[float]
    staging_periapsis_altitudes: list[float]
    final_apoapsis_altitude: float  # predicted altitude (m) after the burn
    final_periapsis_altitude: float  # predicted altitude (m) after the burn
    burn_result: BurnResult  # full propagated trajectory at the optimum


def cross2d(r: Vector, v: Vector) -> float:
    return float(r[0] * v[1] - r[1] * v[0])


def to_rv(state: Vector) -> tuple[Vector, Vector]:
    x, y, vx, vy = state
    return (np.array([x, y]), np.array([vx, vy]))


def to_rvm(state: Vector) -> tuple[Vector, Vector, np.float64]:
    x, y, vx, vy, mass = state
    return (np.array([x, y]), np.array([vx, vy]), mass)


def coast_dynamics(t: float, state: np.ndarray, mu: float) -> list[float]:
    r, v = to_rv(state)

    # This is just a = F/m.
    r_norm = np.linalg.norm(r)

    # Acceleration due to gravity.
    a = -mu / r_norm**3 * r

    return [v[0], v[1], a[0], a[1]]


def linear_tangent_dynamics(
    t: float,
    state: np.ndarray,
    mu: float,
    ve: float,
    thrust: float,
    a_coeff: float,
    b_coeff: float,
    ref_angle: float,
) -> list[float]:
    """Equations of motion with thrust direction given by a linear tangent steering law.

    The thrust angle relative to ref_angle is:
        θ = ref_angle + atan(a_coeff + b_coeff * t)

    Parameters
    ----------
    t : float
        Time [s]
    state : array_like
        [x, y, vx, vy, mass]
    mu : float
        Gravitational parameter [m^3/s^2]
    ve : float
        Effective exhaust velocity [m/s]
    thrust : float
        Thrust magnitude [N]
    a_coeff, b_coeff : float
        Linear tangent coefficients: tan(θ) = a_coeff + b_coeff * t
    ref_angle : float
        Inertial reference angle [rad] for θ = 0 (typically the horizontal
        direction at apoapsis, precomputed before the burn).
    """
    r, v, mass = to_rvm(state)

    r_norm = np.linalg.norm(r)

    a = -mu / r_norm**3 * r

    theta = ref_angle + math.atan(a_coeff + b_coeff * t)
    thrust_dir = np.array([math.cos(theta), math.sin(theta)])
    a += (thrust / mass) * thrust_dir
    mdot = -thrust / ve

    return [v[0], v[1], a[0], a[1], mdot]


@_validate
def orbital_elements(r: Vector, v: Vector, mu: float) -> OrbitalElements:
    """
    Compute orbital elements from 2D position and velocity vectors.

    Parameters
    ----------
    r : array_like, shape (2,)
        Position vector [m]
    v : array_like, shape (2,)
        Velocity vector [m/s]
    mu : float
        Gravitational parameter GM [m^3/s^2]

    Returns
    -------
    dict
        {
            'semi_major_axis': a,
            'eccentricity': e,
            'eccentricity_vector': e_vec,
            'angular_momentum': h,
            'specific_energy': energy,
            'periapsis_radius': rp,
            'apoapsis_radius': ra or np.inf
        }
    """

    r_mag = np.linalg.norm(r)
    v_mag = np.linalg.norm(v)

    # Scalar specific angular momentum (z-component)
    h = cross2d(r, v)

    # Specific orbital energy
    energy = 0.5 * v_mag**2 - mu / r_mag

    # Eccentricity vector: e_vec = (h/mu) * v_perp - r_hat
    v_perp = np.array([v[1], -v[0]])
    e_vec = (h / mu) * v_perp - r / r_mag
    e = np.linalg.norm(e_vec)

    # Semi-major axis
    a = np.inf if np.isclose(energy, 0.0) else -mu / (2 * energy)

    # Periapsis / apoapsis
    if e < 1.0:
        rp = a * (1 - e)
        ra = a * (1 + e)
    else:
        p = h**2 / mu
        rp = p / (1 + e)
        ra = np.float64(np.inf)

    return OrbitalElements(
        semi_major_axis=a,
        eccentricity=e,
        eccentricity_vector=e_vec,
        angular_momentum=h,
        specific_energy=energy,
        periapsis_radius=rp,
        apoapsis_radius=ra,
    )


class Simulator:
    """Orbital simulation."""

    ATOL_THRUST_VECTOR: ClassVar[list[float]] = [
        1.0,  # Position within 1 meter.
        1.0,
        0.001,  # Velocity within 0.001 meters / sec
        0.001,
        0.1,  # Mass within 100 grams
    ]

    ATOL_COAST_VECTOR: ClassVar[list[float]] = ATOL_THRUST_VECTOR[:-1]

    @_validate
    def __init__(
        self,
        mu: float,
        body_radius: float,
        target_altitude: float,
        segments: list[RocketSegment],
        staging_duration: float,
    ) -> None:
        self.mu = mu
        self.body_radius = body_radius
        self.target_radius = body_radius + target_altitude
        self.segments = segments
        self.staging_duration = staging_duration

    @_validate
    def solve_coast(
        self, t_span: tuple[float, float], r: Vector, v: Vector
    ) -> OdeResult:
        return solve_ivp(
            coast_dynamics,
            t_span,
            (r[0], r[1], v[0], v[1]),
            args=(self.mu,),
            rtol=1e-10,
            atol=self.ATOL_COAST_VECTOR,
            dense_output=True,
        )

    @_validate
    def solve_linear_tangent(
        self,
        t_offset: float,
        t_duration: float,
        r: Vector,
        v: Vector,
        segment: RocketSegment,
        a_coeff: float,
        b_coeff: float,
        ref_angle: float,
    ) -> OdeResult:
        assert t_duration <= segment.max_burn_time
        return solve_ivp(
            linear_tangent_dynamics,
            (t_offset, t_offset + t_duration),
            (r[0], r[1], v[0], v[1], segment.initial_mass),
            args=(self.mu, segment.ve, segment.thrust, a_coeff, b_coeff, ref_angle),
            rtol=1e-10,
            atol=self.ATOL_THRUST_VECTOR,
            dense_output=True,
        )

    @_validate
    def total_burn_budget(self) -> float:
        """Total elapsed time available across all remaining segments,
        including a `self.staging_duration` second staging coast after each
        segment whose `last_segment_of_stage` is True (except the last
        segment, which has no following segment to coast into)."""
        n = len(self.segments)
        total = sum(segment.max_burn_time for segment in self.segments)
        total += sum(
            self.staging_duration
            for i, segment in enumerate(self.segments)
            if segment.last_segment_of_stage and i < n - 1
        )
        return total

    @_validate
    def burn_time_for_delta_v(self, delta_v: float) -> float:
        """Estimate the total elapsed time (mirroring the segment/staging-coast
        accounting in `propagate_linear_tangent`) needed to deliver
        `delta_v` of delta-v, assuming each segment fully burns before the
        next segment begins (with a `self.staging_duration` second staging
        coast in between).

        This ignores steering losses (it assumes the full delta-v goes into
        useful velocity change) and is meant only to produce a reasonable
        initial guess for `burn_time`.
        """
        remaining_dv = delta_v
        elapsed = 0.0
        n_segments = len(self.segments)

        for i, segment in enumerate(self.segments):
            mdot = segment.thrust / segment.ve
            m_final = segment.initial_mass - mdot * segment.max_burn_time
            segment_dv = segment.ve * math.log(segment.initial_mass / m_final)

            if remaining_dv <= segment_dv:
                t = (segment.initial_mass * segment.ve / segment.thrust) * (
                    1 - math.exp(-remaining_dv / segment.ve)
                )
                return elapsed + t

            remaining_dv -= segment_dv
            elapsed += segment.max_burn_time

            if segment.last_segment_of_stage and i < n_segments - 1:
                elapsed += self.staging_duration

        # Ran out of segments before delivering the requested delta-v; fall
        # back on the full burn budget as a safe upper bound.
        return self.total_burn_budget()

    def _propagate_linear_tangent_segments(
        self,
        r: Vector,
        v: Vector,
        ref_angle: float,
        burn_time: float,
        coefficients_for_segment: Callable[[int, int], tuple[float, float]],
    ) -> BurnResult:
        """Propagate a linear-tangent burn across all required segments.

        ``coefficients_for_segment`` receives the segment and stage indices
        and returns that segment's ``(a, b)`` coefficients.  Keeping this
        segment/staging bookkeeping here lets each profile choose its own
        coefficient parameterization without duplicating the integrator loop.
        """
        remaining = burn_time
        elapsed = 0.0
        mass = math.nan
        n_segments = len(self.segments)
        phases: list[np.ndarray[Any, np.dtype[np.float64]]] = []
        stage_index = 0

        for i, segment in enumerate(self.segments):
            duration = min(remaining, segment.max_burn_time)
            a_coeff, b_coeff = coefficients_for_segment(i, stage_index)
            solution = self.solve_linear_tangent(
                elapsed,
                duration,
                r,
                v,
                segment,
                a_coeff,
                b_coeff,
                ref_angle,
            )
            assert solution.sol is not None
            phases.append(
                cast(
                    "np.ndarray[Any, np.dtype[np.float64]]",
                    np.vstack((solution.t, solution.y)).T,
                )
            )

            if remaining <= segment.max_burn_time:
                assert solution.t[-1] == elapsed + duration
                r, v, mass = to_rvm(cast(Vector, solution.y[:, -1]))
                return BurnResult(r, v, mass, phases)

            # Fully burn this segment.
            remaining -= segment.max_burn_time
            elapsed += segment.max_burn_time
            # Cast needed because y is type ndarray[float64 | complex128]
            r, v, mass = to_rvm(cast(Vector, solution.y[:, -1]))

            if segment.last_segment_of_stage and i < n_segments - 1:
                # Simulate staging as a `self.staging_duration` second
                # coast, which also counts against the requested burn_time
                # budget. Clamp to `remaining` in case the deadline falls
                # inside this window.
                staging_duration = min(self.staging_duration, remaining)
                coast = self.solve_coast((0, staging_duration), r, v)
                phases.append(
                    cast(
                        "np.ndarray[Any, np.dtype[np.float64]]",
                        np.vstack((coast.t, coast.y)).T,
                    )
                )
                # Cast needed because y is type ndarray[float64 | complex128]
                r, v = to_rv(cast(Vector, coast.y[:, -1]))
                elapsed += staging_duration
                remaining -= staging_duration

                if remaining <= 0:
                    return BurnResult(r, v, mass, phases)

                if segment.last_segment_of_stage:
                    stage_index += 1

        # Ran out of segments before using all of the requested burn time.
        return BurnResult(r, v, mass, phases)

    @_validate
    def propagate_linear_tangent(
        self,
        r: Vector,
        v: Vector,
        a_coeff: float,
        b_coeff: float,
        ref_angle: float,
        burn_time: float,
    ) -> BurnResult:
        """Propagate one linear-tangent law shared by every stage."""

        def shared_coefficients(
            _segment_index: int, _stage_index: int
        ) -> tuple[float, float]:
            return a_coeff, b_coeff

        return self._propagate_linear_tangent_segments(
            r,
            v,
            ref_angle,
            burn_time,
            shared_coefficients,
        )

    @staticmethod
    def prograde_at_apoapsis(orbit: OrbitalElements) -> float:
        # Eccentricity vector points to periapsis, so apoapsis direction is [-ex, -ey].
        # Prograde is perpendicular to that, with sign determined by angular momentum.
        # For h>0 (CCW): rotate apoapsis direction 90° CCW: [ey, -ex]
        # For h<0 (CW):  rotate apoapsis direction 90° CW:  [-ey, ex]
        ex, ey = orbit.eccentricity_vector
        h = orbit.angular_momentum

        if h > 0:
            prograde_x = ey
            prograde_y = -ex
        else:
            prograde_x = -ey
            prograde_y = ex

        return math.atan2(prograde_y, prograde_x)

    @_validate
    def target_velocity(self, r: Vector, v: Vector) -> Vector:
        """Compute the 2D velocity vector for a circular orbit at position r.

        For a circular orbit, the velocity is perpendicular to the position vector.
        This method returns the perpendicular velocity direction aligned with the
        current velocity vector v.

        Parameters
        ----------
        r : array_like, shape (2,)
            Position vector [m]
        v : array_like, shape (2,)
            Current velocity vector [m/s] - used to determine direction

        Returns
        -------
        Vector
            Velocity vector [m/s] for a circular orbit
        """
        r_mag = np.linalg.norm(r)
        v_mag = np.sqrt(self.mu / r_mag)
        # Perpendicular to r in counterclockwise direction (90° rotation)
        v_direction = np.array([-r[1], r[0]]) / r_mag
        # Flip direction if it points opposite to current velocity
        if np.dot(v_direction, v) < 0:
            v_direction = -v_direction
        return cast(Vector, v_mag * v_direction)

    @_validate
    def find_thrust_profile_params(
        self,
        r3d: Vector,
        v3d: Vector,
        time_to_apoapsis: float,
        thrust_profile: ThrustProfile,
        verbose: bool = True,
    ) -> CircularizationPlan:
        """Find the parameters for ``thrust_profile`` that circularize orbit.

        The profile supplies the optimizer vector, its bounds, and the full
        powered-flight evaluation.  This method only owns the orbital
        constraints shared by all profiles.
        """
        with TimingContext(
            label="find_thrust_profile_params", auto_print=False
        ) as timer:
            plane = OrbitalPlane(r3d, v3d)
            r = plane.to_plane(r3d)
            v = plane.to_plane(v3d)

            # Find the prograde direction at apoapsis
            orbit = orbital_elements(r, v, self.mu)
            ref_angle = self.prograde_at_apoapsis(orbit)

            # Our goal is to raise periapsis to get us into orbit.  So periapsis
            # should be below target or someone is very confused.
            assert orbit.periapsis_radius < self.target_radius

            # Simulate coasting (no thrust) up until apoapsis.  We know we need to burn
            # before apoapsis, so that's a good upper bound on when to start burning.
            sol = self.solve_coast((0, time_to_apoapsis), r, v)
            assert sol.sol is not None
            coast_fn = sol.sol

            # Estimate initial coast_time / burn_time by computing the
            # delta-v needed at the pre-burn apoapsis to raise the far apsis
            # (which may switch from periapsis to apoapsis) to the target
            # radius, then converting that delta-v to a burn time assuming
            # each segment fully burns before the next (with staging
            # coasts in between).
            a_before = orbit.semi_major_axis
            ra_before = orbit.apoapsis_radius
            v1 = math.sqrt(self.mu * (2 / ra_before - 1 / a_before))
            a_new = (ra_before + self.target_radius) / 2
            v_new = math.sqrt(self.mu * (2 / ra_before - 1 / a_new))
            delta_v = v_new - v1
            burn_time_guess = self.burn_time_for_delta_v(delta_v)
            coast_time_guess = min(
                max(time_to_apoapsis - burn_time_guess / 2, 0.0), time_to_apoapsis
            )

            @functools.cache
            def simulate(
                params_tuple: tuple[float, ...],
            ) -> ThrustProfileResult:
                """Evaluate a candidate once for all SLSQP callbacks."""
                return thrust_profile.evaluate(self, coast_fn, ref_angle, params_tuple)

            def objective(params: Vector) -> float:
                return thrust_profile.objective(simulate(tuple(params)))

            def eq_constraint(params: Vector) -> Vector:
                result = simulate(tuple(params)).burn_result
                target_v = self.target_velocity(result.r, result.v)
                return result.v - target_v

            def ineq_constraint(params: Vector) -> Vector:
                result = simulate(tuple(params)).burn_result
                r_norm = np.linalg.norm(result.r)
                return np.array([r_norm - self.target_radius])

            initial_params = thrust_profile.initial_parameters(
                self, coast_time_guess, burn_time_guess
            )
            bounds = thrust_profile.parameter_bounds(self, time_to_apoapsis)
            constraints = [
                NonlinearConstraint(eq_constraint, 0.0, 0.0),
                NonlinearConstraint(ineq_constraint, 0.0, np.inf),
            ]

            res = minimize(
                objective,
                x0=initial_params,
                bounds=bounds,
                constraints=constraints,
                method="SLSQP",
                options={"ftol": 1e-6, "maxiter": 200},
            )

            final_params = tuple(float(value) for value in res.x)
            evaluation = simulate(final_params)
            result = evaluation.burn_result
            final_orbit = orbital_elements(result.r, result.v, self.mu)
            eq_residual = eq_constraint(cast(Vector, res.x))
            ineq_residual = ineq_constraint(cast(Vector, res.x))
            ap_alt = final_orbit.apoapsis_radius - self.body_radius
            pe_alt = final_orbit.periapsis_radius - self.body_radius

            # Orbital elements at the end of each stage's burn: the last
            # row of each thrust phase in the propagated trajectory.
            staging_aps: list[float] = []
            staging_pes: list[float] = []
            for phase in result.phases:
                if phase.shape[1] == 6:
                    x, y, vx, vy = phase[-1][1:5]
                    els = orbital_elements(
                        np.array([x, y]), np.array([vx, vy]), self.mu
                    )
                    staging_aps.append(float(els.apoapsis_radius - self.body_radius))
                    staging_pes.append(float(els.periapsis_radius - self.body_radius))

        # Print summary (outside context so timing is finalized)
        if verbose:
            print(f"\n***** SLSQP {thrust_profile.name} solution")
            for name, value in zip(
                thrust_profile.parameter_names(self), final_params, strict=True
            ):
                print(f"{name + ':':<28}{value:8.6f}")
            print(f"Final vessel mass:   {result.mass:8.2f} kg")
            print(f"Apoapsis:            {ap_alt:8.2f} m")
            print(f"Periapsis:           {pe_alt:8.2f} m")
            print(f"Staging aps:         {staging_aps} m")
            print(f"Staging pes:         {staging_pes} m")
            print(
                f"Velocity residual:  ({eq_residual[0]:8.4f}, "
                f"{eq_residual[1]:8.4f}) m/s"
            )
            print(f"Radius residual:     {ineq_residual[0]:8.2f} m")
            print(f"Optimizer success:   {res.success} ({res.message})")
            print(timer.summary())

        return CircularizationPlan(
            plane=plane,
            thrust_profile=thrust_profile,
            profile_parameters=final_params,
            burn_time=float(evaluation.burn_time),
            ref_angle=float(ref_angle),
            coast_time=float(evaluation.burn_start_time),
            staging_apoapsis_altitudes=staging_aps,
            staging_periapsis_altitudes=staging_pes,
            final_apoapsis_altitude=ap_alt,
            final_periapsis_altitude=pe_alt,
            burn_result=result,
        )

    @_validate
    def find_linear_tangent_params(
        self,
        r3d: Vector,
        v3d: Vector,
        time_to_apoapsis: float,
        verbose: bool = True,
        thrust_profile: ThrustProfile | None = None,
    ) -> CircularizationPlan:
        """Run one of the linear-tangent profile implementations.

        The experiment's historical default is the per-stage profile.  Pass
        ``LinearTangentProfile()`` through ``thrust_profile`` to reproduce
        ``sim.py``'s single-law behavior, or call
        ``find_thrust_profile_params`` for a profile-neutral entry point.
        """
        if thrust_profile is None:
            thrust_profile = PerStageLinearTangentProfile()
        return self.find_thrust_profile_params(
            r3d,
            v3d,
            time_to_apoapsis,
            thrust_profile,
            verbose,
        )


def main() -> None:
    with np.errstate(invalid="raise"):
        TARGET_ALTITUDE = 80_000

        SEGMENTS = [
            RocketSegment(
                name='LV-T45 "Swivel" Liquid Fuel Engine',
                ve=3138.128,
                thrust=215000.0,
                max_burn_time=59.0500960010656,
                initial_mass=13885.650390625,
                last_segment_of_stage=True,
            ),
            RocketSegment(
                name='LV-909 "Terrier" Liquid Fuel Engine',
                ve=3383.29425,
                thrust=60000.0,
                max_burn_time=112.77647255563578,
                initial_mass=4449.999407536325,
                last_segment_of_stage=True,
            ),
        ]

        MU = 3.5316e12
        KERBIN_RADIUS = 600_000

        sim = Simulator(
            MU,
            body_radius=KERBIN_RADIUS,
            target_altitude=TARGET_ALTITUDE,
            segments=SEGMENTS,
            staging_duration=2.0,
        )

        R3D = np.array([424370.58766631, -1093.08696926, -470992.64951719])
        V3D = np.array([723.81414935, -1.2033429, -122.60883836])

        TIME_TO_APOAPSIS = 72.12194913376851

        # Swap this one line to compare the two profile implementations.
        thrust_profile: ThrustProfile = PerStageLinearTangentProfile()
        # thrust_profile = LinearTangentProfile()
        sim.find_thrust_profile_params(R3D, V3D, TIME_TO_APOAPSIS, thrust_profile)


if __name__ == "__main__":
    main()

# Four parameter (coast time, a, b, burn time):
#     - Burn time: 119.67 sec
#     - Final vessel mass: 3392.68 kg
#     - Apoapsis at staging: 83032.30

# Another test case, this one with apoapsis already around 81km:
SEGMENTS = [
    RocketSegment(
        name='LV-T45 "Swivel" Liquid Fuel Engine',
        ve=3138.128,
        thrust=215000.0,
        max_burn_time=28.27566676857025,
        initial_mass=11777.2275390625,
        last_segment_of_stage=True,
    ),
    RocketSegment(
        name='LV-909 "Terrier" Liquid Fuel Engine',
        ve=3383.29425,
        thrust=60000.0,
        max_burn_time=112.77647255563578,
        initial_mass=4449.999885100777,
        last_segment_of_stage=True,
    ),
]
R3D = np.array([433284.5917063, -704.8282711, -459791.995176])
V3D = np.array([1323.15860984, 11.49193645, 135.66254872])
TIME_TO_APOAPSIS = 124.74389992322136


# Key finding: the sharp oracle's S(t) is tiny everywhere — S(0)=2.3e-3, coast dips only
# to -5.75e-5, burn 2 peaks at 1.6e-6. So eps=1e-4 is not small relative to S: the coast
# throttle would be expit(-0.575)≈0.36 even at the exact optimum. Let me continue the
# eps schedule below 1e-4 to see if the smeared root sharpens or persists (singular-arc
# limit), and check the sharp-system residual of the smeared root.
