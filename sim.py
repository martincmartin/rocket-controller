#!/usr/bin/env python3

import functools
import math
import resource
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Optional, cast

import numpy as np
from numpy.typing import NDArray

# Type alias for float64 arrays
Vector = NDArray[np.float64]
from pydantic import ConfigDict, validate_call
from scipy.integrate import solve_ivp
from scipy.integrate._ivp.ivp import OdeResult
from scipy.optimize import NonlinearConstraint, OptimizeResult, minimize

_validate = validate_call(config=ConfigDict(arbitrary_types_allowed=True))


class TimingContext:
    """Context manager for measuring wall clock time, CPU time, and resource usage.

    Captures timing and resource metrics on entry/exit, with optional auto-print.
    Reusable across different methods and scenarios.
    """

    def __init__(self, label: str = "", auto_print: bool = True) -> None:
        """Initialize timing context.

        Parameters
        ----------
        label : str
            Optional label for this timing block (used in output)
        auto_print : bool
            If True, print summary on context exit. Default: True
        """
        self.label = label
        self.auto_print = auto_print

        # Timing metrics
        self.wall_time: float = 0.0
        self.user_time: float = 0.0
        self.system_time: float = 0.0

        # Resource metrics from getrusage
        self.peak_memory_kb: float = 0.0  # ru_maxrss
        self.minor_page_faults: int = 0  # ru_minflt (memory not on disk)
        self.major_page_faults: int = 0  # ru_majflt (memory on disk, required I/O)
        self.voluntary_context_switches: int = 0  # ru_nvcsw (yield/blocking)
        self.involuntary_context_switches: int = 0  # ru_nivcsw (preemption)
        self.input_blocks: int = 0  # ru_inblock
        self.output_blocks: int = 0  # ru_oublock

        # Internal state
        self._start_wall: float = 0.0
        self._start_rusage: Optional[resource.struct_rusage] = None

    def __enter__(self) -> "TimingContext":
        """Start timing."""
        self._start_wall = time.perf_counter()
        self._start_rusage = resource.getrusage(resource.RUSAGE_SELF)
        return self

    def __exit__(self, *args: Any) -> None:
        """Stop timing and optionally print summary."""
        end_wall = time.perf_counter()
        end_rusage = resource.getrusage(resource.RUSAGE_SELF)

        assert self._start_rusage is not None

        # Calculate deltas
        self.wall_time = end_wall - self._start_wall
        self.user_time = end_rusage.ru_utime - self._start_rusage.ru_utime
        self.system_time = end_rusage.ru_stime - self._start_rusage.ru_stime
        self.peak_memory_kb = float(end_rusage.ru_maxrss)
        self.minor_page_faults = end_rusage.ru_minflt - self._start_rusage.ru_minflt
        self.major_page_faults = end_rusage.ru_majflt - self._start_rusage.ru_majflt
        self.voluntary_context_switches = (
            end_rusage.ru_nvcsw - self._start_rusage.ru_nvcsw
        )
        self.involuntary_context_switches = (
            end_rusage.ru_nivcsw - self._start_rusage.ru_nivcsw
        )
        self.input_blocks = end_rusage.ru_inblock - self._start_rusage.ru_inblock
        self.output_blocks = end_rusage.ru_oublock - self._start_rusage.ru_oublock

        if self.auto_print:
            print(self.summary())

    def summary(self) -> str:
        """Return formatted timing and resource summary."""
        lines = []
        if self.label:
            lines.append(f"\n***** Timing: {self.label}")
        else:
            lines.append("\n***** Timing Summary")

        # CPU and wall clock timing
        cpu_total = self.user_time + self.system_time
        cpu_pct = (cpu_total / self.wall_time * 100) if self.wall_time > 0 else 0.0

        lines.append(f"Wall clock time:           {self.wall_time:8.3f} s")
        lines.append(f"User CPU time:             {self.user_time:8.3f} s")
        lines.append(f"System CPU time:           {self.system_time:8.3f} s")
        lines.append(f"Total CPU time:            {cpu_total:8.3f} s ({cpu_pct:5.1f}%)")

        # Memory and page faults
        # On macOS and BSD, ru_maxrss is in bytes; on Linux it's in KB
        if sys.platform == "darwin" or sys.platform.startswith("freebsd"):
            peak_memory_mb = self.peak_memory_kb / (1024 * 1024)  # bytes to MB
        else:  # Linux
            peak_memory_mb = self.peak_memory_kb / 1024  # KB to MB
        lines.append(f"Peak memory:               {peak_memory_mb:8.1f} MB")
        lines.append(f"Minor page faults:         {self.minor_page_faults:8d}")
        lines.append(f"Major page faults:         {self.major_page_faults:8d}")

        # Context switches
        lines.append(
            f"Voluntary context switches: {self.voluntary_context_switches:8d}"
        )
        lines.append(
            f"Involuntary context switches: {self.involuntary_context_switches:8d}"
        )

        # I/O
        lines.append(f"Input blocks (fsync):      {self.input_blocks:8d}")
        lines.append(f"Output blocks (fsync):     {self.output_blocks:8d}")

        return "\n".join(lines)


KERBIN_RADIUS = 600_000

# Making this np.inf leads to np.inf - np.inf inside scipi, which is Nan.
MAX_ERROR = 1e18


@dataclass
class Stage:
    name: str
    ve: float
    thrust: float
    max_burn_time: float
    initial_mass: float


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
    """Final state after propagating a linear-tangent burn."""

    r: Vector
    v: Vector
    mass: float


def cross2d(r: Vector, v: Vector) -> float:
    return float(r[0] * v[1] - r[1] * v[0])


@_validate
def project(r: Vector, v: Vector) -> tuple[Vector, Vector, Vector, Vector]:
    # r = 0 means at the center of the body; since we're above the surface,
    # r should never be close to zero, so we can divide with confidence.
    r_norm = np.linalg.norm(r)
    r_hat = r / r_norm

    v_dot_r_hat = np.dot(v, r_hat)

    w = v - v_dot_r_hat * r_hat
    w_norm = np.linalg.norm(w)
    # If u and v are nearly parallel, we can clean things up a bit by doing
    # "twice is enough re-orthogonalization", if norm(w) < 1e-4*norm(v).
    if w_norm < 1e-4 * np.linalg.norm(v):
        w = w - np.dot(w, r_hat) * r_hat
        w_norm = np.linalg.norm(w)

    # Should probably check that norm(w) isn't near zero, that happens when the rocket
    # is going straight up and velocity is parallel to position.  Oh well.
    w_hat = w / w_norm

    r_projected = np.array([r_norm, 0])
    v_projected = np.array([v_dot_r_hat, np.dot(v, w_hat)])

    return (r_hat, w_hat, r_projected, v_projected)


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
    if np.isclose(energy, 0.0):
        a = np.inf
    else:
        a = -mu / (2 * energy)

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

    ATOL_THRUST_VECTOR = [
        1.0,  # Position within 1 meter.
        1.0,
        0.001,  # Velocity within 0.001 meters / sec
        0.001,
        0.1,  # Mass within 100 grams
    ]

    ATOL_COAST_VECTOR = ATOL_THRUST_VECTOR[:-1]

    @_validate
    def __init__(
        self, mu: float, body_radius: float, target_altitude: float, stages: list[Stage]
    ) -> None:
        self.mu = mu
        self.target_radius = body_radius + target_altitude
        self.stages = stages

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
        stage: Stage,
        a_coeff: float,
        b_coeff: float,
        ref_angle: float,
    ) -> OdeResult:
        assert t_duration <= stage.max_burn_time
        return solve_ivp(
            linear_tangent_dynamics,
            (t_offset, t_offset + t_duration),
            (r[0], r[1], v[0], v[1], stage.initial_mass),
            args=(self.mu, stage.ve, stage.thrust, a_coeff, b_coeff, ref_angle),
            rtol=1e-10,
            atol=self.ATOL_THRUST_VECTOR,
            dense_output=True,
        )

    @_validate
    def total_burn_budget(self) -> float:
        """Total elapsed time available across all remaining stages,
        including the mandatory 1 second staging coast between each pair of
        stages."""
        n = len(self.stages)
        return sum(stage.max_burn_time for stage in self.stages) + max(0, n - 1) * 1.0

    @_validate
    def burn_time_for_delta_v(self, delta_v: float) -> float:
        """Estimate the total elapsed time (mirroring the stage/staging-coast
        accounting in `propagate_linear_tangent`) needed to deliver
        `delta_v` of delta-v, assuming each stage fully burns before the next
        stage begins (with a 1 second staging coast in between).

        This ignores steering losses (it assumes the full delta-v goes into
        useful velocity change) and is meant only to produce a reasonable
        initial guess for `burn_time`.
        """
        remaining_dv = delta_v
        elapsed = 0.0
        n_stages = len(self.stages)

        for i, stage in enumerate(self.stages):
            mdot = stage.thrust / stage.ve
            m_final = stage.initial_mass - mdot * stage.max_burn_time
            stage_dv = stage.ve * math.log(stage.initial_mass / m_final)

            if remaining_dv <= stage_dv:
                t = (stage.initial_mass * stage.ve / stage.thrust) * (
                    1 - math.exp(-remaining_dv / stage.ve)
                )
                return elapsed + t

            remaining_dv -= stage_dv
            elapsed += stage.max_burn_time

            if i < n_stages - 1:
                elapsed += 1.0

        # Ran out of stages before delivering the requested delta-v; fall
        # back on the full burn budget as a safe upper bound.
        return self.total_burn_budget()

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
        """Burn under the linear-tangent steering law for `burn_time` seconds
        total, walking across stage boundaries (with a 1 second staging
        coast between them) as needed.

        `burn_time` is the total elapsed time from the start of the first
        stage, including any 1 second staging coasts along the way. The
        time reference `t` used for the linear-tangent steering law (see
        `linear_tangent_dynamics`) is likewise continuous across stage
        boundaries and staging coasts: it is never reset to 0 at the start
        of a later stage.
        """
        remaining = burn_time
        elapsed = 0.0
        mass = math.nan
        n_stages = len(self.stages)

        for i, stage in enumerate(self.stages):
            duration = min(remaining, stage.max_burn_time)
            solution = self.solve_linear_tangent(
                elapsed,
                duration,
                r,
                v,
                stage,
                a_coeff,
                b_coeff,
                ref_angle,
            )
            assert solution.sol is not None

            if remaining <= stage.max_burn_time:
                assert solution.t[-1] == elapsed + duration
                r, v, mass = to_rvm(cast(Vector, solution.y[:, -1]))
                return BurnResult(r, v, mass)

            # Fully burn this stage.
            remaining -= stage.max_burn_time
            elapsed += stage.max_burn_time
            # Cast needed because y is type ndarray[float64 | complex128]
            r, v, mass = to_rvm(cast(Vector, solution.y[:, -1]))

            if i < n_stages - 1:
                # Simulate staging as a 1 second coast, which also counts
                # against the requested burn_time budget. Clamp to
                # `remaining` in case the deadline falls inside this window.
                staging_duration = min(1.0, remaining)
                coast = self.solve_coast((0, staging_duration), r, v)
                # Cast needed because y is type ndarray[float64 | complex128]
                r, v = to_rv(cast(Vector, coast.y[:, -1]))
                elapsed += staging_duration
                remaining -= staging_duration

                if remaining <= 0:
                    state = np.array([r[0], r[1], v[0], v[1], mass])
                    return BurnResult(r, v, mass)

        # Ran out of stages before using all of the requested burn time.
        return BurnResult(r, v, mass)

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
    def find_linear_tangent_params(
        self,
        r3d: Vector,
        v3d: Vector,
        time_to_apoapsis: float,
    ) -> OptimizeResult:
        """Find linear-tangent steering parameters (coast_time, a_coeff,
        b_coeff, burn_time) that circularize the orbit, using SLSQP with
        explicit constraints:

          - Equality (2 components): velocity at burn end matches the
            circular-orbit velocity for the resulting position (see
            `target_velocity`).
          - Inequality (1 component): final orbital radius is at least
            `self.target_radius`.

        The objective is simply `burn_time`, i.e. find the earliest
        (coast_time, a_coeff, b_coeff, burn_time) combination that satisfies
        both constraints.
        """
        with TimingContext(
            label="find_linear_tangent_params", auto_print=False
        ) as timer:
            r_hat, w_hat, r, v = project(r3d, v3d)

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

            budget = self.total_burn_budget()

            # Estimate initial coast_time / burn_time by computing the
            # delta-v needed at the pre-burn apoapsis to raise the far apsis
            # (which may switch from periapsis to apoapsis) to the target
            # radius, then converting that delta-v to a burn time assuming
            # each stage fully burns before the next (with 1 second staging
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

            @functools.lru_cache(maxsize=None)
            def simulate(params_tuple: tuple[float, ...]) -> BurnResult:
                """Coast to coast_time, then burn for burn_time under the
                linear-tangent steering law.  Memoized so the objective and
                both constraint functions can share results when SLSQP
                evaluates them at the same point."""
                coast_time, a_coeff, b_coeff, burn_time = params_tuple
                r0, v0 = to_rv(coast_fn(coast_time))
                return self.propagate_linear_tangent(
                    r0, v0, a_coeff, b_coeff, ref_angle, burn_time
                )

            def objective(params: Vector) -> float:
                return float(params[3])  # burn_time

            def eq_constraint(params: Vector) -> Vector:
                result = simulate(tuple(params))
                target_v = self.target_velocity(result.r, result.v)
                return result.v - target_v

            def ineq_constraint(params: Vector) -> Vector:
                result = simulate(tuple(params))
                r_norm = np.linalg.norm(result.r)
                return np.array([r_norm - self.target_radius])

            initial_params = np.array([coast_time_guess, 0.0, 0.0, burn_time_guess])
            bounds: list[tuple[float, float]] = [
                (0.0, time_to_apoapsis),
                (-5.0, 5.0),
                (-1.0, 1.0),
                (0.0, budget),
            ]
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

            final_params = cast(Vector, res.x)
            coast_time, a_coeff, b_coeff, burn_time = final_params
            result = simulate(tuple(final_params))
            final_orbit = orbital_elements(result.r, result.v, self.mu)
            eq_residual = eq_constraint(final_params)
            ineq_residual = ineq_constraint(final_params)

        # Print summary (outside context so timing is finalized)
        print("\n***** SLSQP linear-tangent solution")
        print(f"Coast time:          {coast_time:8.2f} s")
        print(f"a coefficient:       {a_coeff:8.6f}")
        print(f"b coefficient:       {b_coeff:8.6f}")
        print(f"Burn time:           {burn_time:8.2f} s")
        ap_alt = final_orbit.apoapsis_radius - KERBIN_RADIUS
        pe_alt = final_orbit.periapsis_radius - KERBIN_RADIUS
        print(f"Apoapsis:            {ap_alt:8.2f} m")
        print(f"Periapsis:           {pe_alt:8.2f} m")
        print(f"Velocity residual:  ({eq_residual[0]:8.4f}, {eq_residual[1]:8.4f}) m/s")
        print(f"Radius residual:     {ineq_residual[0]:8.2f} m")
        print(f"Optimizer success:   {res.success} ({res.message})")
        print(timer.summary())
        return res


def main() -> None:
    with np.errstate(invalid="raise"):
        TARGET_ALTITUDE = 80_000

        # Swivel in vacuum:
        SWIVEL = Stage(
            "Swivel",
            ve=320 * 9.80665,  # m / sec
            thrust=215_000.0,  # Newtons = kg m / sec^2
            max_burn_time=46.95725973451462,
            initial_mass=13057.14453125,
        )

        # Terrier
        TERRIER = Stage(
            "Terrier",
            ve=345 * 9.80665,  # m / sec
            thrust=60_000.0,  # Newtons = kg m / sec^2, flow_rate=17.7341950083118 kg/sec
            max_burn_time=112.77647255563578,
            initial_mass=4450.0,  # 2450 mass after burn?
        )

        MU = 3.5316e12

        sim = Simulator(
            MU,
            body_radius=KERBIN_RADIUS,
            target_altitude=TARGET_ALTITUDE,
            stages=[SWIVEL, TERRIER],
        )

        R3D = np.array([428392.15435586, -1053.61873734, -455905.93323801])
        V3D = np.array([1.03031015e03, -9.32270447e-01, -1.19588146e02])

        TIME_TO_APOAPSIS = 103.31401749403551

        sim.find_linear_tangent_params(R3D, V3D, TIME_TO_APOAPSIS)


if __name__ == "__main__":
    main()
