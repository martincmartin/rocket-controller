#!/usr/bin/env python3

import math
import resource
import time
from abc import ABC, abstractmethod
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
from scipy.optimize import OptimizeResult, minimize, minimize_scalar

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
        self.minor_page_faults: int = 0    # ru_minflt (memory not on disk)
        self.major_page_faults: int = 0    # ru_majflt (memory on disk, required I/O)
        self.voluntary_context_switches: int = 0   # ru_nvcsw (yield/blocking)
        self.involuntary_context_switches: int = 0  # ru_nivcsw (preemption)
        self.input_blocks: int = 0         # ru_inblock
        self.output_blocks: int = 0        # ru_oublock
        
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
        lines.append(f"Peak memory:               {self.peak_memory_kb / 1024:8.1f} MB")
        lines.append(f"Minor page faults:         {self.minor_page_faults:8d}")
        lines.append(f"Major page faults:         {self.major_page_faults:8d}")
        
        # Context switches
        lines.append(f"Voluntary context switches: {self.voluntary_context_switches:8d}")
        lines.append(f"Involuntary context switches: {self.involuntary_context_switches:8d}")
        
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
    """Final state after propagating a linear-tangent burn, plus its error
    against the target orbit (see Simulator.error)."""

    r: Vector
    v: Vector
    mass: float
    error: float


"""
- Verify that, after the Swivel has burned out, the mass is 9832.493492035084 .
- Take into account staging, the mass after the stage has decoupled/start of next
  stage, and the new thrust and ve.
"""


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


def prograde_dynamics(
    t: float, state: np.ndarray, mu: float, ve: float, thrust: float
) -> list[float]:
    r, v, mass = to_rvm(state)
    # print(f"***** In prograde_dynamcs, {t=}")
    # print(f"{r=}, {v=}, {mass=}")

    # This is just a = F/m.  Would be easy to do in 3D if we wanted to skip the
    # projection.
    r_norm = np.linalg.norm(r)
    v_norm = np.linalg.norm(v)

    a = -mu / r_norm**3 * r

    # print(f"{a=}")
    if v_norm > 1e-10:
        a += thrust / (mass * v_norm) * v

        mdot = -thrust / ve
    else:
        mdot = 0

    # print(f"{a=}, {mdot=}")
    return [v[0], v[1], a[0], a[1], mdot]


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


class Regime(ABC):
    """Picks which linear-tangent parameters the outer optimizer searches
    over directly, versus which are resolved by a nested search.

    Subclasses let you experiment with different constrained optimization
    regimes (e.g. 3D vs. 4D) without touching the simulation code: implement
    `x0`/`bounds`/`evaluate`/`get_final_state` and pass an instance to
    Simulator.find_linear_tangent_params.
    """

    name: str

    @abstractmethod
    def x0(self, sim: "Simulator", coast_bound: float) -> np.ndarray:
        """Initial guess for the optimizer."""

    @abstractmethod
    def bounds(self, sim: "Simulator", coast_bound: float) -> list[tuple[float, float]]:
        """Bounds for each free parameter, in the same order as x0()."""

    @abstractmethod
    def evaluate(
        self,
        sim: "Simulator",
        coast_fn: Callable[[float], Vector],
        ref_angle: float,
        params: np.ndarray,
    ) -> tuple[float, Vector, Vector, float]:
        """Map an optimizer vector x to (error, r, v, burn_time)."""


class Regime3D(Regime):
    """Free params: (coast_time, a_coeff, b_coeff).

    burn_time isn't searched directly; instead, for each (coast_time, a, b)
    candidate we run a nested 1D search over burn_time to find the cutoff
    that minimizes orbital error, and report that as the objective value.
    """

    name = "3D (coast_time, a, b; burn_time solved by nested search)"

    def x0(self, sim: "Simulator", coast_bound: float) -> np.ndarray:
        return np.array([coast_bound / 2, 0.0, 0.0])

    def bounds(self, sim: "Simulator", coast_bound: float) -> list[tuple[float, float]]:
        return [(0.0, coast_bound), (-5.0, 5.0), (-1.0, 1.0)]

    def evaluate(
        self,
        sim: "Simulator",
        coast_fn: Callable[[float], Vector],
        ref_angle: float,
        params: np.ndarray,
    ) -> tuple[float, Vector, Vector, float]:
        coast_time, a_coeff, b_coeff = params
        r, v = to_rv(coast_fn(coast_time))
        budget = sim.total_burn_budget()

        def inner(burn_time: float) -> float:
            return sim.propagate_linear_tangent(
                r, v, a_coeff, b_coeff, ref_angle, burn_time
            ).error

        res = minimize_scalar(
            inner, bounds=(0.0, budget), method="bounded", options={"xatol": 0.01}
        )
        # Fetch the final state at the optimal burn_time
        result = sim.propagate_linear_tangent(r, v, a_coeff, b_coeff, ref_angle, res.x)
        return (res.fun, result.r, result.v, res.x)


class Regime4D(Regime):
    """Free params: (coast_time, a_coeff, b_coeff, burn_time), all searched
    simultaneously by the outer optimizer."""

    name = "4D (coast_time, a, b, burn_time all searched jointly)"

    def x0(self, sim: "Simulator", coast_bound: float) -> np.ndarray:
        return np.array([coast_bound / 2, 0.0, 0.0, sim.total_burn_budget() / 2])

    def bounds(self, sim: "Simulator", coast_bound: float) -> list[tuple[float, float]]:
        return [
            (0.0, coast_bound),
            (-5.0, 5.0),
            (-1.0, 1.0),
            (0.0, sim.total_burn_budget()),
        ]

    def evaluate(
        self,
        sim: "Simulator",
        coast_fn: Callable[[float], Vector],
        ref_angle: float,
        params: np.ndarray,
    ) -> tuple[float, Vector, Vector, float]:
        coast_time, a_coeff, b_coeff, burn_time = params
        r, v = to_rv(coast_fn(coast_time))
        result = sim.propagate_linear_tangent(
            r, v, a_coeff, b_coeff, ref_angle, burn_time
        )
        return (result.error, result.r, result.v, burn_time)


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
        r: Vector,
        v: Vector,
        stage: Stage,
        a_coeff: float,
        b_coeff: float,
        ref_angle: float,
    ) -> OdeResult:
        return solve_ivp(
            linear_tangent_dynamics,
            (0, stage.max_burn_time),
            (r[0], r[1], v[0], v[1], stage.initial_mass),
            args=(self.mu, stage.ve, stage.thrust, a_coeff, b_coeff, ref_angle),
            rtol=1e-10,
            atol=self.ATOL_THRUST_VECTOR,
            dense_output=True,
        )

    @_validate
    def solve_prograde(self, r: Vector, v: Vector, stage: Stage) -> OdeResult:
        return solve_ivp(
            prograde_dynamics,
            (0, stage.max_burn_time),
            (r[0], r[1], v[0], v[1], stage.initial_mass),
            args=(self.mu, stage.ve, stage.thrust),
            rtol=1e-10,
            atol=self.ATOL_THRUST_VECTOR,
            dense_output=True,
        )

    @_validate
    def total_burn_budget(self) -> float:
        """Total burn time available across all remaining stages."""
        return sum(stage.max_burn_time for stage in self.stages)

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
        """Burn under the linear-tangent steering law for `burn_time` seconds,
        walking across stage boundaries (with a 1 second staging coast
        between them) as needed.

        `burn_time` is measured from the start of the current (first) stage,
        i.e. it's the total elapsed burn time, not a per-stage time.
        """
        remaining = burn_time
        mass = math.nan

        for stage in self.stages:
            solution = self.solve_linear_tangent(
                r, v, stage, a_coeff, b_coeff, ref_angle
            )
            assert solution.sol is not None

            if remaining <= stage.max_burn_time:
                state = solution.sol(remaining)
                r, v, mass = to_rvm(state)
                return BurnResult(r, v, mass, self.error(state))

            remaining -= stage.max_burn_time
            # Cast needed because y is type ndarray[float64 | complex128]
            r, v, mass = to_rvm(cast(Vector, solution.y[:, -1]))
            # Simulate staging as a 1 second coast.
            coast = self.solve_coast((0, 1.0), r, v)
            # Cast needed because y is type ndarray[float64 | complex128]
            r, v = to_rv(cast(Vector, coast.y[:, -1]))

        # Ran out of stages before using all of the requested burn time.
        return BurnResult(r, v, mass, MAX_ERROR)

    # Returns error?  This is after the coast, starts with the actual burn.
    @_validate
    def circularization_burn(self, r: Vector, v: Vector) -> float:

        # If our periapsis is already at or above target, there's nothing to do.

        orbit = orbital_elements(r, v, self.mu)
        assert orbit.periapsis_radius < self.target_radius

        # Iterate over stages to find the one where we'll achieve our periapsis
        # goal.
        for stage in self.stages:
            solution = self.solve_prograde(r, v, stage)

            # for t, state in zip(sol.t, sol.y.T):
            #     x, y, vx, vy, mass = state
            #     elements = orbital_elements(np.array([x, y]), np.array([vx, vy]), self.mu)
            #     print(state)
            #     print(
            #         f't: {t}, apoapsis = {elements["apoapsis_radius"]}, periapsis = {elements["periapsis_radius"]}, mass = {mass}'
            #     )

            # print(f"{solution.y}")

            # Cast needed because y is type ndarray[float64 | complex128]
            r, v, _ = to_rvm(cast(Vector, solution.y[:, -1]))
            orbit = orbital_elements(r, v, self.mu)
            if orbit.periapsis_radius >= self.target_radius:
                print(f"Stage {stage.name} will hit periapsis target.")
                # Somewhere in this stage we hit our periapsis goal, so find
                # the best burn time.
                print("About to find_burn_time", flush=True)
                ret = self.find_burn_time(solution, stage)
                print("back from find_burn_time", flush=True)
                return ret

            print(
                f"Stage {stage.name} will only raise periapsis to { orbit.periapsis_radius}, which is below target of {self.target_radius}"
            )
            # Simulate staging as a 1 second coast.
            solution = self.solve_coast((0, 1.0), r, v)
            # Cast needed because y is type ndarray[float64 | complex128]
            r, v = to_rv(cast(Vector, solution.y[:, -1]))

        return MAX_ERROR

    # Returns error() value at the minimum burn time.
    @_validate
    def find_burn_time(self, solution: OdeResult, stage: Stage) -> float:
        assert solution.sol is not None
        sol_fn = solution.sol

        def objective(t: float) -> float:
            # print(f"** Burn for {t} sec")
            return self.error(sol_fn(t))

        res = minimize_scalar(
            objective,
            bounds=(0, stage.max_burn_time),
            method="bounded",
            options={"xatol": 0.01},  # Find burn time to with xatol seconds.
        )
        print(res)
        if res.success:
            print(f"Burn for {res.x} sec, RMS error: {res.fun / 1000.0} km")
            r, v, _ = to_rvm(sol_fn(res.x))
            elements = orbital_elements(r, v, self.mu)

            ap = elements.apoapsis_radius
            pe = elements.periapsis_radius
            print(f"apo: {ap - KERBIN_RADIUS}, per: {pe - KERBIN_RADIUS}")

            return res.fun
        else:
            print("Couldn't find burn time to minimze orbital error.", flush=True)
            return MAX_ERROR

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
        regime: Regime,
    ) -> OptimizeResult:
        """Find linear-tangent steering parameters that circularize the
        orbit, using whichever free-parameter set `regime` defines (e.g.
        Regime3D or Regime4D). See the `Regime` class for how to add more.
        """
        with TimingContext(label="find_linear_tangent_params", auto_print=False) as timer:
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

            def objective(x: np.ndarray) -> float:
                error, _, _, _ = regime.evaluate(self, coast_fn, ref_angle, x)
                return error

            res = minimize(
                objective,
                x0=regime.x0(self, time_to_apoapsis),
                bounds=regime.bounds(self, time_to_apoapsis),
                method="Nelder-Mead",
                options={"xatol": 0.01, "fatol": 1.0},
            )

            # Fetch the final state at the optimal solution
            error, r_final, v_final, burn_time = regime.evaluate(
                self, coast_fn, ref_angle, res.x
            )
            final_orbit = orbital_elements(r_final, v_final, self.mu)

            # Extract parameters
            if isinstance(regime, Regime3D):
                coast_time, a_coeff, b_coeff = res.x
            else:  # Regime4D
                coast_time, a_coeff, b_coeff, burn_time = res.x
        
        # Print summary (outside context so timing is finalized)
        print(f"\n***** Regime: {regime.name}")
        print(f"Coast time:        {coast_time:8.2f} s")
        print(f"a coefficient:     {a_coeff:8.6f}")
        print(f"b coefficient:     {b_coeff:8.6f}")
        print(f"Burn time:         {burn_time:8.2f} s")
        print(f"RMS orbital error: {error:8.2f} m")
        ap_alt = final_orbit.apoapsis_radius - KERBIN_RADIUS
        pe_alt = final_orbit.periapsis_radius - KERBIN_RADIUS
        print(f"Apoapsis:          {ap_alt:8.2f} m")
        print(f"Periapsis:         {pe_alt:8.2f} m")
        print(timer.summary())
        return res

    # INITIAL ENTRY POINT.
    @_validate
    def find_burn_params(
        self, r3d: Vector, v3d: Vector, time_to_apoapsis: float
    ) -> None:
        initial_mass = self.stages[0].initial_mass
        r_hat, w_hat, r, v = project(r3d, v3d)

        # Simulate coasting (no thrust) up until apoapsis.  We know we need to burn
        # before apoapsis, so that's a good upper bound on when to start burning.
        sol = self.solve_coast((0, time_to_apoapsis), r, v)
        assert sol.sol is not None
        sol_fn = sol.sol

        # print(sol.t[-1], ": ", sol.y[:, -1])

        # for t, state in zip(sol.t, sol.y.T):
        #     x, y, vx, vy, mass = state
        #     elements = orbital_elements(np.array([x, y]), np.array([vx, vy]), MU)
        #     print(
        #         f't: {t}, apoapsis = {elements["apoapsis_radius"]}, periapsis = {elements["periapsis_radius"]}, mass = {mass}'
        #     )

        def start_burn_at(t: float) -> float:
            print(f"***** Simulating starting the burn at {t}")
            r, v = to_rv(sol_fn(t))
            ret = self.circularization_burn(r, v)
            print("Returning from start_burn_at", flush=True)
            print(ret)
            return ret

        for t in np.linspace(0, time_to_apoapsis, 100):
            print("Calling start_burn_at", flush=True)
            err = start_burn_at(t)
            print(f"Starting burn at {t}, error is {err}.", flush=True)

        res = minimize_scalar(
            start_burn_at,
            bounds=(0, time_to_apoapsis),
            method="bounded",
            options={
                "xatol": 0.01,  # Find burn start time to within xatol seconds.
                "disp": 3,
            },
        )

        print("**********  When to start burn  **********")
        print(res)

        r, v = to_rv(sol_fn(res.x))
        print(f"altitude: {np.linalg.norm(r) - KERBIN_RADIUS}")

    @_validate
    def error(self, state: Vector) -> float:
        r, v, _ = to_rvm(state)
        elements = orbital_elements(r, v, self.mu)

        ap = elements.apoapsis_radius
        pe = elements.periapsis_radius

        return math.sqrt(
            ((ap - self.target_radius) ** 2 + (pe - self.target_radius) ** 2) / 2
        )


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

        # Switch which optimization regime to experiment with here.  See the
        # Regime class (and Regime3D / Regime4D) in sim.py to add more.
        regime: Regime = Regime3D()
        # regime: Regime = Regime3D()

        sim.find_linear_tangent_params(R3D, V3D, TIME_TO_APOAPSIS, regime)


if __name__ == "__main__":
    main()
