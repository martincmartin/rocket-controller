#!/usr/bin/env python3

import math
import sys
from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar

KERBIN_RADIUS = 600_000


@dataclass
class Stage:
    name: str
    ve: float
    thrust: float
    max_burn_time: float
    initial_mass: float


"""
- Verify that, after the Swivel has burned out, the mass is 9832.493492035084 .
- Take into account staging, the mass after the stage has decoupled/start of next
  stage, and the new thrust and ve.
"""


def cross2d(r, v):
    return r[0] * v[1] - r[1] * v[0]


def project(r, v):
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


def to_arrays(state):
    x, y, vx, vy, mass = state
    return (np.array([x, y]), np.array([vx, vy]), mass)


def prograde_dynamics(t, state, mu, ve, thrust):
    r, v, mass = to_arrays(state)
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


def orbital_elements(r, v, mu):
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

    # Scalar angular momentum (z-component)
    h = cross2d(r, v)

    # Eccentricity vector
    v_perp = np.array([v[1], -v[0]])
    e_vec = (h / mu) * v_perp - r / r_mag
    e = np.linalg.norm(e_vec)

    # Specific orbital energy
    energy = 0.5 * v_mag**2 - mu / r_mag

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
        ra = np.inf

    return {
        "semi_major_axis": a,
        "eccentricity": e,
        "eccentricity_vector": e_vec,
        "angular_momentum": h,
        "specific_energy": energy,
        "periapsis_radius": rp,
        "apoapsis_radius": ra,
    }


class Simulator:
    """Prograde-burn orbital simulation."""

    ATOL_VECTOR = [
        1.0,  # Position within 1 meter.
        1.0,
        0.001,  # Velocity within 0.001 meters / sec
        0.001,
        0.1,  # Mass within 100 grams
    ]

    def __init__(self, mu, body_radius, target_altitude, stages):
        self.mu = mu
        self.target_radius = body_radius + target_altitude
        self.stages = stages

    def solve(self, t_span, r, v, mass=1.0, ve=1.0, thrust=0.0):
        return solve_ivp(
            prograde_dynamics,
            t_span,
            (r[0], r[1], v[0], v[1], mass),
            args=(self.mu, ve, thrust),
            rtol=1e-10,
            atol=self.ATOL_VECTOR,
            dense_output=True,
        )

    def solve_stage(self, r, v, stage):
        return self.solve(
            (0, stage.max_burn_time), r, v, stage.initial_mass, stage.ve, stage.thrust
        )

    # Returns error?
    def circularization_burn(self, r, v):

        # If our periapsis is already at or above target, there's nothing to do.

        orbit = orbital_elements(r, v, self.mu)
        assert orbit["periapsis_radius"] < self.target_radius

        # Iterate over stages to find the one where we'll achieve our periapsis
        # goal.
        for stage in self.stages:
            solution = self.solve_stage(r, v, stage)

            # for t, state in zip(sol.t, sol.y.T):
            #     x, y, vx, vy, mass = state
            #     elements = orbital_elements(np.array([x, y]), np.array([vx, vy]), self.mu)
            #     print(state)
            #     print(
            #         f't: {t}, apoapsis = {elements["apoapsis_radius"]}, periapsis = {elements["periapsis_radius"]}, mass = {mass}'
            #     )

            # print(f"{solution.y}")

            r, v, _ = to_arrays(solution.y[:, -1])
            orbit = orbital_elements(r, v, self.mu)
            if orbit["periapsis_radius"] >= self.target_radius:
                print(f"Stage {stage.name} will hit periapsis target.")
                # Somewhere in this stage we hit our periapsis goal, so find
                # the best burn time.
                return self.find_burn_time(solution, stage)

            print(
                f"Stage {stage.name} will only raise periapsis to { orbit["periapsis_radius"]}, which is below target of {self.target_radius}"
            )
            # Simulate staging as a 1 second coast.
            solution = self.solve((0, 1.0), r, v)
            r, v, _ = to_arrays(solution.y[:, -1])

        return np.inf

    # Returns error() value at the minimum burn time.
    def find_burn_time(self, solution, stage):

        def objective(t):
            # print(f"** Burn for {t} sec")
            return self.error(solution.sol(t))

        res = minimize_scalar(
            objective,
            bounds=(0, stage.max_burn_time),
            method="bounded",
            options={"xatol": 0.01},  # Find burn time to with xatol seconds.
        )
        print(res)
        if res.success:
            print(f"Burn for {res.x} sec, RMS error: {res.fun / 1000.0} km")
            r, v, _ = to_arrays(solution.sol(res.x))
            elements = orbital_elements(r, v, self.mu)

            ap = elements["apoapsis_radius"]
            pe = elements["periapsis_radius"]
            print(f"apo: {ap - KERBIN_RADIUS}, per: {pe - KERBIN_RADIUS}")

            return res.fun
        else:
            print("Couldn't find burn time to minimze orbital error.")
            return np.inf

    # INITIAL ENTRY POINT.
    def find_burn_params(self, r3d, v3d, time_to_apoapsis):
        initial_mass = self.stages[0].initial_mass
        r_hat, w_hat, r, v = project(r3d, v3d)

        # Simulate coasting (no thrust) up until apopasis.  We know we need to burn
        # before apoapsis, so that's a good upper bound on when to start burning.
        sol = self.solve((0, time_to_apoapsis), r, v)

        # print(sol.t[-1], ": ", sol.y[:, -1])

        # for t, state in zip(sol.t, sol.y.T):
        #     x, y, vx, vy, mass = state
        #     elements = orbital_elements(np.array([x, y]), np.array([vx, vy]), MU)
        #     print(
        #         f't: {t}, apoapsis = {elements["apoapsis_radius"]}, periapsis = {elements["periapsis_radius"]}, mass = {mass}'
        #     )

        def start_burn_at(t):
            print(f"***** Simulating starting the burn at {t}")
            r, v, _ = to_arrays(sol.sol(t))
            return self.circularization_burn(r, v)

        for t in np.linspace(0, time_to_apoapsis, 100):
            err = start_burn_at(t)
            print(f"Starting burn at {t}, error is {err}.")

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

        r, v, _ = to_arrays(sol.sol(res.x))
        print(f"altitude: {np.linalg.norm(r) - KERBIN_RADIUS}")

    def error(self, state):
        r, v, _ = to_arrays(state)
        elements = orbital_elements(r, v, self.mu)

        ap = elements["apoapsis_radius"]
        pe = elements["periapsis_radius"]

        return math.sqrt(
            ((ap - self.target_radius) ** 2 + (pe - self.target_radius) ** 2) / 2
        )


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


# Some unit tests.


@dataclass
class DynamicsTestState:
    r3d: np.array
    v3d: np.array
    elapsed: float
    apopasis: float
    periapsis: flaot
    mass: float


def test():
    # Coasting from an actual KSP run
    start = DynamicsTestState(
        np.array([431112.62181422, -1047.74169562, -454361.98728172]),
        np.array([1.02438523e03, -7.69046995e-01, -1.07508305e02]),
        54.363529664213274,
        65555.19069490046 + KERBIN_RADIUS,
        -574167.8198834253 + KERBIN_RADIUS,
        mass=13055.69140625,
    )
    finish = DynamicsTestState(
        np.array([432296.55963095, -1048.62195439, -454482.19824191]),
        np.array([1.01700685e03, -7.48933331e-01, -9.98698324e01]),
        55.523529664238595,
        65534.03015425219 + KERBIN_RADIUS,
        -574174.862050127 + KERBIN_RADIUS,
        mass=13055.69140625,
    )
    sim = Simulator(
        MU,
        body_radius=KERBIN_RADIUS,
        target_altitude=TARGET_ALTITUDE,
        stages=[SWIVEL, TERRIER],
    )
    r_hat, w_hat, r, v = project(start.r3d, start.v3d)

    print(f"{r=}, {v=}")

    def full(vec):
        return vec[0] * r_hat + vec[1] * w_hat

    np.testing.assert_allclose(full(r), start.r3d)

    solution = sim.solve((start.elapsed, finish.elapsed), r, v)
    assert math.isclose(solution.t[-1], finish.elapsed)
    end_r, end_v, _ = to_arrays(solution.y[:, -1])
    np.testing.assert_allclose(full(end_r), finish.r3d, atol=0.5)
    np.testing.assert_allclose(full(end_v), finish.v3d, atol=0.2)

    # soluiton = self.solve(


test()


R3D = np.array([428392.15435586, -1053.61873734, -455905.93323801])
V3D = np.array([1.03031015e03, -9.32270447e-01, -1.19588146e02])


TIME_TO_APOAPSIS = 103.31401749403551

sim.find_burn_params(R3D, V3D, TIME_TO_APOAPSIS)
