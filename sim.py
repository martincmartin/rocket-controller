import math
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar

KERBIN_RADIUS = 600_000
TARGET_ALTITUDE = 80_000

# Swivel in vacuum:
VE = 320 * 9.80665  # m / sec
THRUST = 215_000  # Newtons = kg m / sec^2

R3D = np.array([429802.01356461, -1035.2718727, -453946.03678487])
V3D = np.array([1.03478094e03, -4.53893509e-01, -1.07819109e02])
M0 = 13047.7685546875
MU = 3531599999999.9995
MAX_BURN_TIME = 46.92997535726185
TIME_TO_APOAPSIS = 102.76744504657796

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


def prograde_dynamics(t, state, mu, thrust, ve):

    x, y, vx, vy, m = state

    # This is just a = F/m.  Would be easy to do in 3D if we wanted to skip the
    # projection.
    r = np.hypot(x, y)
    v = np.hypot(vx, vy)

    minus_mu_over_r_cubed = -mu / r**3

    ax = minus_mu_over_r_cubed * x
    ay = minus_mu_over_r_cubed * y

    if v > 1e-10:
        ax += thrust / m * vx / v
        ay += thrust / m * vy / v
        mdot = -thrust / ve
    else:
        mdot = 0

    return [vx, vy, ax, ay, mdot]


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


def error(state):
    elements = orbital_elements(
        np.array([state[0], state[1]]), np.array([state[2], state[3]]), MU
    )

    ap = elements["apoapsis_radius"]
    pe = elements["periapsis_radius"]

    target = KERBIN_RADIUS + TARGET_ALTITUDE

    return math.sqrt((ap - target) ** 2 + (pe - target) ** 2 / 2)


def find_burn_params(time_to_apopasis, mass):
    r_hat, w_hat, r, v = project(R3D, V3D)

    # Simulate coasting (no thrust) up until apopasis.  We know we need to burn
    # before apoapsis, so that's a good upper bound on when to start burning.
    atol_vector = [
        1.0,  # Position within 1 meter.
        1.0,
        0.001,  # Velocity within 0.001 meters / sec
        0.001,
        0.1,  # Mass within 100 grams
    ]

    sol = solve_ivp(
        prograde_dynamics,
        (0, time_to_apopasis),
        (r[0], r[1], v[0], v[1], mass),
        args=(MU, 0.0, VE),
        rtol=1e-10,
        atol=atol_vector,
        dense_output=True,
    )

    # print(sol.t[-1], ": ", sol.y[:, -1])

    # for t, state in zip(sol.t, sol.y.T):
    #     x, y, vx, vy, mass = state
    #     elements = orbital_elements(np.array([x, y]), np.array([vx, vy]), MU)
    #     print(
    #         f't: {t}, apoapsis = {elements["apoapsis_radius"]}, periapsis = {elements["periapsis_radius"]}, mass = {mass}'
    #     )

    def start_burn_at(t):
        print(f"***** Start burn at {t}")
        x, y, vx, vy, mass = sol.sol(t)
        return burn(np.array([x, y]), np.array([vx, vy]), mass, MAX_BURN_TIME)

    res = minimize_scalar(
        start_burn_at,
        bounds=(0, time_to_apopasis),
        method="bounded",
        options={
            "xatol": 0.01,  # Find burn start time to within xatol seconds.
            "disp": 3,
        },
    )

    print("**********  When to start burn  **********")
    print(res)


def burn(r, v, mass, max_burn_time):
    atol_vector = [
        1.0,  # Position within 1 meter.
        1.0,
        0.001,  # Velocity within 0.001 meters / sec
        0.001,
        0.1,  # Mass within 100 grams
    ]

    sol = solve_ivp(
        prograde_dynamics,
        (0, max_burn_time),
        (r[0], r[1], v[0], v[1], mass),
        args=(MU, THRUST, VE),
        rtol=1e-10,
        atol=atol_vector,
        dense_output=True,
    )

    # for t, state in zip(sol.t, sol.y.T):
    #     x, y, vx, vy, mass = state
    #     elements = orbital_elements(np.array([x, y]), np.array([vx, vy]), MU)
    #     print(state)
    #     print(
    #         f't: {t}, apoapsis = {elements["apoapsis_radius"]}, periapsis = {elements["periapsis_radius"]}, mass = {mass}'
    #     )

    # print(f"{sol.y}")

    def objective(t):
        # print(f"** Burn for {t} sec")
        return error(sol.sol(t))

    res = minimize_scalar(
        objective,
        bounds=(0, max_burn_time),
        method="bounded",
        options={"xatol": 0.01},  # Find burn time to with xatol seconds.
    )
    print(res)
    if res.success:
        print(f"Burn for {res.x} sec, RMS error: {res.fun / 1000.0} km")
        return res.fun
    else:
        print("Couldn't find burn time to minimze orbital error.")
        return np.inf


find_burn_params(TIME_TO_APOAPSIS, M0)
