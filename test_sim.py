"""Unit tests for sim.py orbital mechanics functions."""

import math
from dataclasses import dataclass

import numpy as np
import pytest

pytestmark = pytest.mark.filterwarnings("error::RuntimeWarning")

from sim import (
    Stage,
    Simulator,
    orbital_elements,
    project,
    to_rv,
)

KERBIN_RADIUS = 600_000
MU = 3.5316e12  # Kerbin
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


def vector(*coords: float) -> np.ndarray:
    return np.asarray(coords, dtype=float)


@pytest.fixture
def sim():
    return Simulator(
        MU,
        body_radius=KERBIN_RADIUS,
        target_altitude=TARGET_ALTITUDE,
        stages=[SWIVEL, TERRIER],
    )


# --- Coast dynamics test ---


@dataclass
class DynamicsTestState:
    r3d: np.ndarray
    v3d: np.ndarray
    elapsed: float
    apoapsis: float
    periapsis: float
    mass: float


# Coasting states from an actual KSP run
COAST_START = DynamicsTestState(
    vector(431112.62181422, -1047.74169562, -454361.98728172),
    vector(1.02438523e03, -7.69046995e-01, -1.07508305e02),
    54.363529664213274,
    65555.19069490046 + KERBIN_RADIUS,
    -574167.8198834253 + KERBIN_RADIUS,
    mass=13055.69140625,
)
COAST_FINISH = DynamicsTestState(
    vector(432296.55963095, -1048.62195439, -454482.19824191),
    vector(1.01700685e03, -7.48933331e-01, -9.98698324e01),
    55.523529664238595,
    65534.03015425219 + KERBIN_RADIUS,
    -574174.862050127 + KERBIN_RADIUS,
    mass=13055.69140625,
)


def test_coast_dynamics(sim):
    """Verify coasting propagation matches recorded KSP trajectory."""
    r_hat, w_hat, r, v = project(COAST_START.r3d, COAST_START.v3d)

    def full(vec):
        return vec[0] * r_hat + vec[1] * w_hat

    np.testing.assert_allclose(full(r), COAST_START.r3d)

    with np.errstate(invalid="raise"):
        solution = sim.solve_coast((COAST_START.elapsed, COAST_FINISH.elapsed), r, v)
        assert math.isclose(solution.t[-1], COAST_FINISH.elapsed)
        end_r, end_v = to_rv(solution.y[:, -1])
        np.testing.assert_allclose(full(end_r), COAST_FINISH.r3d, atol=0.5)
        np.testing.assert_allclose(full(end_v), COAST_FINISH.v3d, atol=0.2)


# --- orbital_elements tests ---


def test_circular_orbit():
    """Circular orbit: e=0, a=R, rp=ra=R, zero eccentricity vector."""
    R = 680_000.0
    v_c = math.sqrt(MU / R)
    elems = orbital_elements(vector(R, 0.0), vector(0.0, v_c), MU)

    assert elems["eccentricity"] == pytest.approx(0.0, abs=1e-10)
    assert elems["semi_major_axis"] == pytest.approx(R, rel=1e-10)
    assert elems["periapsis_radius"] == pytest.approx(R, rel=1e-10)
    assert elems["apoapsis_radius"] == pytest.approx(R, rel=1e-10)
    assert elems["angular_momentum"] == pytest.approx(R * v_c, rel=1e-10)
    np.testing.assert_allclose(elems["eccentricity_vector"], [0.0, 0.0], atol=1e-10)


def test_elliptical_orbit_at_periapsis():
    """Elliptical orbit viewed from periapsis: e_vec points toward periapsis (+x)."""
    a = 750_000.0
    e = 0.1
    rp = a * (1 - e)
    ra = a * (1 + e)
    v_p = math.sqrt(MU * (2.0 / rp - 1.0 / a))

    elems = orbital_elements(vector(rp, 0.0), vector(0.0, v_p), MU)

    assert elems["eccentricity"] == pytest.approx(e, rel=1e-10)
    assert elems["semi_major_axis"] == pytest.approx(a, rel=1e-10)
    assert elems["periapsis_radius"] == pytest.approx(rp, rel=1e-10)
    assert elems["apoapsis_radius"] == pytest.approx(ra, rel=1e-10)
    e_vec = elems["eccentricity_vector"]
    assert e_vec[0] == pytest.approx(e, abs=1e-10)
    assert e_vec[1] == pytest.approx(0.0, abs=1e-10)


def test_elliptical_orbit_at_apoapsis():
    """Elliptical orbit viewed from apoapsis: e_vec points toward periapsis (-x)."""
    a = 750_000.0
    e = 0.1
    rp = a * (1 - e)
    ra = a * (1 + e)
    v_a = math.sqrt(MU * (2.0 / ra - 1.0 / a))

    # At apoapsis along +x, velocity is -y (clockwise orbit)
    elems = orbital_elements(vector(ra, 0.0), vector(0.0, -v_a), MU)

    assert elems["eccentricity"] == pytest.approx(e, rel=1e-10)
    assert elems["semi_major_axis"] == pytest.approx(a, rel=1e-10)
    assert elems["periapsis_radius"] == pytest.approx(rp, rel=1e-10)
    assert elems["apoapsis_radius"] == pytest.approx(ra, rel=1e-10)
    e_vec = elems["eccentricity_vector"]
    assert e_vec[0] == pytest.approx(-e, abs=1e-10)
    assert e_vec[1] == pytest.approx(0.0, abs=1e-10)


def test_hyperbolic_orbit():
    """Hyperbolic orbit: e>1, a<0, ra=inf, positive energy."""
    R = 680_000.0
    v_esc = math.sqrt(2 * MU / R)
    v_hyp = v_esc * 1.2

    elems = orbital_elements(vector(R, 0.0), vector(0.0, v_hyp), MU)

    assert elems["eccentricity"] > 1.0
    assert elems["semi_major_axis"] < 0
    assert elems["periapsis_radius"] == pytest.approx(R, rel=1e-6)
    assert elems["apoapsis_radius"] == np.inf
    assert elems["specific_energy"] > 0


def test_rotated_elliptical_orbit():
    """Same ellipse rotated 90°: periapsis along +y, e_vec points +y."""
    a = 750_000.0
    e = 0.1
    rp = a * (1 - e)
    ra = a * (1 + e)
    v_p = math.sqrt(MU * (2.0 / rp - 1.0 / a))

    elems = orbital_elements(vector(0.0, rp), vector(-v_p, 0.0), MU)

    assert elems["eccentricity"] == pytest.approx(e, rel=1e-10)
    assert elems["semi_major_axis"] == pytest.approx(a, rel=1e-10)
    assert elems["periapsis_radius"] == pytest.approx(rp, rel=1e-10)
    assert elems["apoapsis_radius"] == pytest.approx(ra, rel=1e-10)
    e_vec = elems["eccentricity_vector"]
    assert e_vec[0] == pytest.approx(0.0, abs=1e-10)
    assert e_vec[1] == pytest.approx(e, abs=1e-10)


# --- prograde_at_apoapsis tests ---


def test_prograde_at_apoapsis_ccw_periapsis_along_x():
    """CCW orbit (h>0), periapsis along +x.
    Apoapsis is along -x. For CCW motion (+x→+y→-x→-y), at -x the velocity
    points in -y direction → prograde angle = -π/2.
    """
    a = 750_000.0
    e = 0.1
    rp = a * (1 - e)
    v_p = math.sqrt(MU * (2.0 / rp - 1.0 / a))

    # At periapsis along +x, CCW velocity is +y
    elems = orbital_elements(vector(rp, 0.0), vector(0.0, v_p), MU)
    assert elems["angular_momentum"] > 0  # confirm CCW

    angle = Simulator.prograde_at_apoapsis(elems)
    assert angle == pytest.approx(-math.pi / 2, abs=1e-10)


def test_prograde_at_apoapsis_cw_periapsis_along_x():
    """CW orbit (h<0), periapsis along +x.
    Apoapsis along -x. For CW motion (+x→-y→-x→+y), at -x the velocity
    points in +y direction → prograde angle = π/2.
    """
    a = 750_000.0
    e = 0.1
    rp = a * (1 - e)
    v_p = math.sqrt(MU * (2.0 / rp - 1.0 / a))

    # At periapsis along +x, CW velocity is -y
    elems = orbital_elements(vector(rp, 0.0), vector(0.0, -v_p), MU)
    assert elems["angular_momentum"] < 0  # confirm CW

    angle = Simulator.prograde_at_apoapsis(elems)
    assert angle == pytest.approx(math.pi / 2, abs=1e-10)


def test_prograde_at_apoapsis_ccw_periapsis_along_y():
    """CCW orbit, periapsis along +y.
    Apoapsis is along -y. For CCW motion (+y→-x→-y→+x), at -y the velocity
    points in +x direction → angle = 0.
    """
    a = 750_000.0
    e = 0.1
    rp = a * (1 - e)
    v_p = math.sqrt(MU * (2.0 / rp - 1.0 / a))

    # At periapsis along +y, CCW velocity is -x
    elems = orbital_elements(vector(0.0, rp), vector(-v_p, 0.0), MU)
    assert elems["angular_momentum"] > 0  # confirm CCW

    angle = Simulator.prograde_at_apoapsis(elems)
    assert math.cos(angle) == pytest.approx(1.0, abs=1e-10)
    assert math.sin(angle) == pytest.approx(0.0, abs=1e-10)


def test_prograde_at_apoapsis_matches_actual_velocity(sim):
    """The angle returned should match the actual velocity direction at apoapsis,
    verified by propagating to apoapsis and checking the velocity direction.
    """
    a = 750_000.0
    e = 0.1
    rp = a * (1 - e)
    ra = a * (1 + e)
    v_p = math.sqrt(MU * (2.0 / rp - 1.0 / a))

    # Start at periapsis (CCW)
    r0 = vector(rp, 0.0)
    v0 = vector(0.0, v_p)

    # Compute half-period: T/2 = π * sqrt(a³/mu)
    half_period = math.pi * math.sqrt(a**3 / MU)

    # Propagate to apoapsis
    solution = sim.solve_coast((0.0, half_period), r0, v0)
    r_apo, v_apo = to_rv(solution.y[:, -1])

    # At apoapsis the radius should be ~ra (within 100m — integrator stops at T/2,
    # not the exact apoapsis moment)
    assert np.linalg.norm(r_apo) == pytest.approx(ra, abs=10.0)

    # The actual prograde angle from propagation
    actual_angle = math.atan2(v_apo[1], v_apo[0])

    # The angle predicted by prograde_at_apoapsis from the initial orbit elements
    elems = orbital_elements(r0, v0, MU)
    predicted_angle = Simulator.prograde_at_apoapsis(elems)

    # Compare as unit vectors to avoid ±π wrapping issues
    assert math.cos(actual_angle) == pytest.approx(math.cos(predicted_angle), abs=1e-4)
    assert math.sin(actual_angle) == pytest.approx(math.sin(predicted_angle), abs=1e-4)
