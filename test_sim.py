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
    to_rvm,
)

KERBIN_RADIUS = 600_000
MU = 3.5316e12  # Kerbin
TARGET_ALTITUDE = 80_000

# Real flight state used for find_linear_tangent_params integration tests
# (matches sim.py's main()).
R3D = np.array([428392.15435586, -1053.61873734, -455905.93323801])
V3D = np.array([1.03031015e03, -9.32270447e-01, -1.19588146e02])
TIME_TO_APOAPSIS = 103.31401749403551

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

    assert elems.eccentricity == pytest.approx(0.0, abs=1e-10)
    assert elems.semi_major_axis == pytest.approx(R, rel=1e-10)
    assert elems.periapsis_radius == pytest.approx(R, rel=1e-10)
    assert elems.apoapsis_radius == pytest.approx(R, rel=1e-10)
    assert elems.angular_momentum == pytest.approx(R * v_c, rel=1e-10)
    np.testing.assert_allclose(elems.eccentricity_vector, [0.0, 0.0], atol=1e-10)


def test_elliptical_orbit_at_periapsis():
    """Elliptical orbit viewed from periapsis: e_vec points toward periapsis (+x)."""
    a = 750_000.0
    e = 0.1
    rp = a * (1 - e)
    ra = a * (1 + e)
    v_p = math.sqrt(MU * (2.0 / rp - 1.0 / a))

    elems = orbital_elements(vector(rp, 0.0), vector(0.0, v_p), MU)

    assert elems.eccentricity == pytest.approx(e, rel=1e-10)
    assert elems.semi_major_axis == pytest.approx(a, rel=1e-10)
    assert elems.periapsis_radius == pytest.approx(rp, rel=1e-10)
    assert elems.apoapsis_radius == pytest.approx(ra, rel=1e-10)
    e_vec = elems.eccentricity_vector
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

    assert elems.eccentricity == pytest.approx(e, rel=1e-10)
    assert elems.semi_major_axis == pytest.approx(a, rel=1e-10)
    assert elems.periapsis_radius == pytest.approx(rp, rel=1e-10)
    assert elems.apoapsis_radius == pytest.approx(ra, rel=1e-10)
    e_vec = elems.eccentricity_vector
    assert e_vec[0] == pytest.approx(-e, abs=1e-10)
    assert e_vec[1] == pytest.approx(0.0, abs=1e-10)


def test_hyperbolic_orbit():
    """Hyperbolic orbit: e>1, a<0, ra=inf, positive energy."""
    R = 680_000.0
    v_esc = math.sqrt(2 * MU / R)
    v_hyp = v_esc * 1.2

    elems = orbital_elements(vector(R, 0.0), vector(0.0, v_hyp), MU)

    assert elems.eccentricity > 1.0
    assert elems.semi_major_axis < 0
    assert elems.periapsis_radius == pytest.approx(R, rel=1e-6)
    assert elems.apoapsis_radius == np.inf
    assert elems.specific_energy > 0


def test_rotated_elliptical_orbit():
    """Same ellipse rotated 90°: periapsis along +y, e_vec points +y."""
    a = 750_000.0
    e = 0.1
    rp = a * (1 - e)
    ra = a * (1 + e)
    v_p = math.sqrt(MU * (2.0 / rp - 1.0 / a))

    elems = orbital_elements(vector(0.0, rp), vector(-v_p, 0.0), MU)

    assert elems.eccentricity == pytest.approx(e, rel=1e-10)
    assert elems.semi_major_axis == pytest.approx(a, rel=1e-10)
    assert elems.periapsis_radius == pytest.approx(rp, rel=1e-10)
    assert elems.apoapsis_radius == pytest.approx(ra, rel=1e-10)
    e_vec = elems.eccentricity_vector
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
    assert elems.angular_momentum > 0  # confirm CCW

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
    assert elems.angular_momentum < 0  # confirm CW

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
    assert elems.angular_momentum > 0  # confirm CCW

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


# --- target_velocity tests ---


def test_target_velocity_magnitude_and_perpendicularity(sim):
    """Magnitude should be sqrt(mu/r), and result should be perpendicular to r,
    regardless of the (nonzero) input velocity used to pick a direction."""
    R = 680_000.0
    r = vector(R, 0.0)
    v = vector(0.0, 5000.0)  # arbitrary CCW-ish velocity

    result = sim.target_velocity(r, v)

    expected_speed = math.sqrt(MU / R)
    assert np.linalg.norm(result) == pytest.approx(expected_speed, rel=1e-10)
    assert np.dot(result, r) == pytest.approx(0.0, abs=1e-6)


def test_target_velocity_ccw_direction():
    """At r=(R,0) with a CCW-pointing v (+y), target velocity should point +y."""
    sim = Simulator(
        MU, body_radius=KERBIN_RADIUS, target_altitude=TARGET_ALTITUDE, stages=[]
    )
    R = 680_000.0
    r = vector(R, 0.0)
    v = vector(0.0, 1.0)  # tiny CCW-pointing velocity

    result = sim.target_velocity(r, v)

    expected_speed = math.sqrt(MU / R)
    np.testing.assert_allclose(result, [0.0, expected_speed], atol=1e-6)


def test_target_velocity_cw_direction():
    """At r=(R,0) with a CW-pointing v (-y), target velocity should point -y."""
    sim = Simulator(
        MU, body_radius=KERBIN_RADIUS, target_altitude=TARGET_ALTITUDE, stages=[]
    )
    R = 680_000.0
    r = vector(R, 0.0)
    v = vector(0.0, -1.0)  # tiny CW-pointing velocity

    result = sim.target_velocity(r, v)

    expected_speed = math.sqrt(MU / R)
    np.testing.assert_allclose(result, [0.0, -expected_speed], atol=1e-6)


def test_target_velocity_rotated_position():
    """At r=(0,R) with CCW-pointing v (-x direction, per the CCW convention
    used elsewhere in this file: periapsis along +y, CCW velocity is -x),
    target velocity should point in -x."""
    sim = Simulator(
        MU, body_radius=KERBIN_RADIUS, target_altitude=TARGET_ALTITUDE, stages=[]
    )
    R = 680_000.0
    r = vector(0.0, R)
    v = vector(-1.0, 0.0)

    result = sim.target_velocity(r, v)

    expected_speed = math.sqrt(MU / R)
    np.testing.assert_allclose(result, [-expected_speed, 0.0], atol=1e-6)


def test_target_velocity_independent_of_input_speed_magnitude():
    """Only the sign/direction of v should matter, not its magnitude."""
    sim = Simulator(
        MU, body_radius=KERBIN_RADIUS, target_altitude=TARGET_ALTITUDE, stages=[]
    )
    r = vector(680_000.0, 0.0)

    slow = sim.target_velocity(r, vector(0.0, 1.0))
    fast = sim.target_velocity(r, vector(0.0, 100_000.0))

    np.testing.assert_allclose(slow, fast, atol=1e-9)


def test_target_velocity_with_radial_component_ccw():
    """v has both a radial (+x, outward) and a CCW tangential (+y) component.
    The radial component should be ignored; result should match the pure
    CCW-tangential case."""
    sim = Simulator(
        MU, body_radius=KERBIN_RADIUS, target_altitude=TARGET_ALTITUDE, stages=[]
    )
    R = 680_000.0
    r = vector(R, 0.0)
    v = vector(200.0, 3000.0)  # radial + CCW tangential

    result = sim.target_velocity(r, v)

    expected_speed = math.sqrt(MU / R)
    np.testing.assert_allclose(result, [0.0, expected_speed], atol=1e-6)
    assert np.dot(result, r) == pytest.approx(0.0, abs=1e-6)


def test_target_velocity_with_radial_component_cw():
    """v has both a radial (-x, inward) and a CW tangential (-y) component.
    The radial component (and its sign) should not affect the result."""
    sim = Simulator(
        MU, body_radius=KERBIN_RADIUS, target_altitude=TARGET_ALTITUDE, stages=[]
    )
    R = 680_000.0
    r = vector(R, 0.0)
    v = vector(-200.0, -3000.0)  # radial (inward) + CW tangential

    result = sim.target_velocity(r, v)

    expected_speed = math.sqrt(MU / R)
    np.testing.assert_allclose(result, [0.0, -expected_speed], atol=1e-6)
    assert np.dot(result, r) == pytest.approx(0.0, abs=1e-6)


def test_target_velocity_purely_radial_input_defaults_to_ccw():
    """Edge case: if v has zero tangential component (purely radial), the
    flip condition (`dot < 0`) never triggers, so the implementation
    defaults to the CCW direction.  This test documents that behavior."""
    sim = Simulator(
        MU, body_radius=KERBIN_RADIUS, target_altitude=TARGET_ALTITUDE, stages=[]
    )
    R = 680_000.0
    r = vector(R, 0.0)
    v = vector(500.0, 0.0)  # purely radial, no tangential component

    result = sim.target_velocity(r, v)

    expected_speed = math.sqrt(MU / R)
    np.testing.assert_allclose(result, [0.0, expected_speed], atol=1e-6)


def test_target_velocity_arbitrary_angle_ccw():
    """r at an arbitrary angle (not on an axis); v purely tangential CCW."""
    sim = Simulator(
        MU, body_radius=KERBIN_RADIUS, target_altitude=TARGET_ALTITUDE, stages=[]
    )
    R = 750_000.0
    theta = math.radians(37.0)  # arbitrary angle, not a multiple of 45
    r = vector(R * math.cos(theta), R * math.sin(theta))

    # CCW tangent direction: rotate r_hat by +90 deg.
    tangent_ccw = vector(-math.sin(theta), math.cos(theta))
    v = 4000.0 * tangent_ccw

    result = sim.target_velocity(r, v)

    expected_speed = math.sqrt(MU / R)
    expected = expected_speed * tangent_ccw
    np.testing.assert_allclose(result, expected, atol=1e-6)
    assert np.dot(result, r) == pytest.approx(0.0, abs=1e-6)


def test_target_velocity_arbitrary_angle_cw_with_radial_component():
    """r at an arbitrary angle, in a different quadrant; v has both a
    radial (outward) and a CW tangential component.  Result direction
    should follow the tangential (CW) sign, ignoring the radial part."""
    sim = Simulator(
        MU, body_radius=KERBIN_RADIUS, target_altitude=TARGET_ALTITUDE, stages=[]
    )
    R = 750_000.0
    theta = math.radians(163.0)  # arbitrary angle, second quadrant
    r = vector(R * math.cos(theta), R * math.sin(theta))
    r_hat = vector(math.cos(theta), math.sin(theta))

    tangent_ccw = vector(-math.sin(theta), math.cos(theta))
    tangent_cw = -tangent_ccw

    v = 500.0 * r_hat + 2500.0 * tangent_cw  # radial (outward) + CW tangential

    result = sim.target_velocity(r, v)

    expected_speed = math.sqrt(MU / R)
    expected = expected_speed * tangent_cw
    np.testing.assert_allclose(result, expected, atol=1e-6)
    assert np.dot(result, r) == pytest.approx(0.0, abs=1e-6)


# --- find_linear_tangent_params (SLSQP) tests ---


def test_find_linear_tangent_params_converges(sim):
    """End-to-end regression test using real flight data (from main()):
    the SLSQP solve should succeed and produce a circular orbit at (or
    just above) the target altitude, within the burn/coast time bounds.
    """
    res = sim.find_linear_tangent_params(R3D, V3D, TIME_TO_APOAPSIS)

    assert res.success

    coast_time, a_coeff, b_coeff, burn_time = res.x
    assert -1e-6 <= coast_time <= TIME_TO_APOAPSIS + 1e-6
    assert -1e-6 <= burn_time <= sim.total_burn_budget() + 1e-6

    # Recompute the final state via the public API to check the resulting orbit.
    r_hat, w_hat, r0, v0 = project(R3D, V3D)
    orbit0 = orbital_elements(r0, v0, MU)
    ref_angle = Simulator.prograde_at_apoapsis(orbit0)

    sol = sim.solve_coast((0, TIME_TO_APOAPSIS), r0, v0)
    assert sol.sol is not None
    r_coast, v_coast = to_rv(sol.sol(coast_time))

    result = sim.propagate_linear_tangent(
        r_coast, v_coast, a_coeff, b_coeff, ref_angle, burn_time
    )
    final_orbit = orbital_elements(result.r, result.v, MU)

    # Circular orbit: periapsis ~= apoapsis.
    assert final_orbit.periapsis_radius == pytest.approx(
        final_orbit.apoapsis_radius, rel=1e-3
    )
    # Radius at/above target (inequality constraint), and close to it (since
    # the objective minimizes burn_time, pushing the solution toward the
    # constraint boundary).
    assert final_orbit.apoapsis_radius >= sim.target_radius - 10.0
    assert final_orbit.periapsis_radius == pytest.approx(sim.target_radius, abs=50.0)


def test_find_linear_tangent_params_memoizes_simulation(monkeypatch):
    """No two calls to propagate_linear_tangent during a single
    find_linear_tangent_params run should have identical arguments --
    i.e. the memoized `simulate` helper should dedupe repeated evaluations
    at the same optimizer point."""
    sim = Simulator(
        MU,
        body_radius=KERBIN_RADIUS,
        target_altitude=TARGET_ALTITUDE,
        stages=[SWIVEL, TERRIER],
    )

    calls: list[tuple[float, ...]] = []
    original = Simulator.propagate_linear_tangent

    def spy(self, r, v, a_coeff, b_coeff, ref_angle, burn_time):
        key = (
            round(float(r[0]), 6),
            round(float(r[1]), 6),
            round(float(v[0]), 6),
            round(float(v[1]), 6),
            round(float(a_coeff), 9),
            round(float(b_coeff), 9),
            round(float(burn_time), 9),
        )
        calls.append(key)
        return original(self, r, v, a_coeff, b_coeff, ref_angle, burn_time)

    monkeypatch.setattr(Simulator, "propagate_linear_tangent", spy)

    sim.find_linear_tangent_params(R3D, V3D, TIME_TO_APOAPSIS)

    assert len(calls) == len(set(calls)), (
        "propagate_linear_tangent was called more than once with identical "
        "arguments; memoization should have deduped these."
    )
    assert len(calls) > 0


# --- Continuous-time / staging-time bookkeeping tests ---


def test_solve_linear_tangent_t_offset(sim):
    """solve_linear_tangent's t domain should start at t_offset (not 0),
    so that the linear-tangent steering law's time reference continues
    from wherever a previous stage left off."""
    r_hat, w_hat, r0, v0 = project(R3D, V3D)
    t_offset = 100.0

    solution = sim.solve_linear_tangent(
        r0, v0, SWIVEL, 0.0, 0.0, 0.0, t_offset=t_offset
    )
    assert solution.sol is not None

    assert solution.t[0] == pytest.approx(t_offset)
    assert solution.t[-1] == pytest.approx(t_offset + SWIVEL.max_burn_time)

    initial_state = solution.sol(t_offset)
    assert initial_state[0] == pytest.approx(r0[0])
    assert initial_state[1] == pytest.approx(r0[1])
    assert initial_state[2] == pytest.approx(v0[0])
    assert initial_state[3] == pytest.approx(v0[1])
    assert initial_state[4] == pytest.approx(SWIVEL.initial_mass)


def test_total_burn_budget_includes_staging(sim):
    expected = SWIVEL.max_burn_time + TERRIER.max_burn_time + 1.0
    assert sim.total_burn_budget() == pytest.approx(expected)


def test_total_burn_budget_single_stage_no_staging():
    single_stage_sim = Simulator(
        MU, body_radius=KERBIN_RADIUS, target_altitude=TARGET_ALTITUDE, stages=[SWIVEL]
    )
    assert single_stage_sim.total_burn_budget() == pytest.approx(SWIVEL.max_burn_time)


def test_propagate_linear_tangent_burn_time_includes_staging_seconds(sim):
    """burn_time should include the 1 second staging coast, so the second
    stage should only burn for `burn_time - stage1.max_burn_time - 1.0`
    seconds, not `burn_time - stage1.max_burn_time`."""
    r_hat, w_hat, r0, v0 = project(R3D, V3D)
    partial_stage2_time = 10.0
    burn_time = SWIVEL.max_burn_time + 1.0 + partial_stage2_time

    result = sim.propagate_linear_tangent(r0, v0, 0.0, 0.0, 0.0, burn_time)

    mdot2 = TERRIER.thrust / TERRIER.ve
    expected_mass = TERRIER.initial_mass - mdot2 * partial_stage2_time
    assert result.mass == pytest.approx(expected_mass)


def test_propagate_linear_tangent_deadline_inside_staging_window(sim):
    """If burn_time's deadline falls inside the mandatory 1 second staging
    coast, only a partial coast should be applied, and stage 2 should not
    be started at all."""
    r_hat, w_hat, r0, v0 = project(R3D, V3D)
    partial_staging_time = 0.5
    burn_time = SWIVEL.max_burn_time + partial_staging_time

    result = sim.propagate_linear_tangent(r0, v0, 0.0, 0.0, 0.0, burn_time)

    # Manually reproduce: burn stage 1 fully, then coast for only
    # partial_staging_time (not the full 1.0 second).
    solution1 = sim.solve_linear_tangent(r0, v0, SWIVEL, 0.0, 0.0, 0.0)
    r1, v1, mass1 = to_rvm(solution1.y[:, -1])
    coast = sim.solve_coast((0, partial_staging_time), r1, v1)
    r_expected, v_expected = to_rv(coast.y[:, -1])

    assert result.mass == pytest.approx(mass1)
    assert result.r == pytest.approx(r_expected)
    assert result.v == pytest.approx(v_expected)


def test_propagate_linear_tangent_continuous_time_across_stages(sim, monkeypatch):
    """solve_linear_tangent's t_offset should continue from wherever the
    previous stage plus staging coast left off, not reset to 0 at the
    start of each stage."""
    r_hat, w_hat, r0, v0 = project(R3D, V3D)

    t_offsets: list[float] = []
    original = Simulator.solve_linear_tangent

    def spy(self, r, v, stage, a_coeff, b_coeff, ref_angle, t_offset=0.0):
        t_offsets.append(t_offset)
        return original(self, r, v, stage, a_coeff, b_coeff, ref_angle, t_offset)

    monkeypatch.setattr(Simulator, "solve_linear_tangent", spy)

    burn_time = SWIVEL.max_burn_time + 1.0 + 10.0
    sim.propagate_linear_tangent(r0, v0, 0.0, 0.0, 0.0, burn_time)

    assert t_offsets == [pytest.approx(0.0), pytest.approx(SWIVEL.max_burn_time + 1.0)]
