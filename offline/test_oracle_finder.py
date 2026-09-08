"""Tests for the standalone offline oracle finder."""

from __future__ import annotations

import math
from typing import cast

import numpy as np
import pytest
from scipy.integrate import solve_ivp

import offline.oracle_finder as oracle_finder
from offline.oracle_finder import (
    DEFAULT_MAX_BURN_TIME,
    DEFAULT_TARGET_RADIUS,
    MIN_PHASE_DURATION_SECONDS,
    Candidate,
    ExampleSegment,
    OracleResult,
    OrbitalParameters,
    OrbitalPlane,
    _impulse_timing_seed,
    _polar_rhs,
    _prepare_problem,
    _valid_explicit,
    find_oracle,
    kerbin_examples,
)


@pytest.fixture(scope="module")
def first_oracle_result() -> OracleResult:
    position, velocity, segments = kerbin_examples()["kerbin-first-example"]
    return find_oracle(position, velocity, segments)


def test_kerbin_state_maps_to_an_orthonormal_plane() -> None:
    position, velocity, segments = kerbin_examples()["kerbin-example"]
    problem = _prepare_problem(
        position,
        velocity,
        segments,
        3.5316e12,
        DEFAULT_TARGET_RADIUS,
        DEFAULT_MAX_BURN_TIME,
    )
    plane = OrbitalPlane(position, velocity)

    np.testing.assert_allclose(plane.r_hat @ plane.r_hat, 1.0)
    np.testing.assert_allclose(plane.t_hat @ plane.t_hat, 1.0)
    np.testing.assert_allclose(plane.n_hat @ plane.n_hat, 1.0)
    np.testing.assert_allclose(plane.r_hat @ plane.t_hat, 0.0, atol=1e-15)
    np.testing.assert_allclose(plane.n_hat, np.cross(plane.r_hat, plane.t_hat))
    np.testing.assert_allclose(plane.r_hat, problem.physical.plane.r_hat)
    np.testing.assert_allclose(plane.t_hat, problem.physical.plane.t_hat)
    np.testing.assert_allclose(plane.n_hat, problem.physical.plane.n_hat)
    np.testing.assert_allclose(
        np.array([problem.radius, *problem.velocity]),
        [0.932317760533369, 0.252573095512292, 0.199946283806988],
        rtol=0.0,
        atol=2e-12,
    )


def test_burn_limit_is_configurable_and_overrides_segment_data() -> None:
    position, velocity, segments = kerbin_examples()["kerbin-first-example"]
    short_limit = 91.25
    problem = _prepare_problem(
        position,
        velocity,
        segments,
        3.5316e12,
        DEFAULT_TARGET_RADIUS,
        short_limit,
    )

    assert problem.physical.max_burn_time_s == pytest.approx(short_limit)
    assert problem.stage.max_burn_tau == pytest.approx(
        short_limit / problem.physical.time_scale_s
    )
    assert segments[0].max_burn_time == DEFAULT_MAX_BURN_TIME


@pytest.mark.parametrize(
    (
        "name",
        "expected_burn_raise_apoapsis",
        "expected_coast",
        "expected_burn_circularize",
    ),
    [
        (
            "kerbin-example",
            0.06792177476991536,
            0.18432643732513448,
            0.2514428839093345,
        ),
        (
            "kerbin-first-example",
            0.02854588766904428,
            0.2658601723931313,
            0.2411753882232254,
        ),
        (
            "kerbin-coast-first-example",
            0.0,
            0.32051720214768065,
            0.19509153751120623,
        ),
    ],
)
def test_impulse_seed_preserves_timing_values(
    name: str,
    expected_burn_raise_apoapsis: float,
    expected_coast: float,
    expected_burn_circularize: float,
) -> None:
    position, velocity, segments = kerbin_examples()[name]
    problem = _prepare_problem(
        position,
        velocity,
        segments,
        3.5316e12,
        DEFAULT_TARGET_RADIUS,
        DEFAULT_MAX_BURN_TIME,
    )
    seed = _impulse_timing_seed(problem)

    assert seed.burn_raise_apoapsis == pytest.approx(
        expected_burn_raise_apoapsis, abs=2e-10
    )
    assert seed.coast == pytest.approx(expected_coast, abs=2e-10)
    assert seed.burn_circularize == pytest.approx(expected_burn_circularize, abs=2e-10)
    assert (
        seed.burn_raise_apoapsis + seed.burn_circularize
        <= problem.stage.max_burn_tau + 1e-12
    )


@pytest.mark.parametrize(
    (
        "name",
        "expected_apoapsis",
        "expected_period",
        "expected_time",
        "expected_mean_anomaly",
    ),
    [
        (
            "kerbin-example",
            0.9620157867739283,
            2.1541772038232403,
            0.2375974004923687,
            2.448581654712432,
        ),
        (
            "kerbin-first-example",
            0.9790183534641476,
            2.277970283505132,
            0.3462435056862553,
            2.186570490485096,
        ),
        (
            "kerbin-coast-first-example",
            1.001629074246341,
            2.5760223121604957,
            0.418062970902478,
            2.1218937543002507,
        ),
    ],
)
def test_orbital_helpers_preserve_reference_values(
    name: str,
    expected_apoapsis: float,
    expected_period: float,
    expected_time: float,
    expected_mean_anomaly: float,
) -> None:
    position, velocity, segments = kerbin_examples()[name]
    problem = _prepare_problem(
        position,
        velocity,
        segments,
        3.5316e12,
        DEFAULT_TARGET_RADIUS,
        DEFAULT_MAX_BURN_TIME,
    )
    parameters = OrbitalParameters(problem.radius, problem.velocity)
    apo_state, time_to_apoapsis = parameters.coast_to_apoapsis()

    assert parameters.apoapsis == pytest.approx(expected_apoapsis, abs=2e-10)
    assert parameters.mean_anomaly == pytest.approx(expected_mean_anomaly, abs=2e-10)
    assert parameters.period() == pytest.approx(expected_period, abs=2e-10)
    assert time_to_apoapsis == pytest.approx(expected_time, abs=2e-10)
    assert apo_state.radius == pytest.approx(expected_apoapsis, abs=2e-10)
    assert apo_state.velocity[0] == pytest.approx(0.0, abs=2e-12)


def test_kerbin_example_selects_expected_family(
    first_oracle_result: OracleResult,
) -> None:
    assert first_oracle_result.selected.name == "lawden"
    assert first_oracle_result.selected.success
    assert first_oracle_result.selected.residual_norm < 3e-6
    assert (
        first_oracle_result.selected.final_mass_kg(first_oracle_result.problem)
        is not None
    )
    assert (
        first_oracle_result.selected.fuel_consumed_kg(first_oracle_result.problem)
        is not None
    )
    assert (
        first_oracle_result.selected.delta_v_m_per_s(first_oracle_result.problem)
        is not None
    )
    assert first_oracle_result.selected.phases


def test_result_contains_plane_and_solve_ivp_style_phases(
    first_oracle_result: OracleResult,
) -> None:
    result = first_oracle_result
    record = result.to_dict()
    selected = cast(dict[str, object], record["selected"])
    assert "initial_costate" in selected
    assert "phases" in selected
    phases = cast(list[dict[str, object]], selected["phases"])
    assert phases
    first_phase = phases[0]
    time_tau = cast(list[object], first_phase["time_tau"])
    state = cast(list[object], first_phase["state"])
    control = cast(dict[str, object], first_phase["control"])
    throttle = cast(list[object], control["throttle"])
    plane = cast(dict[str, object], record["plane"])
    assert len(time_tau) == len(state)
    assert len(time_tau) == len(throttle)
    assert len(cast(list[object], plane["r_hat"])) == 3


def test_mass_fuel_and_delta_v_are_consistent(
    first_oracle_result: OracleResult,
) -> None:
    candidate = first_oracle_result.selected
    initial_mass = first_oracle_result.problem.physical.mass0_kg
    ve = first_oracle_result.problem.physical.ve_m_per_s

    final_mass = candidate.final_mass_kg(first_oracle_result.problem)
    fuel = candidate.fuel_consumed_kg(first_oracle_result.problem)
    delta_v = candidate.delta_v_m_per_s(first_oracle_result.problem)
    assert final_mass is not None
    assert fuel == pytest.approx(initial_mass - final_mass)
    assert delta_v == pytest.approx(ve * math.log(initial_mass / final_mass))


def test_polar_dynamics_match_independent_cartesian_dynamics() -> None:
    state = np.array([0.83, 0.21, 0.92, 1.0], dtype=float)
    gamma = 1.8
    kappa = 1.35
    throttle = 0.8
    alpha = 0.37
    duration = 0.17
    polar_solution = solve_ivp(
        _polar_rhs,
        (0.0, duration),
        state,
        args=(gamma, kappa, throttle, alpha),
        method="DOP853",
        rtol=2e-11,
        atol=2e-13,
    )

    def cartesian_rhs(_time: float, cart_state: np.ndarray) -> np.ndarray:
        position = cart_state[:2]
        velocity = cart_state[2:4]
        mass = cart_state[4]
        radius = float(np.linalg.norm(position))
        radial = position / radius
        tangent = np.array([-radial[1], radial[0]])
        acceleration = throttle * gamma / mass
        total_acceleration = -position / radius**3 + acceleration * (
            math.cos(alpha) * radial + math.sin(alpha) * tangent
        )
        return np.array(
            [
                velocity[0],
                velocity[1],
                total_acceleration[0],
                total_acceleration[1],
                -throttle * gamma / kappa,
            ],
            dtype=float,
        )

    cartesian_solution = solve_ivp(
        cartesian_rhs,
        (0.0, duration),
        np.array([state[0], 0.0, state[1], state[2], state[3]]),
        method="DOP853",
        rtol=2e-11,
        atol=2e-13,
    )
    cartesian_final = cartesian_solution.y[:, -1]
    position = cartesian_final[:2]
    velocity = cartesian_final[2:4]
    radius = float(np.linalg.norm(position))
    radial = position / radius
    tangent = np.array([-radial[1], radial[0]])
    recovered_polar = np.array(
        [
            radius,
            np.dot(velocity, radial),
            np.dot(velocity, tangent),
            cartesian_final[4],
        ],
        dtype=float,
    )
    np.testing.assert_allclose(polar_solution.y[:, -1], recovered_polar, atol=2e-10)


def test_degenerate_first_burn_is_rejected() -> None:
    position, velocity, segments = kerbin_examples()["kerbin-coast-first-example"]
    problem = _prepare_problem(
        position,
        velocity,
        segments,
        3.5316e12,
        DEFAULT_TARGET_RADIUS,
        DEFAULT_MAX_BURN_TIME,
    )
    parameters = np.array(
        [
            0.0,
            1.0,
            0.0,
            0.5 * problem.min_phase_duration,
            problem.min_phase_duration,
            problem.min_phase_duration,
        ],
        dtype=float,
    )

    assert not _valid_explicit(problem, "burn_coast_burn", parameters, np.zeros(6), [])


def test_lawden_plan_has_no_terminal_coast_or_short_phase(
    first_oracle_result: OracleResult,
) -> None:
    result = first_oracle_result
    lawden = next(
        candidate for candidate in result.candidates if candidate.name == "lawden"
    )

    assert lawden.success
    assert lawden.sequence == ("burn", "coast", "burn")
    assert lawden.phases[-1].kind == "burn"
    assert all(
        phase.duration_tau * result.problem.physical.time_scale_s
        >= MIN_PHASE_DURATION_SECONDS - 1e-9
        for phase in lawden.phases
    )


def test_already_targeted_state_returns_empty_plan() -> None:
    position = np.array([DEFAULT_TARGET_RADIUS, 0.0, 0.0])
    velocity = np.array([0.0, math.sqrt(3.5316e12 / DEFAULT_TARGET_RADIUS), 0.0])
    segments = [ExampleSegment(215000.0, 3138.128, 13885.650390625)]

    result = find_oracle(position, velocity, segments)

    assert result.selected.name == "empty"
    assert result.selected.sequence == ()
    assert result.selected.fuel_consumed_kg(result.problem) == pytest.approx(0.0)
    assert result.selected.delta_v_m_per_s(result.problem) == pytest.approx(0.0)
    assert result.selected.phases == []


def test_unsupported_lawden_sequence_is_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_lawden(problem: object, _timing: object) -> Candidate:
        return Candidate(
            name="lawden",
            formulation="lawden-sigmoid-hard-polish",
            success=False,
            message="unsupported test sequence",
            parameters=np.zeros(4, dtype=float),
            residual=np.ones(4, dtype=float),
            final_state=None,
            final_eta=None,
            phases=[],
            powered_tau=0.0,
            final_tau=0.0,
            limit_hit=False,
            sequence=("burn", "burn"),
            diagnostics={
                "mass0": 1.0,
                "ve": 1.0,
                "phase_sequence": "burn/burn",
                "unsupported_sequence": True,
            },
        )

    monkeypatch.setattr(oracle_finder, "_solve_lawden", fake_lawden)
    position, velocity, segments = kerbin_examples()["kerbin-first-example"]

    with pytest.warns(RuntimeWarning, match="unsupported phase sequence"):
        result = find_oracle(position, velocity, segments)

    assert any("burn/burn" in message for message in result.warnings)


def test_all_candidates_failed_error_path(monkeypatch: pytest.MonkeyPatch) -> None:
    def failed_candidate(name: str) -> Candidate:
        return Candidate(
            name=name,
            formulation="test",
            success=False,
            message="forced failure",
            parameters=np.zeros(1, dtype=float),
            residual=np.full(1, 4.0),
            final_state=None,
            final_eta=None,
            phases=[],
            powered_tau=0.0,
            final_tau=0.0,
            limit_hit=False,
            sequence=(),
            diagnostics={"mass0": 1.0, "ve": 1.0},
        )

    def fail_explicit(problem: object, mode: str, timing: object) -> Candidate:
        del problem, timing
        return failed_candidate(mode)

    def fail_lawden(problem: object, timing: object) -> Candidate:
        del problem, timing
        return failed_candidate("lawden")

    monkeypatch.setattr(oracle_finder, "_solve_explicit_mode", fail_explicit)
    monkeypatch.setattr(oracle_finder, "_solve_lawden", fail_lawden)
    position, velocity, segments = kerbin_examples()["kerbin-first-example"]

    with pytest.raises(ValueError, match="all oracle candidates failed") as error:
        find_oracle(position, velocity, segments)

    message = str(error.value)
    assert "lawden" in message
    assert "burn_coast_burn" in message
    assert "coast_burn" in message
    assert "burn_only" in message


def test_lawden_succeeds_without_explicit_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failed_explicit(problem: object, mode: str, timing: object) -> Candidate:
        del problem, timing
        return Candidate(
            name=mode,
            formulation="test",
            success=False,
            message="forced failure",
            parameters=np.zeros(1, dtype=float),
            residual=np.full(1, 4.0),
            final_state=None,
            final_eta=None,
            phases=[],
            powered_tau=0.0,
            final_tau=0.0,
            limit_hit=False,
            sequence=(),
            diagnostics={"kappa": 1.0},
        )

    monkeypatch.setattr(oracle_finder, "_solve_explicit_mode", failed_explicit)
    position, velocity, segments = kerbin_examples()["kerbin-first-example"]

    result = find_oracle(position, velocity, segments)

    lawden = next(
        candidate for candidate in result.candidates if candidate.name == "lawden"
    )
    assert lawden.success
    assert result.selected.name == "lawden"
    seed_results = cast(list[object], lawden.diagnostics["seed_results"])
    assert len(seed_results) == 8


def test_limit_binding_candidate_warns() -> None:
    position, velocity, segments = kerbin_examples()["kerbin-first-example"]

    with pytest.warns(RuntimeWarning, match="max_burn_time"):
        result = find_oracle(position, velocity, segments, max_burn_time=80.0)

    assert result.warnings
    assert any("max_burn_time" in message for message in result.warnings)
