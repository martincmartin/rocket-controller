"""Tests for the pluggable thrust profiles in sim_experiment.py."""

import numpy as np
import pytest

from sim_experiment import (
    LinearTangentProfile,
    OrbitalPlane,
    PerStageLinearTangentProfile,
    RocketSegment,
    Simulator,
)

MU = 3.5316e12
R3D = np.array([424370.58766631, -1093.08696926, -470992.64951719])
V3D = np.array([723.81414935, -1.2033429, -122.60883836])


@pytest.fixture
def sim() -> Simulator:
    return Simulator(
        MU,
        body_radius=600_000.0,
        target_altitude=80_000.0,
        segments=[
            RocketSegment(
                "Swivel",
                ve=3138.1279999999997,
                thrust=215_000.0,
                max_burn_time=59.0500960010656,
                initial_mass=13885.650390625,
                last_segment_of_stage=True,
            ),
            RocketSegment(
                "Terrier",
                ve=3383.2942499999995,
                thrust=60_000.0,
                max_burn_time=112.77647255563578,
                initial_mass=4449.999407536325,
                last_segment_of_stage=True,
            ),
        ],
        staging_duration=2.0,
    )


def test_profile_parameterizations(sim: Simulator) -> None:
    single = LinearTangentProfile()
    per_stage = PerStageLinearTangentProfile()

    assert len(single.initial_parameters(sim, 12.0, 120.0)) == 4
    assert len(single.parameter_bounds(sim, 72.0)) == 4
    assert len(per_stage.initial_parameters(sim, 12.0, 120.0)) == 6
    assert len(per_stage.parameter_bounds(sim, 72.0)) == 6
    assert per_stage.parameter_names(sim)[-1] == "burn time"


def test_per_stage_profile_uses_each_stage_coefficients(
    sim: Simulator, monkeypatch: pytest.MonkeyPatch
) -> None:
    plane = OrbitalPlane(R3D, V3D)
    r = plane.to_plane(R3D)
    v = plane.to_plane(V3D)
    coast = sim.solve_coast((0.0, 1.0), r, v)
    assert coast.sol is not None

    coefficients: list[tuple[float, float]] = []
    original = sim.solve_linear_tangent

    def spy(*args: object, **kwargs: object):
        coefficients.append((float(args[5]), float(args[6])))
        return original(*args, **kwargs)

    monkeypatch.setattr(sim, "solve_linear_tangent", spy)
    burn_time = sim.segments[0].max_burn_time + sim.staging_duration + 5.0
    PerStageLinearTangentProfile().evaluate(
        sim,
        coast.sol,
        0.0,
        (0.0, 0.1, 0.2, 0.3, 0.4, burn_time),
    )

    assert coefficients == [(0.1, 0.2), (0.3, 0.4)]


def test_profiles_match_when_coefficients_are_shared(sim: Simulator) -> None:
    plane = OrbitalPlane(R3D, V3D)
    r = plane.to_plane(R3D)
    v = plane.to_plane(V3D)
    coast = sim.solve_coast((0.0, 1.0), r, v)
    assert coast.sol is not None
    burn_time = sim.segments[0].max_burn_time + sim.staging_duration + 5.0

    single = LinearTangentProfile().evaluate(
        sim, coast.sol, 0.0, (0.0, 0.1, 0.2, burn_time)
    )
    per_stage = PerStageLinearTangentProfile().evaluate(
        sim, coast.sol, 0.0, (0.0, 0.1, 0.2, 0.1, 0.2, burn_time)
    )

    np.testing.assert_allclose(single.burn_result.r, per_stage.burn_result.r)
    np.testing.assert_allclose(single.burn_result.v, per_stage.burn_result.v)
    assert single.burn_result.mass == pytest.approx(per_stage.burn_result.mass)
    assert single.burn_time == pytest.approx(per_stage.burn_time)


def test_profile_neutral_optimizer_stores_selected_profile(sim: Simulator) -> None:
    profile = LinearTangentProfile()
    plan = sim.find_thrust_profile_params(
        R3D,
        V3D,
        72.12194913376851,
        profile,
        verbose=False,
    )

    assert plan.thrust_profile is profile
    assert len(plan.profile_parameters) == 4
    assert plan.coast_time == pytest.approx(plan.profile_parameters[0])
    assert plan.burn_time == pytest.approx(plan.profile_parameters[-1])
