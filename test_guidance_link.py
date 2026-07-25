"""Unit tests for guidance_link.py."""

import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from guidance_link import (
    GuidanceCommand,
    GuidanceLink,
    evaluate_target,
)
from sim import OrbitalPlane

PLANE = OrbitalPlane(np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))


def _cmd(**overrides: object) -> GuidanceCommand:
    fields: dict[str, object] = {
        "ref_angle": 0.0,
        "a_coeff": 0.0,
        "b_coeff": 0.0,
        "t0": 0.0,
        "plane": PLANE,
    }
    fields.update(overrides)
    return GuidanceCommand(**fields)  # type: ignore[arg-type]


# ─── evaluate_target ───────────────────────────────────────────────────────


class TestEvaluateTarget:
    def test_pre_burn_holds_t0_attitude_with_zero_rate(self) -> None:
        cmd = _cmd(a_coeff=1.0, b_coeff=5.0, t0=10.0)
        thrust_dir, thrust_dir_dot = evaluate_target(cmd, ut=5.0)

        theta0 = math.atan(1.0)
        expected_dir = PLANE.from_plane(np.array([math.cos(theta0), math.sin(theta0)]))
        np.testing.assert_allclose(thrust_dir, expected_dir)
        np.testing.assert_allclose(thrust_dir_dot, np.zeros(3))

    def test_pre_burn_ignores_ut_progress_and_always_uses_t0(self) -> None:
        cmd = _cmd(a_coeff=0.3, b_coeff=2.0, t0=10.0)
        dir_early, _ = evaluate_target(cmd, ut=0.0)
        dir_late, _ = evaluate_target(cmd, ut=9.999)
        np.testing.assert_allclose(dir_early, dir_late)

    def test_during_burn_tracks_linear_tangent_law(self) -> None:
        cmd = _cmd(ref_angle=0.2, a_coeff=0.1, b_coeff=0.5, t0=100.0)
        ut = 103.0
        t = ut - cmd.t0
        thrust_dir, thrust_dir_dot = evaluate_target(cmd, ut)

        tan_val = cmd.a_coeff + cmd.b_coeff * t
        theta = cmd.ref_angle + math.atan(tan_val)
        expected_dir = PLANE.from_plane(np.array([math.cos(theta), math.sin(theta)]))
        np.testing.assert_allclose(thrust_dir, expected_dir)

        dtheta_dt = cmd.b_coeff / (1.0 + tan_val**2)
        expected_dot = PLANE.from_plane(
            np.array([-math.sin(theta) * dtheta_dt, math.cos(theta) * dtheta_dt])
        )
        np.testing.assert_allclose(thrust_dir_dot, expected_dot)

    def test_matches_original_inline_formulas(self) -> None:
        """Equivalence check against the exact formulas that used to live
        inline in GravityTurn.circularize()'s two branches, over random
        parameters, to prove the refactor didn't change behavior."""
        rng = np.random.default_rng(1234)
        for _ in range(50):
            ref_angle = rng.uniform(-math.pi, math.pi)
            a_coeff = rng.uniform(-5, 5)
            b_coeff = rng.uniform(-5, 5)
            t0 = rng.uniform(-100, 100)
            ut = t0 + rng.uniform(-20, 20)
            cmd = _cmd(ref_angle=ref_angle, a_coeff=a_coeff, b_coeff=b_coeff, t0=t0)

            if ut < t0:
                # Original "coast" branch.
                theta = ref_angle + math.atan(a_coeff)
                expected_dir = PLANE.from_plane(
                    np.array([math.cos(theta), math.sin(theta)])
                )
                expected_dot = np.zeros(3)
            else:
                # Original "burn" branch.
                t = ut - t0
                tan_val = a_coeff + b_coeff * t
                theta = ref_angle + math.atan(tan_val)
                expected_dir = PLANE.from_plane(
                    np.array([math.cos(theta), math.sin(theta)])
                )
                dtheta_dt = b_coeff / (1.0 + tan_val**2)
                expected_dot = PLANE.from_plane(
                    np.array(
                        [-math.sin(theta) * dtheta_dt, math.cos(theta) * dtheta_dt]
                    )
                )

            thrust_dir, thrust_dir_dot = evaluate_target(cmd, ut)
            np.testing.assert_allclose(thrust_dir, expected_dir, atol=1e-12)
            np.testing.assert_allclose(thrust_dir_dot, expected_dot, atol=1e-12)


# ─── GuidanceLink ──────────────────────────────────────────────────────────


class TestGuidanceLink:
    def test_default_initial_command_is_none(self) -> None:
        link = GuidanceLink()
        assert link.get() is None

    def test_set_then_get_round_trips(self) -> None:
        link = GuidanceLink()
        cmd = _cmd(ref_angle=1.5)
        link.set(cmd)
        assert link.get() is cmd

    def test_set_none_disables(self) -> None:
        link = GuidanceLink()
        cmd = _cmd(ref_angle=1.5, a_coeff=2.0, b_coeff=3.0, t0=4.0)
        link.set(cmd)
        assert link.get() is cmd

        link.set(None)
        assert link.get() is None

    def test_concurrent_set_get_never_raises_or_tears(self) -> None:
        link = GuidanceLink()
        commands = [_cmd(ref_angle=float(i)) for i in range(200)]
        errors: list[Exception] = []

        def setter() -> None:
            for cmd in commands:
                link.set(cmd)

        def getter() -> None:
            for _ in range(500):
                got = link.get()
                # A torn/partial object would fail this identity-consistency
                # check (e.g. ref_angle from one command, a_coeff from
                # another); since GuidanceCommand is frozen/immutable and
                # always swapped in whole, `got` must always be `is` one of
                # the exact objects passed to set() (or the initial default).
                if got is not None and got not in commands:
                    errors.append(RuntimeError(f"unexpected object: {got!r}"))

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(setter) for _ in range(2)] + [
                pool.submit(getter) for _ in range(2)
            ]
            for f in futures:
                f.result()

        assert not errors


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
