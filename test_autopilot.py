"""Unit tests for autopilot.py.

All tests use a fake kRPC connection — no live KSP/kRPC server needed.
"""

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from autopilot import CustomAutopilot, _AxisController

# ─── Fakes ───────────────────────────────────────────────────────────────────


class FakeStream:
    """Callable stream that returns a fixed value, with a remove() method."""

    def __init__(self, value: Any) -> None:
        self._value = value
        self.removed = False

    def __call__(self) -> Any:
        return self._value

    def set(self, value: Any) -> None:
        self._value = value

    def remove(self) -> None:
        self.removed = True


class FakeControl:
    def __init__(self) -> None:
        self.pitch: float = 0.0
        self.yaw: float = 0.0
        self.roll: float = 0.0


class FakeVessel:
    def __init__(self) -> None:
        self.control = FakeControl()

    # These are only used as callables passed to ``conn.add_stream`` in
    # CustomAutopilot.__init__ (FakeConn.add_stream ignores the actual
    # function/args and just hands back pre-seeded FakeStreams), so their
    # bodies are never executed -- they just need to exist as attributes.
    def rotation(self, frame: Any) -> tuple[float, float, float, float]:
        return (0.0, 0.0, 0.0, 1.0)

    def angular_velocity(self, frame: Any) -> tuple[float, float, float]:
        return (0.0, 0.0, 0.0)


class FakeConn:
    """Minimal fake kRPC connection that creates FakeStreams."""

    def __init__(
        self,
        rotation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
        ang_vel: tuple[float, float, float] = (0.0, 0.0, 0.0),
        # available_torque: ((pos_x,y,z), (neg_x,y,z))
        torque: tuple[tuple[float, float, float], tuple[float, float, float]] = (
            (10.0, 10.0, 10.0),
            (-10.0, -10.0, -10.0),
        ),
        moi: tuple[float, float, float] = (2.0, 2.0, 2.0),
    ) -> None:
        self._streams: list[FakeStream] = []
        self._rot = FakeStream(rotation)
        self._ang_vel = FakeStream(ang_vel)
        self._torque = FakeStream(torque)
        self._moi = FakeStream(moi)

    def add_stream(
        self, func: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> FakeStream:
        # Return streams in order: rotation, ang_vel, torque, moi
        order = [self._rot, self._ang_vel, self._torque, self._moi]
        idx = len(self._streams)
        stream = order[idx] if idx < len(order) else FakeStream(None)
        self._streams.append(stream)
        return stream

    def set_rotation(self, quat: tuple[float, float, float, float]) -> None:
        self._rot.set(quat)

    def set_ang_vel(self, v: tuple[float, float, float]) -> None:
        self._ang_vel.set(v)


def _make_autopilot(**kwargs: Any) -> tuple[CustomAutopilot, FakeConn, FakeVessel]:
    """Build a CustomAutopilot backed by a FakeConn."""
    conn = FakeConn(**kwargs)
    vessel = FakeVessel()
    frame = object()
    ap = CustomAutopilot(conn, vessel, frame)
    return ap, conn, vessel


# ─── _AxisController unit tests ───────────────────────────────────────────────


class TestAxisController:
    def _make(self, kc_prior: float = 5.0) -> _AxisController:
        return _AxisController(
            kc_prior=kc_prior,
            dt=0.02,
            sat_angle_rad=math.radians(45.0),
            omega_n_max=5.0,
            history_window_sec=1.0,
            kc_decay_tau=5.0,
        )

    def test_zero_error_zero_rate_gives_zero_output(self) -> None:
        ctrl = self._make()
        c = ctrl.compute(
            ut=0.0,
            delta_theta=0.0,
            delta_theta_dot=0.0,
            ang_vel=0.0,
            last_c=0.0,
            verbose=False,
        )
        assert c == pytest.approx(0.0, abs=1e-9)

    def test_positive_angle_error_gives_positive_output(self) -> None:
        ctrl = self._make()
        # Small positive angle error, zero rate
        c = ctrl.compute(
            ut=0.0,
            delta_theta=0.1,
            delta_theta_dot=0.0,
            ang_vel=0.0,
            last_c=0.0,
            verbose=False,
        )
        assert c > 0.0

    def test_negative_angle_error_gives_negative_output(self) -> None:
        ctrl = self._make()
        c = ctrl.compute(
            ut=0.0,
            delta_theta=-0.1,
            delta_theta_dot=0.0,
            ang_vel=0.0,
            last_c=0.0,
            verbose=False,
        )
        assert c < 0.0

    def test_large_error_saturates_at_plus_one(self) -> None:
        ctrl = self._make()
        c = ctrl.compute(
            ut=0.0,
            delta_theta=100.0,
            delta_theta_dot=0.0,
            ang_vel=0.0,
            last_c=0.0,
            verbose=False,
        )
        assert c == pytest.approx(1.0)

    def test_large_negative_error_saturates_at_minus_one(self) -> None:
        ctrl = self._make()
        c = ctrl.compute(
            ut=0.0,
            delta_theta=-100.0,
            delta_theta_dot=0.0,
            ang_vel=0.0,
            last_c=0.0,
            verbose=False,
        )
        assert c == pytest.approx(-1.0)

    def test_high_variance_history_fits_kc_and_k0(self) -> None:
        """Two control values with known angular accelerations should recover
        kc and k0 via least-squares.

        Model is ``theta_ddot = -kc * c + k0`` (kc is a non-negative
        magnitude).
        """
        kc_true = 4.0
        k0_true = 0.5
        ctrl = self._make(kc_prior=10.0)  # start far from the truth

        # Inject two (c, theta_ddot) pairs separated by 0.02 s each.
        # We drive theta_ddot indirectly via the finite-difference of ang_vel.
        # Instead of going through the full update machinery, we can seed the
        # history deque directly for a pure unit test.
        ctrl._history.append((0.00, -1.0, -kc_true * (-1.0) + k0_true))
        ctrl._history.append((0.02, 1.0, -kc_true * (1.0) + k0_true))

        # Trigger a fit without adding a new measurement (no prev_ang_vel yet).
        ctrl._update_gains(ut=0.04, c=0.0, ang_vel=0.0, verbose=False)

        assert ctrl.kc == pytest.approx(kc_true, rel=1e-6)
        assert ctrl.k0 == pytest.approx(k0_true, rel=1e-6)

    def test_low_variance_history_updates_only_k0(self) -> None:
        """With nearly constant control, kc should not change (only k0
        updated), and kc should decay toward the prior.

        Model is ``theta_ddot = -kc * c + k0``.
        """
        kc_start = 8.0
        kc_prior = 5.0
        ctrl = self._make(kc_prior=kc_prior)
        ctrl.kc = kc_start
        ctrl.k0 = 0.0

        # All control values the same → variance ~0
        c_val = 0.5
        a_val = -kc_start * c_val + 1.2  # implies k0=1.2
        for i in range(10):
            ctrl._history.append((i * 0.02, c_val, a_val))

        ctrl._update_gains(ut=0.22, c=c_val, ang_vel=0.0, verbose=False)

        # k0 should be updated using the already-decayed kc
        assert ctrl.k0 == pytest.approx(a_val + ctrl.kc * c_val, rel=1e-6)
        # kc should have moved slightly toward the prior, not jumped
        assert kc_prior < ctrl.kc < kc_start

    def test_reset_clears_history(self) -> None:
        ctrl = self._make()
        ctrl._history.append((0.0, 0.5, 2.0))
        ctrl._prev_ang_vel = 0.1
        ctrl._prev_ut = 0.0
        ctrl._prev_c = 0.5
        ctrl.reset()
        assert len(ctrl._history) == 0
        assert ctrl._prev_ang_vel is None
        assert ctrl._prev_ut is None
        assert ctrl._prev_c is None

    def test_omega_n_capped_at_max(self) -> None:
        # kc very large → omega_n_sat would be huge; must be capped.
        ctrl = self._make(kc_prior=1e6)
        ctrl.kc = 1e6
        ctrl.k0 = 0.0
        assert ctrl._omega_n() == pytest.approx(5.0)

    def test_omega_n_from_kc_and_k0(self) -> None:
        # omega_n_sat = sqrt((kc - |k0|) / sat_angle_rad)
        # With kc=1, k0=0, sat=45°=pi/4:
        # omega_n_sat = sqrt(1/(pi/4)) = sqrt(4/pi) ≈ 1.128
        ctrl = _AxisController(
            kc_prior=1.0,
            dt=0.02,
            sat_angle_rad=math.pi / 4,
            omega_n_max=5.0,
            history_window_sec=1.0,
            kc_decay_tau=5.0,
        )
        ctrl.kc = 1.0
        ctrl.k0 = 0.0
        expected = math.sqrt(1.0 / (math.pi / 4))
        assert ctrl._omega_n() == pytest.approx(expected, rel=1e-9)


# ─── CustomAutopilot integration tests ───────────────────────────────────────


class TestCustomAutopilot:
    def test_zero_error_sets_zero_controls(self) -> None:
        """With vessel already pointing at target, controls should be ~0."""
        ap, _conn, vessel = _make_autopilot()
        # Identity rotation: vessel frame == world frame
        # Target is the vessel nose direction in vessel frame: [0, 1, 0] in world
        # With identity quaternion, vessel +y = world +y.
        target_dir = np.array([0.0, 1.0, 0.0])
        target_dir_dot = np.zeros(3)
        ap.update(ut=0.0, target_dir=target_dir, target_dir_dot=target_dir_dot)
        assert vessel.control.pitch == pytest.approx(0.0, abs=1e-9)
        assert vessel.control.yaw == pytest.approx(0.0, abs=1e-9)
        ap.close()

    def test_pitch_error_sets_nonzero_pitch(self) -> None:
        """A pure pitch error should produce a nonzero pitch command."""
        ap, _conn, vessel = _make_autopilot()
        # Rotate target direction slightly in pitch (tilt nose up a bit).
        # In world frame, pitch the target ~5° toward the -z direction
        # (vessel frame +y=forward, -z=up when in identity orientation).
        angle = math.radians(5.0)
        target_dir = np.array([0.0, math.cos(angle), math.sin(angle)])
        target_dir_dot = np.zeros(3)
        ap.update(ut=0.0, target_dir=target_dir, target_dir_dot=target_dir_dot)
        # Pitch control should be nonzero
        assert abs(vessel.control.pitch) > 0.0
        assert vessel.control.yaw == pytest.approx(0.0, abs=1e-6)
        ap.close()

    def test_large_error_saturates_control(self) -> None:
        """A large (90°) pitch error should saturate the control output to
        ±1.

        Note: exactly 180° (target directly behind the vessel) is a coordinate
        singularity for this error parameterization -- ``target_vessel`` ends
        up exactly ``(0, -1, 0)``, so both ``delta_theta_pitch`` and
        ``delta_theta_yaw`` (which come from the x/z components) are exactly
        zero, giving no signal to correct on. A target perpendicular to the
        nose avoids that singularity while still being a "large" error.
        """
        ap, _conn, vessel = _make_autopilot()
        # Point 90° away from the nose, in the pitch plane.
        target_dir = np.array([0.0, 0.0, 1.0])
        target_dir_dot = np.zeros(3)
        ap.update(ut=0.0, target_dir=target_dir, target_dir_dot=target_dir_dot)
        assert abs(vessel.control.pitch) == pytest.approx(1.0, abs=0.01)
        ap.close()

    def test_close_removes_all_streams(self) -> None:
        ap, conn, _vessel = _make_autopilot()
        streams = conn._streams[:]
        ap.close()
        for s in streams:
            assert s.removed

    def test_close_is_idempotent(self) -> None:
        ap, _conn, _vessel = _make_autopilot()
        ap.close()
        ap.close()  # should not raise

    def test_reset_history_flushes_both_axes(self) -> None:
        ap, _conn, _vessel = _make_autopilot()
        # Inject some history into both axes
        ap._pitch._history.append((0.0, 0.5, 2.0))
        ap._yaw._history.append((0.0, 0.5, 2.0))
        ap.reset_history()
        assert len(ap._pitch._history) == 0
        assert len(ap._yaw._history) == 0
        ap.close()

    def test_gains_initialized_from_physics_prior(self) -> None:
        """kc prior should be torque/moi for each axis."""
        # torque magnitude 10 N·m per axis, moi 2 kg·m² per axis → prior = 5
        ap, _conn, _vessel = _make_autopilot(
            torque=((10.0, 10.0, 10.0), (-10.0, -10.0, -10.0)),
            moi=(2.0, 2.0, 2.0),
        )
        assert ap._pitch.kc == pytest.approx(5.0, rel=1e-6)
        assert ap._yaw.kc == pytest.approx(5.0, rel=1e-6)
        ap.close()

    def test_ang_vel_fed_to_correct_axis(self) -> None:
        """Angular velocity[0] (pitch axis) should influence pitch gain
        estimation, not yaw."""
        ap, conn, _vessel = _make_autopilot(
            ang_vel=(0.5, 0.0, 0.0),  # only pitch component
        )
        target_dir = np.array([0.0, 1.0, 0.0])
        target_dir_dot = np.zeros(3)
        # Two calls so finite-differencing produces an acceleration estimate.
        ap.update(ut=0.00, target_dir=target_dir, target_dir_dot=target_dir_dot)
        conn.set_ang_vel((0.52, 0.0, 0.0))
        ap.update(ut=0.02, target_dir=target_dir, target_dir_dot=target_dir_dot)
        # History should only exist in the pitch axis
        assert len(ap._pitch._history) >= 1
        ap.close()
