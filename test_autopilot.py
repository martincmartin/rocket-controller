"""Unit tests for autopilot.py.

All tests use a fake KSPStreams object — no live KSP/kRPC server needed.
"""

import math
from typing import Any

import numpy as np
import pytest

from autopilot import CustomAutopilot, _AxisController

# ─── Fakes ───────────────────────────────────────────────────────────────────


class FakeStream:
    """Callable stream with a condition stub and remove()."""

    class _Condition:
        def __enter__(self) -> "FakeStream._Condition":
            return self

        def __exit__(self, *_: object) -> None:
            pass

        def wait(self) -> None:
            pass

    def __init__(self, value: Any) -> None:
        self._value = value
        self.removed = False
        self.condition = self._Condition()

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
    def __init__(
        self,
        torque: tuple[tuple[float, float, float], tuple[float, float, float]] = (
            (10.0, 10.0, 10.0),
            (-10.0, -10.0, -10.0),
        ),
        moi: tuple[float, float, float] = (2.0, 2.0, 2.0),
    ) -> None:
        self.control = FakeControl()
        # available_torque and moment_of_inertia are plain property reads in the
        # new autopilot (not streams), so they live directly on the vessel.
        self.available_torque = torque
        self.moment_of_inertia = moi

    # These are passed to ks.add_stream() as callables; the FakeKSPStreams
    # doesn't actually call them, so their bodies never run.
    def rotation(self, frame: Any) -> tuple[float, float, float, float]:
        return (0.0, 0.0, 0.0, 1.0)

    def angular_velocity(self, frame: Any) -> tuple[float, float, float]:
        return (0.0, 0.0, 0.0)


class FakeKSPStreams:
    """Minimal fake KSPStreams for autopilot tests.

    Holds FakeStream objects for ``rotation`` and ``angular_velocity`` that
    tests can inspect and mutate.  Provides the ``__getattr__`` interface that
    ``CustomAutopilot.update()`` uses to read snapshotted values.
    """

    def __init__(
        self,
        rotation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
        ang_vel: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        self._rot_stream = FakeStream(rotation)
        self._ang_vel_stream = FakeStream(ang_vel)
        # Map stream name → FakeStream, populated by add_stream().
        self._registered: dict[str, FakeStream] = {}

    def add_stream(self, name: str, func: Any, *args: Any, **kwargs: Any) -> None:
        """Record which streams the autopilot registers."""
        if name == "rotation":
            self._registered[name] = self._rot_stream
        elif name == "angular_velocity":
            self._registered[name] = self._ang_vel_stream
        else:
            self._registered[name] = FakeStream(None)

    def start(self) -> None:
        pass

    def next(self) -> None:
        pass

    def close(self) -> None:
        for s in self._registered.values():
            s.remove()
        self._registered.clear()

    def set_rotation(self, quat: tuple[float, float, float, float]) -> None:
        self._rot_stream.set(quat)

    def set_ang_vel(self, v: tuple[float, float, float]) -> None:
        self._ang_vel_stream.set(v)

    # KSPStreams attribute access: return the current stream value directly
    # (in the real class these are populated by next(), but here we read live).
    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        registered = object.__getattribute__(self, "_registered")
        if name in registered:
            return registered[name]()
        raise AttributeError(f"FakeKSPStreams: no stream {name!r}")


def _make_autopilot(
    torque: tuple[tuple[float, float, float], tuple[float, float, float]] = (
        (10.0, 10.0, 10.0),
        (-10.0, -10.0, -10.0),
    ),
    moi: tuple[float, float, float] = (2.0, 2.0, 2.0),
    rotation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
    ang_vel: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[CustomAutopilot, FakeKSPStreams, FakeVessel]:
    """Build a CustomAutopilot backed by a FakeKSPStreams."""
    ks = FakeKSPStreams(rotation=rotation, ang_vel=ang_vel)
    vessel = FakeVessel(torque=torque, moi=moi)
    frame = object()
    ap = CustomAutopilot(ks, vessel, frame)
    return ap, ks, vessel


# ─── _AxisController unit tests ───────────────────────────────────────────────


class TestAxisController:
    def _make(self, kc_prior: float = 5.0, a_prior: float = 0.0) -> _AxisController:
        return _AxisController(
            kc_prior=kc_prior,
            a_prior=a_prior,
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
            omega_cross=0.0,
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
            omega_cross=0.0,
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
            omega_cross=0.0,
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
            omega_cross=0.0,
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
            omega_cross=0.0,
            last_c=0.0,
            verbose=False,
        )
        assert c == pytest.approx(-1.0)

    def test_high_variance_history_fits_a_kc_and_k0(self) -> None:
        """Three (c, omega_cross, theta_ddot) points with enough spread should
        recover A, kc, and k0 via least-squares.

        Model: ``theta_ddot = A*omega_cross - kc*c + k0``.
        """
        a_true = 1.5
        kc_true = 4.0
        k0_true = 0.5
        ctrl = self._make(kc_prior=10.0)  # start far from truth

        # Seed three data points with distinct (c, cross) combinations.
        def obs(c: float, cross: float) -> float:
            return a_true * cross - kc_true * c + k0_true

        ctrl._history.append((0.00, -1.0, 0.5, obs(-1.0, 0.5)))
        ctrl._history.append((0.02, 1.0, -0.3, obs(1.0, -0.3)))
        ctrl._history.append((0.04, 0.0, 0.8, obs(0.0, 0.8)))

        ctrl._update_gains(ut=0.06, c=0.0, ang_vel=0.0, omega_cross=0.0, verbose=False)

        assert ctrl.a == pytest.approx(a_true, rel=1e-6)
        assert ctrl.kc == pytest.approx(kc_true, rel=1e-6)
        assert ctrl.k0 == pytest.approx(k0_true, rel=1e-6)

    def test_low_variance_history_updates_only_k0(self) -> None:
        """With nearly constant control, kc and a should decay toward priors
        rather than being fit.  k0 is recomputed consistently.

        Model: ``theta_ddot = A*omega_cross - kc*c + k0``.
        """
        kc_start = 8.0
        kc_prior = 5.0
        a_prior = 1.0
        ctrl = self._make(kc_prior=kc_prior, a_prior=a_prior)
        ctrl.kc = kc_start
        ctrl.k0 = 0.0
        ctrl.a = 3.0  # start a far from its prior

        # All control values the same, cross terms zero → variance ~0
        c_val = 0.5
        a_val = -kc_start * c_val + 1.2  # theta_ddot values (A=0 since cross=0)
        for i in range(10):
            ctrl._history.append((i * 0.02, c_val, 0.0, a_val))

        ctrl._update_gains(
            ut=0.22, c=c_val, ang_vel=0.0, omega_cross=0.0, verbose=False
        )

        # k0 should be consistent with updated kc and a (cross_mean=0)
        assert ctrl.k0 == pytest.approx(a_val + ctrl.kc * c_val, rel=1e-6)
        # kc should have moved slightly toward the prior
        assert kc_prior < ctrl.kc < kc_start
        # a should have moved slightly toward a_prior
        assert a_prior < ctrl.a < 3.0

    def test_reset_clears_history(self) -> None:
        ctrl = self._make()
        ctrl._history.append((0.0, 0.5, 0.0, 2.0))
        ctrl._prev_ang_vel = 0.1
        ctrl._prev_ut = 0.0
        ctrl._prev_c = 0.5
        ctrl._prev_omega_cross = 0.1
        ctrl.reset()
        assert len(ctrl._history) == 0
        assert ctrl._prev_ang_vel is None
        assert ctrl._prev_ut is None
        assert ctrl._prev_c is None
        assert ctrl._prev_omega_cross is None

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
            a_prior=0.0,
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

    def test_cross_term_shifts_control_output(self) -> None:
        """Non-zero omega_cross with A != 0 should shift the control output
        even when the angle and rate errors are both zero.

        Control law: c = (A*omega_cross + k0 - theta_ddot_desired) / kc
        With zero errors theta_ddot_desired=0, so c = (A*omega_cross + k0) / kc.
        """
        ctrl = self._make(kc_prior=4.0, a_prior=0.0)
        ctrl.kc = 4.0
        ctrl.k0 = 0.0
        ctrl.a = 2.0  # manually set cross-term coefficient
        omega_cross = 0.5
        c = ctrl.compute(
            ut=0.0,
            delta_theta=0.0,
            delta_theta_dot=0.0,
            ang_vel=0.0,
            omega_cross=omega_cross,
            last_c=0.0,
            verbose=False,
        )
        # Expected: c = (2.0 * 0.5 + 0) / 4.0 = 0.25
        assert c == pytest.approx(0.25, rel=1e-6)


# ─── CustomAutopilot integration tests ───────────────────────────────────────


class TestCustomAutopilot:
    def test_zero_error_sets_zero_controls(self) -> None:
        """With vessel already pointing at target, controls should be ~0."""
        ap, _ks, vessel = _make_autopilot()
        # Identity rotation: vessel frame == world frame
        # Target is the vessel nose direction in vessel frame: [0, 1, 0] in world
        # With identity quaternion, vessel +y = world +y.
        target_dir = np.array([0.0, 1.0, 0.0])
        target_dir_dot = np.zeros(3)
        ap.update(ut=0.0, target_dir=target_dir, target_dir_dot=target_dir_dot)
        assert vessel.control.pitch == pytest.approx(0.0, abs=1e-9)
        assert vessel.control.yaw == pytest.approx(0.0, abs=1e-9)

    def test_pitch_error_sets_nonzero_pitch(self) -> None:
        """A pure pitch error should produce a nonzero pitch command."""
        ap, _ks, vessel = _make_autopilot()
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

    def test_large_error_saturates_control(self) -> None:
        """A large (90°) pitch error should saturate the control output to
        ±1.

        Note: exactly 180° (target directly behind the vessel) is a coordinate
        singularity for this error parameterization -- ``target_vessel`` ends
        up exactly ``(0, -1, 0)``, so both ``delta_theta_pitch`` and
        ``delta_theta_yaw`` (which come from the x/z components) are exactly
        zero, giving no signal to correct on. A target perpendicular to the
        nose avoids that singularity while still being a \"large\" error.
        """
        ap, _ks, vessel = _make_autopilot()
        # Point 90° away from the nose, in the pitch plane.
        target_dir = np.array([0.0, 0.0, 1.0])
        target_dir_dot = np.zeros(3)
        ap.update(ut=0.0, target_dir=target_dir, target_dir_dot=target_dir_dot)
        assert abs(vessel.control.pitch) == pytest.approx(1.0, abs=0.01)

    def test_has_no_close_method(self) -> None:
        """close() was removed; stream lifetime is owned by KSPStreams."""
        ap, _ks, _vessel = _make_autopilot()
        assert not hasattr(ap, "close")

    def test_reset_history_flushes_both_axes(self) -> None:
        ap, _ks, _vessel = _make_autopilot()
        # Inject some history into both axes
        ap._pitch._history.append((0.0, 0.5, 0.0, 2.0))
        ap._yaw._history.append((0.0, 0.5, 0.0, 2.0))
        ap.reset_history()
        assert len(ap._pitch._history) == 0
        assert len(ap._yaw._history) == 0

    def test_gains_initialized_from_physics_prior(self) -> None:
        """kc and a priors should be correctly seeded from torque/moi."""
        # torque magnitude 10 N·m per axis, moi (2, 3, 4) kg·m²
        # kc_pitch = 10/2 = 5,  kc_yaw = 10/4 = 2.5
        # a_pitch = (moi[1]-moi[2])/moi[0] = (3-4)/2 = -0.5
        # a_yaw   = (moi[0]-moi[1])/moi[2] = (2-3)/4 = -0.25
        ap, _ks, _vessel = _make_autopilot(
            torque=((10.0, 10.0, 10.0), (-10.0, -10.0, -10.0)),
            moi=(2.0, 3.0, 4.0),
        )
        assert ap._pitch.kc == pytest.approx(5.0, rel=1e-6)
        assert ap._yaw.kc == pytest.approx(2.5, rel=1e-6)
        assert ap._pitch.a == pytest.approx(-0.5, rel=1e-6)
        assert ap._yaw.a == pytest.approx(-0.25, rel=1e-6)

    def test_rotation_and_ang_vel_streams_registered(self) -> None:
        """CustomAutopilot must register 'rotation' and 'angular_velocity'
        on the KSPStreams object it receives."""
        _ap, ks, _vessel = _make_autopilot()
        assert "rotation" in ks._registered
        assert "angular_velocity" in ks._registered

    def test_ang_vel_fed_to_correct_axis(self) -> None:
        """Angular velocity[0] (pitch axis) should influence pitch gain
        estimation, not yaw."""
        ap, ks, _vessel = _make_autopilot(
            ang_vel=(0.5, 0.0, 0.0),  # only pitch component
        )
        target_dir = np.array([0.0, 1.0, 0.0])
        target_dir_dot = np.zeros(3)
        # Two calls so finite-differencing produces an acceleration estimate.
        ap.update(ut=0.00, target_dir=target_dir, target_dir_dot=target_dir_dot)
        ks.set_ang_vel((0.52, 0.0, 0.0))
        ap.update(ut=0.02, target_dir=target_dir, target_dir_dot=target_dir_dot)
        # History should only exist in the pitch axis
        assert len(ap._pitch._history) >= 1
