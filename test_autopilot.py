"""Unit tests for autopilot.py.

All tests use a fake KSPStreams object — no live KSP/kRPC server needed.
"""

import math
from typing import Any, cast

import numpy as np
import pytest

from autopilot import (
    UNWARPED_PHYSICS_TIMESTEP,
    CustomAutopilot,
    _low_pass_update,
    _omega_n_sat,
    _rate_gain_sat,
    _solve_rotational_torque,
    _torque_to_command,
)
from KSPUtils import KSPStreams

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
    def __init__(self) -> None:
        self.control = FakeControl()

    # These are passed to ks.add_stream() as callables; the FakeKSPStreams
    # doesn't actually call them, so their bodies never run.
    def rotation(self, frame: Any) -> tuple[float, float, float, float]:
        return (0.0, 0.0, 0.0, 1.0)

    def angular_velocity(self, frame: Any) -> tuple[float, float, float]:
        return (0.0, 0.0, 0.0)


def _diag9(x: float, y: float, z: float) -> tuple[float, ...]:
    """Build a flat, row-major 3x3 diagonal inertia tensor."""
    return (x, 0.0, 0.0, 0.0, y, 0.0, 0.0, 0.0, z)


class FakeKSPStreams:
    """Minimal fake KSPStreams for autopilot tests.

    Holds FakeStream objects for ``rotation``, ``angular_velocity``,
    ``available_torque``, and ``inertia_tensor`` that tests can inspect and
    mutate.  Provides the ``__getattr__`` interface that
    ``CustomAutopilot.update()`` uses to read snapshotted values.
    """

    def __init__(
        self,
        rotation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
        ang_vel: tuple[float, float, float] = (0.0, 0.0, 0.0),
        available_torque: tuple[
            tuple[float, float, float], tuple[float, float, float]
        ] = ((10.0, 10.0, 10.0), (-10.0, -10.0, -10.0)),
        inertia_tensor: tuple[float, ...] = _diag9(2.0, 2.0, 2.0),
        ut: float = 0.0,
    ) -> None:
        self._rot_stream = FakeStream(rotation)
        self._ang_vel_stream = FakeStream(ang_vel)
        self._torque_stream = FakeStream(available_torque)
        self._inertia_stream = FakeStream(inertia_tensor)
        # ``ut`` is managed directly by the real KSPStreams (not via
        # add_stream()), so it's a plain attribute here too, settable via
        # set_ut() to simulate the passage of time between update() calls.
        self.ut = ut
        # Map stream name → FakeStream, populated by add_stream().
        self._registered: dict[str, FakeStream] = {}

    def add_stream(self, name: str, func: Any, *args: Any, **kwargs: Any) -> None:
        """Record which streams the autopilot registers."""
        if name == "rotation":
            self._registered[name] = self._rot_stream
        elif name == "angular_velocity":
            self._registered[name] = self._ang_vel_stream
        elif name == "available_torque":
            self._registered[name] = self._torque_stream
        elif name == "inertia_tensor":
            self._registered[name] = self._inertia_stream
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

    def set_available_torque(
        self,
        torque: tuple[tuple[float, float, float], tuple[float, float, float]],
    ) -> None:
        self._torque_stream.set(torque)

    def set_inertia_tensor(self, tensor: tuple[float, ...]) -> None:
        self._inertia_stream.set(tensor)

    def set_ut(self, ut: float) -> None:
        self.ut = ut

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
    inertia_tensor: tuple[float, ...] = _diag9(2.0, 2.0, 2.0),
    rotation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
    ang_vel: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ut: float = 0.0,
    cutoff_freq_hz: float = 1.0,
    sat_angle_deg: float | None = None,
    omega_n_max_normalized: float | None = None,
) -> tuple[CustomAutopilot, FakeKSPStreams, FakeVessel]:
    """Build a CustomAutopilot backed by a FakeKSPStreams.

    ``cutoff_freq_hz`` is converted to the ``ticks_per_filter_cycle``
    parameter now used by ``CustomAutopilot``, preserving the same
    low-pass-filter behavior as before: alpha = 1 - exp(-2*pi*fc*dt)
    == 1 - exp(-2*pi/ticks_per_filter_cycle) when
    ticks_per_filter_cycle == 1 / (fc * dt).

    ``sat_angle_deg`` and ``omega_n_max_normalized`` are passed through
    unchanged (to ``CustomAutopilot``'s own defaults if left ``None``).
    """
    ks = FakeKSPStreams(
        rotation=rotation,
        ang_vel=ang_vel,
        available_torque=torque,
        inertia_tensor=inertia_tensor,
        ut=ut,
    )
    vessel = FakeVessel()
    frame = object()
    ticks_per_filter_cycle = 1.0 / (cutoff_freq_hz * UNWARPED_PHYSICS_TIMESTEP)
    kwargs: dict[str, Any] = {}
    if sat_angle_deg is not None:
        kwargs["sat_angle_deg"] = sat_angle_deg
    if omega_n_max_normalized is not None:
        kwargs["omega_n_max_normalized"] = omega_n_max_normalized
    ap = CustomAutopilot(
        cast(KSPStreams, ks),
        vessel,
        frame,
        ticks_per_filter_cycle=ticks_per_filter_cycle,
        **kwargs,
    )
    return ap, ks, vessel


# ─── Pure function unit tests ─────────────────────────────────────────────────


class TestOmegaNSat:
    def test_known_values(self) -> None:
        # kc = (10+10)/2/2 = 5, sat=45deg=pi/4
        # omega_n_sat = sqrt(5/(pi/4))
        expected = math.sqrt(5.0 / (math.pi / 4))
        result = _omega_n_sat(
            torque_pos=10.0,
            torque_neg=-10.0,
            inertia_diag=2.0,
            sat_angle_rad=math.pi / 4,
            omega_n_max=5.0,
        )
        assert result[0] == pytest.approx(expected, rel=1e-9)
        assert result[1] is False

    def test_capped_at_max(self) -> None:
        result = _omega_n_sat(
            torque_pos=1e6,
            torque_neg=-1e6,
            inertia_diag=1e-3,
            sat_angle_rad=math.radians(45.0),
            omega_n_max=5.0,
        )
        assert result[0] == pytest.approx(5.0)
        assert result[1] is True

    def test_zero_torque_falls_back_to_max(self) -> None:
        result = _omega_n_sat(
            torque_pos=0.0,
            torque_neg=0.0,
            inertia_diag=2.0,
            sat_angle_rad=math.radians(45.0),
            omega_n_max=5.0,
        )
        assert result[0] == pytest.approx(5.0)
        assert result[1] is True


class TestRateGainSat:
    def test_known_values(self) -> None:
        # kc = (10+10)/2/2 = 5, sat_rate = 0.1 rad/s
        # lambda_roll = kc / sat_rate = 50
        result = _rate_gain_sat(
            torque_pos=10.0,
            torque_neg=-10.0,
            inertia_diag=2.0,
            sat_rate_rad_s=0.1,
            gain_max=100.0,
        )
        assert result == pytest.approx(50.0, rel=1e-9)

    def test_capped_at_max(self) -> None:
        result = _rate_gain_sat(
            torque_pos=1e6,
            torque_neg=-1e6,
            inertia_diag=1e-3,
            sat_rate_rad_s=0.1,
            gain_max=5.0,
        )
        assert result == pytest.approx(5.0)

    def test_zero_torque_falls_back_to_max(self) -> None:
        result = _rate_gain_sat(
            torque_pos=0.0,
            torque_neg=0.0,
            inertia_diag=2.0,
            sat_rate_rad_s=0.1,
            gain_max=5.0,
        )
        assert result == pytest.approx(5.0)


class TestLowPassUpdate:
    def test_alpha_one_returns_measured(self) -> None:
        measured = np.array([1.0, 2.0, 3.0])
        prev = np.array([9.0, 9.0, 9.0])
        result = _low_pass_update(measured, prev, alpha=1.0)
        assert result == pytest.approx(measured)

    def test_alpha_zero_returns_previous(self) -> None:
        measured = np.array([1.0, 2.0, 3.0])
        prev = np.array([9.0, 9.0, 9.0])
        result = _low_pass_update(measured, prev, alpha=0.0)
        assert result == pytest.approx(prev)

    def test_alpha_half_averages(self) -> None:
        measured = np.array([2.0, 0.0, 4.0])
        prev = np.array([0.0, 2.0, 0.0])
        result = _low_pass_update(measured, prev, alpha=0.5)
        assert result == pytest.approx([1.0, 1.0, 2.0])


class TestSolveRotationalTorque:
    def test_zero_everything_gives_zero_torque(self) -> None:
        inertia = np.diag([2.0, 3.0, 4.0])
        ang_vel = np.array([0.0, 0.0, 0.0])
        tau = _solve_rotational_torque(inertia, ang_vel, 0.0, 0.0, 0.0)
        assert tau == pytest.approx([0.0, 0.0, 0.0])

    def test_diagonal_tensor_is_decoupled(self) -> None:
        inertia = np.diag([2.0, 3.0, 4.0])
        ang_vel = np.array([0.0, 0.0, 0.0])
        tau = _solve_rotational_torque(inertia, ang_vel, 1.0, 0.0, 0.0)
        # Only the pitch axis should get nonzero torque.
        assert tau[0] == pytest.approx(2.0)
        assert tau[1] == pytest.approx(0.0)
        assert tau[2] == pytest.approx(0.0)

    def test_nondiagonal_tensor_couples_pitch_into_yaw(self) -> None:
        """The key regression test for using the full 3x3 tensor: a pure
        pitch angular-acceleration demand should produce nonzero yaw torque
        when the tensor has pitch/yaw cross-coupling — something a
        diagonal-only (principal-axis) model could never capture."""
        inertia = np.array(
            [
                [2.0, 0.0, 0.5],
                [0.0, 2.0, 0.0],
                [0.5, 0.0, 2.0],
            ]
        )
        ang_vel = np.array([0.0, 0.0, 0.0])
        tau = _solve_rotational_torque(inertia, ang_vel, 1.0, 0.0, 0.0)
        assert tau[0] == pytest.approx(2.0)
        assert tau[1] == pytest.approx(0.0)
        assert tau[2] == pytest.approx(0.5)  # nonzero due to cross-coupling

    def test_gyroscopic_term_with_nonzero_angular_velocity(self) -> None:
        inertia = np.diag([2.0, 3.0, 4.0])
        ang_vel = np.array([1.0, 2.0, 3.0])
        tau = _solve_rotational_torque(inertia, ang_vel, 0.0, 0.0, 0.0)
        # gyro = omega x (I @ omega)
        expected_gyro = np.cross(ang_vel, inertia @ ang_vel)
        assert tau == pytest.approx(expected_gyro)

    def test_round_trip_recovers_requested_alphas(self) -> None:
        inertia = np.array(
            [
                [3.0, 0.2, 0.1],
                [0.2, 2.5, 0.0],
                [0.1, 0.0, 4.0],
            ]
        )
        ang_vel = np.array([0.3, -0.2, 0.5])
        alpha_pitch, alpha_roll, alpha_yaw = 1.2, -0.4, 0.7
        tau = _solve_rotational_torque(
            inertia, ang_vel, alpha_pitch, alpha_roll, alpha_yaw
        )
        gyro = np.cross(ang_vel, inertia @ ang_vel)
        recovered = np.linalg.solve(inertia, tau - gyro)
        assert recovered == pytest.approx([alpha_pitch, alpha_roll, alpha_yaw])


class TestTorqueToCommand:
    def test_zero_torque_gives_zero_command(self) -> None:
        assert _torque_to_command(0.0, 10.0, -10.0) == pytest.approx(0.0)

    def test_sign_is_flipped(self) -> None:
        # Positive torque -> negative command; negative torque -> positive.
        assert _torque_to_command(5.0, 10.0, -10.0) < 0.0
        assert _torque_to_command(-5.0, 10.0, -10.0) > 0.0

    def test_magnitude_uses_correct_direction_limit(self) -> None:
        # tau <= 0 uses torque_neg; tau > 0 uses torque_pos.
        assert _torque_to_command(-4.0, 10.0, -8.0) == pytest.approx(-4.0 / -8.0)
        assert _torque_to_command(4.0, 10.0, -8.0) == pytest.approx(-4.0 / 10.0)

    def test_clips_to_unit_range(self) -> None:
        assert _torque_to_command(1000.0, 10.0, -10.0) == pytest.approx(-1.0)
        assert _torque_to_command(-1000.0, 10.0, -10.0) == pytest.approx(1.0)

    def test_near_zero_available_torque_saturates(self) -> None:
        assert _torque_to_command(5.0, 0.0, -10.0) == pytest.approx(-1.0)
        assert _torque_to_command(-5.0, 10.0, 0.0) == pytest.approx(1.0)
        assert _torque_to_command(0.0, 0.0, 0.0) == pytest.approx(0.0)


# ─── CustomAutopilot integration tests ───────────────────────────────────────


class TestCustomAutopilot:
    def test_zero_error_sets_zero_controls(self) -> None:
        """With vessel already pointing at target and no angular velocity,
        all three controls should be ~0."""
        ap, _ks, vessel = _make_autopilot()
        # Identity rotation: vessel frame == world frame
        # Target is the vessel nose direction in vessel frame: [0, 1, 0] in world
        # With identity quaternion, vessel +y = world +y.
        target_dir = np.array([0.0, 1.0, 0.0])
        target_dir_dot = np.zeros(3)
        ap.update(target_dir=target_dir, target_dir_dot=target_dir_dot)
        assert vessel.control.pitch == pytest.approx(0.0, abs=1e-9)
        assert vessel.control.yaw == pytest.approx(0.0, abs=1e-9)
        assert vessel.control.roll == pytest.approx(0.0, abs=1e-9)

    def test_pitch_error_sets_nonzero_pitch(self) -> None:
        """A pure pitch error should produce a nonzero pitch command."""
        ap, _ks, vessel = _make_autopilot()
        # Rotate target direction slightly in pitch (tilt nose up a bit).
        # In world frame, pitch the target ~5° toward the -z direction
        # (vessel frame +y=forward, -z=up when in identity orientation).
        angle = math.radians(5.0)
        target_dir = np.array([0.0, math.cos(angle), math.sin(angle)])
        target_dir_dot = np.zeros(3)
        ap.update(target_dir=target_dir, target_dir_dot=target_dir_dot)
        # Pitch control should be nonzero
        assert abs(vessel.control.pitch) > 0.0
        assert vessel.control.yaw == pytest.approx(0.0, abs=1e-6)
        assert vessel.control.roll == pytest.approx(0.0, abs=1e-6)

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
        # The (deliberately conservative) class defaults cap omega_n low
        # enough that this synthetic torque/inertia combo wouldn't actually
        # saturate; use an omega_n_max_normalized corresponding to the
        # previous default (omega_n_max=5.0 at dt=0.02) to exercise the
        # saturation path being tested here.
        ap, _ks, vessel = _make_autopilot(omega_n_max_normalized=5.0 * 0.02)
        # Point 90° away from the nose, in the pitch plane.
        target_dir = np.array([0.0, 0.0, 1.0])
        target_dir_dot = np.zeros(3)
        ap.update(target_dir=target_dir, target_dir_dot=target_dir_dot)
        assert abs(vessel.control.pitch) == pytest.approx(1.0, abs=0.01)

    def test_has_no_close_method(self) -> None:
        """CustomAutopilot never owned stream lifetime; still true."""
        ap, _ks, _vessel = _make_autopilot()
        assert not hasattr(ap, "close")

    def test_has_no_reset_history_method(self) -> None:
        """reset_history() was removed entirely: there is no more history
        buffer to flush."""
        ap, _ks, _vessel = _make_autopilot()
        assert not hasattr(ap, "reset_history")

    def test_streams_registered(self) -> None:
        """CustomAutopilot must register 'rotation', 'angular_velocity',
        'available_torque', and 'inertia_tensor' on the KSPStreams object it
        receives."""
        _ap, ks, _vessel = _make_autopilot()
        assert "rotation" in ks._registered
        assert "angular_velocity" in ks._registered
        assert "available_torque" in ks._registered
        assert "inertia_tensor" in ks._registered

    def test_roll_rate_is_actively_damped(self) -> None:
        """A nonzero roll rate (with zero pitch/yaw error) should produce a
        nonzero roll command that opposes it."""
        ap, _ks, vessel = _make_autopilot(ang_vel=(0.0, 0.5, 0.0))
        target_dir = np.array([0.0, 1.0, 0.0])
        target_dir_dot = np.zeros(3)
        ap.update(target_dir=target_dir, target_dir_dot=target_dir_dot)
        assert vessel.control.roll != pytest.approx(0.0, abs=1e-9)

    def test_pitch_rate_decoupled_from_yaw_roll_with_diagonal_tensor(self) -> None:
        """With a diagonal inertia tensor, a pure pitch angular velocity (no
        pointing error) should only affect the pitch command, not yaw/roll."""
        ap, _ks, vessel = _make_autopilot(ang_vel=(0.5, 0.0, 0.0))
        target_dir = np.array([0.0, 1.0, 0.0])
        target_dir_dot = np.zeros(3)
        ap.update(target_dir=target_dir, target_dir_dot=target_dir_dot)
        assert abs(vessel.control.pitch) > 0.0
        assert vessel.control.yaw == pytest.approx(0.0, abs=1e-9)
        assert vessel.control.roll == pytest.approx(0.0, abs=1e-9)

    def test_pitch_rate_couples_into_yaw_with_nondiagonal_tensor(self) -> None:
        """With a non-diagonal (pitch/yaw-coupled) inertia tensor, a pure
        pitch angular velocity should now also produce a nonzero yaw
        command — the integration-level version of
        ``TestSolveRotationalTorque.test_nondiagonal_tensor_couples_pitch_into_yaw``."""
        coupled_tensor = (2.0, 0.0, 0.5, 0.0, 2.0, 0.0, 0.5, 0.0, 2.0)
        ap, _ks, vessel = _make_autopilot(
            inertia_tensor=coupled_tensor, ang_vel=(0.5, 0.0, 0.0)
        )
        target_dir = np.array([0.0, 1.0, 0.0])
        target_dir_dot = np.zeros(3)
        ap.update(target_dir=target_dir, target_dir_dot=target_dir_dot)
        assert vessel.control.yaw != pytest.approx(0.0, abs=1e-9)

    def test_first_call_uses_unfiltered_angular_velocity(self) -> None:
        """With no previous ``ut``, the filter has no history to blend with,
        so the very first update() call must behave exactly as if the raw
        measured angular velocity were used directly (matching pre-filter
        behavior)."""
        ap, _ks, vessel = _make_autopilot(ang_vel=(0.0, 0.5, 0.0), ut=10.0)
        target_dir = np.array([0.0, 1.0, 0.0])
        target_dir_dot = np.zeros(3)
        ap.update(target_dir=target_dir, target_dir_dot=target_dir_dot)
        roll_first_call = vessel.control.roll

        # An effectively-unfiltered autopilot (huge cutoff => tau ~ 0 =>
        # alpha ~ 1 always) should produce the identical result on its
        # first call too.
        ap_unfiltered, _ks2, vessel2 = _make_autopilot(
            ang_vel=(0.0, 0.5, 0.0), ut=10.0, cutoff_freq_hz=1e6
        )
        ap_unfiltered.update(target_dir=target_dir, target_dir_dot=target_dir_dot)
        assert roll_first_call == pytest.approx(vessel2.control.roll)

    def test_second_call_partially_tracks_step_change(self) -> None:
        """A step change in angular velocity between two update() calls
        should be only partially reflected in the second call's command,
        compared to an effectively-unfiltered ('high cutoff') autopilot
        given the exact same inputs."""
        target_dir = np.array([0.0, 1.0, 0.0])
        target_dir_dot = np.zeros(3)

        ap, ks, vessel = _make_autopilot(ang_vel=(0.0, 0.0, 0.0), ut=0.0)
        ap_unfiltered, ks_unfiltered, vessel_unfiltered = _make_autopilot(
            ang_vel=(0.0, 0.0, 0.0), ut=0.0, cutoff_freq_hz=1e6
        )
        ap.update(target_dir=target_dir, target_dir_dot=target_dir_dot)
        ap_unfiltered.update(target_dir=target_dir, target_dir_dot=target_dir_dot)

        # Step change in roll rate, small dt relative to tau (tau = 1/(2*pi) ~ 0.159 s).
        ks.set_ang_vel((0.0, 0.5, 0.0))
        ks.set_ut(0.05)
        ks_unfiltered.set_ang_vel((0.0, 0.5, 0.0))
        ks_unfiltered.set_ut(0.05)

        ap.update(target_dir=target_dir, target_dir_dot=target_dir_dot)
        ap_unfiltered.update(target_dir=target_dir, target_dir_dot=target_dir_dot)

        assert vessel.control.roll != pytest.approx(0.0, abs=1e-9)
        assert vessel_unfiltered.control.roll != pytest.approx(0.0, abs=1e-9)
        # The filtered response should be strictly smaller in magnitude than
        # the unfiltered ('instant step') response.
        assert abs(vessel.control.roll) < abs(vessel_unfiltered.control.roll)
