#!/usr/bin/env python3
"""Custom PD autopilot for Kerbal Space Program.

Replaces KSP's built-in autopilot with a physics-based critically-damped
controller.

The controller tracks a target direction (unit vector) and its time
derivative, both expressed in ``body.non_rotating_reference_frame``.  It
commands ``vessel.control.pitch``, ``vessel.control.roll`` and
``vessel.control.yaw`` to align the vessel nose with the target direction
while holding roll angular velocity at zero.

Physics model
-------------
Rather than fitting a linear model online, the controller uses the exact 3D
rigid-body (Euler) rotation equation, evaluated fresh every tick from live
kRPC streams:

    I @ omega_dot = tau - omega x (I @ omega)

where ``I`` is the vessel's full 3x3 inertia tensor (``inertia_tensor``
stream — *not* just the diagonal ``moment_of_inertia``, so any cross-axis
coupling from an asymmetric mass distribution is captured exactly, not just
the gyroscopic ``omega x (I @ omega)`` term), ``omega`` is the vessel-frame
angular velocity, and ``tau`` is the applied torque.

``omega`` is derived from the raw ``angular_velocity`` stream by a simple
exponential low-pass filter, to reduce oscillation from noisy/high-frequency
rate measurements:

    omega_filtered(t) = alpha * omega_measured(t) + (1 - alpha) * omega_filtered(t - dt)

The filter is applied to ``angular_velocity`` while it is still expressed in
``frame`` (the non-rotating reference frame passed to the constructor) —
*before* rotating it into the vessel body frame. ``frame`` doesn't rotate
tick to tick, so averaging consecutive samples expressed in it is
well-defined; the vessel body frame does rotate every tick (as the vessel
turns), so filtering post-rotation would incorrectly mix vectors expressed
in different frames.

Given desired angular accelerations for all three axes
(``alpha_pitch``, ``alpha_roll``, ``alpha_yaw``), the required torque is
simply:

    tau = I @ [alpha_pitch, alpha_roll, alpha_yaw] + omega x (I @ omega)

No matrix inversion, history buffer, or online regression is needed — this
is an exact, direct solve each tick.

Control law (critical damping for pitch/yaw, rate damping for roll)
--------------------------------------------------------------------
Pitch and yaw track a target direction with the usual critically-damped
second-order law:

    alpha = -2 * omega_n * delta_theta_dot - omega_n**2 * delta_theta

``omega_n`` is chosen so the controller saturates at ``sat_angle_deg`` when
the rate error is zero:

    omega_n_sat = sqrt(kc / sat_angle_rad)

where ``kc = avg(|available_torque|) / inertia_diag`` for that axis, read
fresh from the live ``available_torque``/``inertia_tensor`` streams each
tick (no more online-fitted ``kc``, and no more ``k0`` bias term — the exact
torque solve above already accounts for the cross-coupling/bias that ``k0``
used to approximate).

Roll has no target *angle* (we don't care which way the vessel is rolled),
only a target *rate* of zero, so it uses a first-order rate-damping law
instead of the second-order critically-damped one:

    alpha_roll = -lambda_roll * ang_vel_roll

``lambda_roll`` is sized the same way as ``omega_n``, but without the square
root (this is the correct saturation condition for a first-order system):

    lambda_roll = kc_roll / sat_roll_rate_rad_s

Torque -> command conversion
-----------------------------
``available_torque`` gives, per axis, ``(torque_pos, torque_neg)`` with
``torque_pos >= 0`` the max achievable torque in the positive-axis direction
and ``torque_neg <= 0`` the max achievable torque in the negative-axis
direction.

Empirically, increasing a control command *decreases* the corresponding
own-axis torque (see "Axis mapping" below) — command and torque have
*opposite* signs.  So, given a desired torque ``tau``, the corresponding
command is:

    tau <= 0  =>  c =  tau / torque_neg      (both <= 0, so c >= 0)
    tau >  0  =>  c = -tau / torque_pos      (c < 0)

clipped to ``[-1, 1]``.

Axis mapping (confirmed empirically in KSP)
-------------------------------------------
In ``ang_vel_vessel = world_to_vessel.apply(ang_vel_world)``:

    ang_vel_vessel[0]  -- pitch   ('w' key increases it)
    ang_vel_vessel[1]  -- roll    ('q' key increases it)
    ang_vel_vessel[2]  -- yaw     ('a' key increases it)

    Positive pitch control reduces (makes negative) x axis angular velocity,
    increases (makes positive) z axis direction error.

    Positive yaw control reduces (makes negative) z axis angular velocity,
    reduces (makes negative) x axis direction error.

    Roll is assumed (by symmetry with pitch/yaw, but *not yet empirically
    confirmed*) to follow the same pattern: positive roll control reduces
    (makes negative) y axis angular velocity. If this turns out to be
    backwards in practice, only the sign convention applied to the roll axis
    in ``_torque_to_command`` needs to change.

"""

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from KSPUtils import KSPStreams

Vector = NDArray[np.float64]


UNWARPED_PHYSICS_TIMESTEP = 0.02  # seconds

# ─── Pure physics/gain helper functions ───────────────────────────────────────


def _omega_n_sat(
    torque_pos: float,
    torque_neg: float,
    inertia_diag: float,
    sat_angle_rad: float,
    omega_n_max: float,
) -> tuple[float, bool]:
    """Natural frequency for a critically-damped second-order axis (pitch/yaw).

    Chosen so the controller saturates (``c = ±1``) at ``sat_angle_rad`` when
    the rate error is zero, given the current available torque and own-axis
    inertia.  Clamped to ``omega_n_max`` for stability against the physics
    tick.
    """
    kc = (abs(torque_pos) + abs(torque_neg)) / 2.0 / max(inertia_diag, 1e-9)
    if kc <= 0 or sat_angle_rad <= 0:
        return omega_n_max, True

    omega_n = math.sqrt(kc / sat_angle_rad)
    if omega_n < omega_n_max:
        return (omega_n, False)
    return (omega_n_max, True)


def _rate_gain_sat(
    torque_pos: float,
    torque_neg: float,
    inertia_diag: float,
    sat_rate_rad_s: float,
    gain_max: float,
) -> float:
    """Rate-feedback gain for a first-order axis (roll).

    Chosen so the controller saturates (``c = ±1``) at ``sat_rate_rad_s`` of
    angular velocity error, given the current available torque and own-axis
    inertia.  Unlike ``_omega_n_sat`` this is a linear (no square root)
    relationship, since roll damping is a first-order system (rate only, no
    angle target).  Clamped to ``gain_max``.
    """
    kc = (abs(torque_pos) + abs(torque_neg)) / 2.0 / max(inertia_diag, 1e-9)
    if kc <= 0 or sat_rate_rad_s <= 0:
        return gain_max
    return min(gain_max, kc / sat_rate_rad_s)


def _solve_rotational_torque(
    inertia_tensor: Vector,
    ang_vel: Vector,
    alpha_pitch: float,
    alpha_roll: float,
    alpha_yaw: float,
) -> Vector:
    """Solve the exact 3D Euler rigid-body equation for the required torque.

    Given the current (possibly non-diagonal) inertia tensor and angular
    velocity, and the desired angular accelerations for all three axes,
    returns the torque vector ``(tau_pitch, tau_roll, tau_yaw)`` that would
    produce them:

        tau = I @ omega_dot + omega x (I @ omega)

    This is a direct, exact solve (no matrix inversion, no approximation of
    the gyroscopic/cross-coupling term).
    """
    omega_dot = np.array([alpha_pitch, alpha_roll, alpha_yaw], dtype=np.float64)
    gyro = np.cross(ang_vel, inertia_tensor @ ang_vel)
    tau: Vector = inertia_tensor @ omega_dot + gyro
    return tau


def _low_pass_update(measured: Vector, filtered_prev: Vector, alpha: float) -> Vector:
    """One step of the exponential low-pass filter.

    filtered(t) = alpha * measured(t) + (1 - alpha) * filtered(t - dt)
    """
    result: Vector = alpha * measured + (1.0 - alpha) * filtered_prev
    return result


def _torque_to_command(tau: float, torque_pos: float, torque_neg: float) -> float:
    """Convert a desired own-axis torque (N·m) to a normalized ``[-1, 1]`` command.

    Increasing a control command *decreases* the corresponding own-axis
    torque (empirically confirmed for pitch/yaw; assumed by symmetry for
    roll), so command and torque have opposite signs:

        tau <= 0  =>  c =  tau / torque_neg      (torque_neg <= 0, so c >= 0)
        tau >  0  =>  c = -tau / torque_pos      (c < 0)

    Guards the near-zero-available-torque case by saturating toward
    ``sign(-tau)`` instead of dividing by ~0.
    """
    eps = 1e-6
    if tau <= 0:
        if abs(torque_neg) < eps:
            return 0.0 if tau == 0 else 1.0
        c = tau / torque_neg
    else:
        if abs(torque_pos) < eps:
            return -1.0
        c = -tau / torque_pos
    return float(np.clip(c, -1.0, 1.0))


# ─── Public autopilot class ───────────────────────────────────────────────────


class CustomAutopilot:
    """Physics-based critically-damped autopilot for the coast and
    circularization phases.

    Registers kRPC streams for vessel orientation, angular velocity,
    available torque, and inertia tensor on the provided ``KSPStreams``
    object.  The caller is responsible for calling ``ks.start()`` and
    ``ks.next()`` — this class never calls either.  Call ``update()`` once
    per physics tick (after ``ks.next()``) to compute and apply
    pitch/roll/yaw control commands.

    Parameters
    ----------
    ks : KSPStreams
        Shared stream manager.  ``rotation``, ``angular_velocity``,
        ``available_torque``, and ``inertia_tensor`` streams are registered
        on it by this constructor.
    vessel : Any
        kRPC vessel object.
    frame : Any
        Reference frame for direction inputs
        (``body.non_rotating_reference_frame``).
    sat_angle_deg : float
        Angle (degrees) at which the pitch/yaw controllers saturate
        (``c = ±1``) when the rate error is zero.
    sat_roll_rate_deg_s : float
        Roll angular velocity (degrees/second) at which the roll controller
        saturates (``c = ±1``).
    omega_n_max : float
        Hard upper bound on the natural frequency (rad/s) for pitch/yaw, and
        on the rate-feedback gain (1/s) for roll.  Must be less than
        ``1 / (10 * dt)`` for numerical stability given the physics tick.
    cutoff_freq_hz : float
        Cutoff frequency (Hz) of the exponential low-pass filter applied to
        the raw ``angular_velocity`` stream before it's used by the control
        law, to reduce oscillation.
    """

    def __init__(
        self,
        ks: KSPStreams,
        vessel: Any,
        frame: Any,
        sat_angle_deg: float = 5.0,
        sat_roll_rate_deg_s: float = 5.0,
        omega_n_max: float = 5.0,
        cutoff_freq_hz: float = 3.0,
    ) -> None:
        self._ks = ks
        self._vessel = vessel

        # ── Register streams on the shared KSPStreams object ───────────────
        ks.add_stream("rotation", vessel.rotation, frame)
        ks.add_stream("angular_velocity", vessel.angular_velocity, frame)
        # available_torque returns ((pos_x, pos_y, pos_z), (neg_x, neg_y, neg_z))
        # and can change over the flight (gimbal lock/unlock, reaction wheels
        # destroyed, etc.), so it must be a live stream, not a one-shot read.
        ks.add_stream("available_torque", getattr, vessel, "available_torque")
        # inertia_tensor is the full 3x3 tensor (flat, row-major, 9 floats),
        # in the vessel's reference frame.  Changes continuously as
        # propellant burns, so it must also be a live stream.
        ks.add_stream("inertia_tensor", getattr, vessel, "inertia_tensor")

        self._sat_angle_rad = math.radians(sat_angle_deg)
        self._sat_roll_rate_rad_s = math.radians(sat_roll_rate_deg_s)
        self._omega_n_max = omega_n_max
        self.pitch_max = False
        self.yaw_max = False

        # ── Angular-velocity low-pass filter state ─────────────────────────
        # tau_filter = 1 / (2*pi*fc); alpha computed fresh each tick from the
        # actual dt (self._ks.ut - self._last_ut), since KSP physics ticks
        # aren't perfectly uniform.
        self._alpha_filter = 1.0 - math.exp(
            -2.0 * math.pi * cutoff_freq_hz * UNWARPED_PHYSICS_TIMESTEP
        )
        self._filtered_ang_vel_world: Vector | None = None
        self._last_ut: float | None = None

        print(f"cutoff freq: {cutoff_freq_hz}, alpha: {self._alpha_filter}")

    def update(
        self,
        target_dir: Vector,
        target_dir_dot: Vector,
    ) -> None:
        """Compute and apply pitch/roll/yaw controls for one physics tick.

        Must be called *after* the caller has called ``ks.next()`` so that
        ``ks.ut``, ``ks.rotation``, ``ks.angular_velocity``,
        ``ks.available_torque``, and ``ks.inertia_tensor`` reflect the
        current tick.

        Parameters
        ----------
        target_dir : ndarray, shape (3,)
            Desired nose direction (unit vector) in ``frame``.
        target_dir_dot : ndarray, shape (3,)
            Time derivative of ``target_dir`` in ``frame`` (rad/s, tangent to
            the unit sphere).
        """
        # ── Read from the shared KSPStreams snapshot ───────────────────────
        # These values were captured atomically by the caller's ks.next() call,
        # so all streams are guaranteed to be from the same physics tick.
        ut = self._ks.ut
        quat = self._ks.rotation  # (x, y, z, w) — scipy order
        ang_vel_world_measured = np.array(self._ks.angular_velocity)
        torque_pos, torque_neg = self._ks.available_torque
        inertia_tensor = np.array(self._ks.inertia_tensor, dtype=np.float64).reshape(
            3, 3
        )

        # ── Low-pass filter the angular velocity in the (non-rotating)
        # world frame, before rotating it into the (rotating) vessel frame.
        # See module docstring for why the filter must be applied here.
        if self._last_ut is None or self._filtered_ang_vel_world is None:
            ang_vel_world = ang_vel_world_measured
        else:
            ang_vel_world = _low_pass_update(
                ang_vel_world_measured, self._filtered_ang_vel_world, self._alpha_filter
            )
        self._filtered_ang_vel_world = ang_vel_world
        self._last_ut = ut

        world_to_vessel = Rotation.from_quat(quat).inv()

        # Angular velocity in vessel frame.
        # Axis mapping (empirically confirmed in KSP):
        #   [0] pitch  ('w' key increases it)
        #   [1] roll   ('q' key increases it)
        #   [2] yaw    ('a' key increases it)
        ang_vel_vessel = world_to_vessel.apply(ang_vel_world)

        # ── Target direction in vessel frame ─────────────────────────────
        target_vessel = world_to_vessel.apply(target_dir)
        target_dot_vessel = world_to_vessel.apply(target_dir_dot)

        # Pitch: Positive pitch reduces (makes negative) x axis angular
        # velocity, increases (makes positive) z axis direction error.
        delta_theta_pitch = -target_vessel[2]
        target_pitch_dot = -target_dot_vessel[2]
        delta_theta_dot_pitch = target_pitch_dot + ang_vel_vessel[0]

        # Yaw: Positive yaw reduces (makes negative) z axis angular velocity,
        # reduces (makes negative) x axis direction error.
        delta_theta_yaw = target_vessel[0]
        target_yaw_dot = target_dot_vessel[0]
        delta_theta_dot_yaw = target_yaw_dot + ang_vel_vessel[2]

        # ── Natural-frequency / rate-gain sizing from live torque/inertia ──
        omega_n_pitch, self.pitch_max = _omega_n_sat(
            torque_pos[0],
            torque_neg[0],
            inertia_tensor[0, 0],
            self._sat_angle_rad,
            self._omega_n_max,
        )
        omega_n_yaw, self.yaw_max = _omega_n_sat(
            torque_pos[2],
            torque_neg[2],
            inertia_tensor[2, 2],
            self._sat_angle_rad,
            self._omega_n_max,
        )
        lambda_roll = _rate_gain_sat(
            torque_pos[1],
            torque_neg[1],
            inertia_tensor[1, 1],
            self._sat_roll_rate_rad_s,
            self._omega_n_max,
        )

        # ── Desired angular accelerations ──────────────────────────────────
        alpha_pitch = (
            -2.0 * omega_n_pitch * delta_theta_dot_pitch
            - omega_n_pitch**2 * delta_theta_pitch
        )
        alpha_yaw = (
            -2.0 * omega_n_yaw * delta_theta_dot_yaw - omega_n_yaw**2 * delta_theta_yaw
        )
        # Roll has no angle target, only a rate target of zero.
        alpha_roll = -lambda_roll * ang_vel_vessel[1]

        # ── Exact 3D rigid-body torque solve ───────────────────────────────
        tau_pitch, tau_roll, tau_yaw = _solve_rotational_torque(
            inertia_tensor, ang_vel_vessel, alpha_pitch, alpha_roll, alpha_yaw
        )

        # ── Convert torque to normalized commands ──────────────────────────
        pitch_c = _torque_to_command(tau_pitch, torque_pos[0], torque_neg[0])
        roll_c = _torque_to_command(tau_roll, torque_pos[1], torque_neg[1])
        yaw_c = _torque_to_command(tau_yaw, torque_pos[2], torque_neg[2])

        # ── Apply commands ───────────────────────────────────────────────
        self._vessel.control.pitch = pitch_c
        self._vessel.control.roll = roll_c
        self._vessel.control.yaw = yaw_c
