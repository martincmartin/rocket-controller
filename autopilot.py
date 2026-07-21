#!/usr/bin/env python3
"""Custom PD autopilot for Kerbal Space Program.

Replaces KSP's built-in autopilot with a self-calibrating PD controller.

The controller tracks a target direction (unit vector) and its time derivative,
both expressed in ``body.non_rotating_reference_frame``.  It commands
``vessel.control.pitch`` and ``vessel.control.yaw`` to align the vessel nose
with the target direction.

Physics model
-------------
For each axis (pitch, yaw):

    theta_ddot = -kc * c + k0

where ``c`` ∈ [-1, +1] is the control input, ``kc`` is the (non-negative)
gain magnitude, and ``k0`` is the zero-input offset (e.g. from gimbal
asymmetry).  Both are estimated online from live telemetry.  The minus sign
in front of ``kc`` reflects the empirically-confirmed fact that, in this
vessel-frame axis convention, positive pitch/yaw control *reduces* the
corresponding own-axis angular velocity (see "Axis mapping" below) --
keeping ``kc`` itself non-negative matches its physical origin
(available_torque / moment_of_inertia, which are magnitudes).

Control law (critical damping)
-------------------------------
    c = (2 * omega_n * delta_theta_dot + omega_n**2 * delta_theta + k0) / kc

where ``omega_n`` is chosen so the controller saturates at ``sat_angle_deg``
when the rate error is zero:

    omega_n_sat = sqrt((kc - |k0|) / sat_angle_rad)

subject to a hard cap ``omega_n_max`` (stability against the physics tick).

Axis mapping (confirmed empirically in KSP)
-------------------------------------------
In ``ang_vel_vessel = world_to_vessel.apply(ang_vel_world)``:

    ang_vel_vessel[0]  -- pitch   ('w' key increases it)
    ang_vel_vessel[1]  -- roll    ('q' key increases it)
    ang_vel_vessel[2]  -- yaw     ('a' key increases it)

    Positive pitch reduces (makes negative) x axis angular velocity, increases
    (makes positive) z axis direction error.

    Positive yaw reduces (makes negative) z axis angular velocity, reduces
    (makes negative) x axis direction error.

"""

import math
from collections import deque
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from KSPUtils import KSPStreams

Vector = NDArray[np.float64]


# ─── Single-axis controller ───────────────────────────────────────────────────


class _AxisController:
    """PD controller for one rotational axis.

    Maintains a rolling history of (control_value, angular_acceleration)
    pairs and uses them to estimate the linear model ``theta_ddot = -kc*c + k0``
    online.  From those estimates it computes the critically-damped control
    output each tick.

    Parameters
    ----------
    kc_prior : float
        Initial gain estimate (available_torque / moment_of_inertia for this
        axis).  Used for initialization and as the decay target when the
        control variance is too low for regression.
    dt : float
        Physics tick duration (seconds).
    sat_angle_rad : float
        Angle (radians) at which the controller saturates when rate error is
        zero.  Used to derive ``omega_n``.
    omega_n_max : float
        Hard upper bound on the natural frequency (rad/s).
    history_window_sec : float
        Length of the rolling history window (seconds of game time).
    kc_decay_tau : float
        Time constant (seconds) for decaying ``kc`` toward ``kc_prior`` when
        control variance is low.
    """

    def __init__(
        self,
        kc_prior: float,
        dt: float,
        sat_angle_rad: float,
        omega_n_max: float,
        history_window_sec: float,
        kc_decay_tau: float,
    ) -> None:
        self.kc: float = kc_prior
        self.k0: float = 0.0
        self.kc_prior: float = kc_prior
        self.dt: float = dt
        self.sat_angle_rad: float = sat_angle_rad
        self.omega_n_max: float = omega_n_max
        self.history_window_sec: float = history_window_sec
        self.kc_decay_tau: float = kc_decay_tau

        # Each entry: (game_time, c, theta_ddot)
        self._history: deque[tuple[float, float, float]] = deque()

        # Previous angular velocity for finite-differencing (rad/s)
        self._prev_ang_vel: float | None = None
        # Previous game time for finite-differencing
        self._prev_ut: float | None = None
        # Control value from the previous tick, paired with the theta_ddot
        # measured on the CURRENT tick (the angular acceleration between the
        # previous tick and now was caused by the command applied at the
        # previous tick).
        self._prev_c: float | None = None

    def reset(self) -> None:
        """Flush the history window (call on phase transitions)."""
        self._history.clear()
        self._prev_ang_vel = None
        self._prev_ut = None
        self._prev_c = None

    def _omega_n(self) -> float:
        """Compute the natural frequency from the current gain estimates."""
        effective = self.kc - abs(self.k0)
        if effective <= 0 or self.sat_angle_rad <= 0:
            return self.omega_n_max
        omega_sat = math.sqrt(effective / self.sat_angle_rad)
        return min(self.omega_n_max, omega_sat)

    def _update_gains(self, ut: float, c: float, ang_vel: float, verbose: bool) -> None:
        """Update gain estimates from the latest measurement.

        Estimates angular acceleration by finite-differencing ``ang_vel``,
        appends (c, theta_ddot) to the rolling window, prunes old entries,
        then refits the model.
        """
        if (
            self._prev_ang_vel is not None
            and self._prev_ut is not None
            and self._prev_c is not None
        ):
            elapsed = ut - self._prev_ut
            if elapsed > 0:
                theta_ddot = (ang_vel - self._prev_ang_vel) / elapsed
                # Pair theta_ddot with the command from the *previous* tick:
                # the acceleration between the previous tick and now was caused
                # by the command that was applied at the previous tick.
                self._history.append((ut, self._prev_c, theta_ddot))

        self._prev_ang_vel = ang_vel
        self._prev_ut = ut
        self._prev_c = c

        # Prune entries older than the window.
        cutoff = ut - self.history_window_sec
        while self._history and self._history[0][0] < cutoff:
            self._history.popleft()

        if len(self._history) < 2:
            return

        _times, controls, accels = zip(*self._history, strict=False)
        c_arr = np.array(controls, dtype=np.float64)
        a_arr = np.array(accels, dtype=np.float64)

        c_var = float(c_arr.max() - c_arr.min())
        if c_var > 0.1:
            # Enough variance: fit kc and k0 by least-squares.
            # Fit theta_ddot = -kc*c + k0, i.e. regress against -c so that
            # the fitted slope comes out directly as kc (non-negative).
            A = np.column_stack([-c_arr, np.ones_like(c_arr)])
            result, _, _, _ = np.linalg.lstsq(A, a_arr, rcond=None)
            kc_fit, k0_fit = float(result[0]), float(result[1])
            if kc_fit > 0:  # sanity check: gain magnitude must be positive
                self.kc = kc_fit
                self.k0 = k0_fit
        else:
            # Low variance: decay kc toward the prior first, then update k0
            # using the already-adjusted kc so they stay consistent.
            c_mean = float(c_arr.mean())
            a_mean = float(a_arr.mean())
            self.kc += (self.dt / self.kc_decay_tau) * (self.kc_prior - self.kc)
            self.k0 = a_mean + self.kc * c_mean

    def compute(
        self,
        ut: float,
        delta_theta: float,
        delta_theta_dot: float,
        ang_vel: float,
        last_c: float,
        verbose: bool,
    ) -> float:
        """Update gain estimates and return the control output for this tick.

        Parameters
        ----------
        ut : float
            Current universal time (seconds).
        delta_theta : float
            Angle error (radians).  Positive means the vessel needs to rotate
            in the positive direction for this axis.
        delta_theta_dot : float
            Angular velocity error (rad/s).
        ang_vel : float
            Current angular velocity for this axis (rad/s), used for gain
            estimation.
        last_c : float
            Control value commanded on the previous tick (used in the history).

        Returns
        -------
        float
            Control output, clamped to [-1, +1].
        """
        self._update_gains(ut, last_c, ang_vel, verbose)

        omega_n = self._omega_n()
        theta_ddot_desired = -2.0 * omega_n * delta_theta_dot - omega_n**2 * delta_theta

        if abs(self.kc) < 1e-9:
            return 0.0

        c = (self.k0 - theta_ddot_desired) / self.kc
        return float(np.clip(c, -1.0, 1.0))


# ─── Public autopilot class ───────────────────────────────────────────────────


class CustomAutopilot:
    """Self-calibrating PD autopilot for the coast and circularization phases.

    Registers kRPC streams for vessel orientation and angular velocity on the
    provided ``KSPStreams`` object.  The caller is responsible for calling
    ``ks.start()`` and ``ks.next()`` — this class never calls either.  Call
    ``update()`` once per physics tick (after ``ks.next()``) to compute and
    apply pitch/yaw control commands.

    ``available_torque`` and ``moment_of_inertia`` are read once at
    construction time via plain kRPC property accesses to bootstrap the
    controller gain priors; they are not streamed.

    Parameters
    ----------
    ks : KSPStreams
        Shared stream manager.  ``rotation`` and ``angular_velocity`` streams
        are registered on it by this constructor.
    vessel : Any
        kRPC vessel object.
    frame : Any
        Reference frame for direction inputs
        (``body.non_rotating_reference_frame``).
    dt : float
        Physics tick duration.  KSP default is 0.02 s.
    sat_angle_deg : float
        Angle (degrees) at which the controller saturates (``c = ±1``) when
        the rate error is zero.
    omega_n_max : float
        Hard upper bound on the natural frequency (rad/s).  Must be less than
        ``1 / (10 * dt)`` for numerical stability.
    history_window_sec : float
        Length of the rolling gain-estimation window (seconds of game time).
    kc_decay_tau : float
        Time constant (seconds) for decaying ``kc`` toward the physics prior.
    """

    def __init__(
        self,
        ks: KSPStreams,
        vessel: Any,
        frame: Any,
        dt: float = 0.02,
        sat_angle_deg: float = 45.0,
        omega_n_max: float = 5.0,
        history_window_sec: float = 1.0,
        kc_decay_tau: float = 5.0,
    ) -> None:
        self._ks = ks
        self._vessel = vessel
        self._dt = dt

        # ── Register streams on the shared KSPStreams object ───────────────
        ks.add_stream("rotation", vessel.rotation, frame)
        ks.add_stream("angular_velocity", vessel.angular_velocity, frame)

        # ── Bootstrap gains from physics ──────────────────────────────────
        # available_torque and moment_of_inertia are only needed once here
        # to seed the gain priors, so they are plain kRPC property accesses
        # rather than streams.
        #
        # available_torque returns ((pos_x, pos_y, pos_z), (neg_x, neg_y, neg_z))
        # moment_of_inertia returns (x, y, z) in kg·m²
        torque_pair = vessel.available_torque
        moi = vessel.moment_of_inertia
        torque_pos, torque_neg = torque_pair
        # Pitch axis: index 0 in ang_vel_vessel → moment_of_inertia[0]
        # Yaw axis:   index 2 in ang_vel_vessel → moment_of_inertia[2]
        kc_pitch_prior = (
            (abs(torque_pos[0]) + abs(torque_neg[0])) / 2.0 / max(moi[0], 1e-9)
        )
        kc_yaw_prior = (
            (abs(torque_pos[2]) + abs(torque_neg[2])) / 2.0 / max(moi[2], 1e-9)
        )

        sat_angle_rad = math.radians(sat_angle_deg)
        axis_kwargs = {
            "dt": dt,
            "sat_angle_rad": sat_angle_rad,
            "omega_n_max": omega_n_max,
            "history_window_sec": history_window_sec,
            "kc_decay_tau": kc_decay_tau,
        }
        self._pitch = _AxisController(kc_prior=kc_pitch_prior, **axis_kwargs)
        self._yaw = _AxisController(kc_prior=kc_yaw_prior, **axis_kwargs)

        # Last commanded control values (used as the "previous" input when
        # estimating gains on the next tick).
        self._last_pitch_c: float = 0.0
        self._last_yaw_c: float = 0.0

    def reset_history(self) -> None:
        """Flush the gain-estimation history window.

        Call this at phase transitions (e.g. switching from coast pointing to
        the burn) so stale data from the previous phase does not corrupt the
        new gain estimates.
        """
        self._pitch.reset()
        self._yaw.reset()

    def update(
        self,
        ut: float,
        target_dir: Vector,
        target_dir_dot: Vector,
    ) -> None:
        """Compute and apply pitch/yaw controls for one physics tick.

        Must be called *after* the caller has called ``ks.next()`` so that
        ``ks.rotation`` and ``ks.angular_velocity`` reflect the current tick.

        Parameters
        ----------
        ut : float
            Current universal time (seconds), used for gain estimation timing.
        target_dir : ndarray, shape (3,)
            Desired nose direction (unit vector) in ``frame``.
        target_dir_dot : ndarray, shape (3,)
            Time derivative of ``target_dir`` in ``frame`` (rad/s, tangent to
            the unit sphere).
        """
        # ── Read from the shared KSPStreams snapshot ───────────────────────
        # These values were captured atomically by the caller's ks.next() call,
        # so rotation and angular_velocity are guaranteed to be from the same
        # physics tick — essential for accurate finite-difference estimates.
        quat = self._ks.rotation  # (x, y, z, w) — scipy order
        ang_vel_world = np.array(self._ks.angular_velocity)

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

        # ── Compute control outputs ───────────────────────────────────────
        pitch_c = self._pitch.compute(
            ut=ut,
            delta_theta=delta_theta_pitch,
            delta_theta_dot=delta_theta_dot_pitch,
            ang_vel=ang_vel_vessel[0],
            last_c=self._last_pitch_c,
            verbose=True,
        )
        yaw_c = self._yaw.compute(
            ut=ut,
            delta_theta=delta_theta_yaw,
            delta_theta_dot=delta_theta_dot_yaw,
            ang_vel=ang_vel_vessel[2],
            last_c=self._last_yaw_c,
            verbose=False,
        )

        # ── Apply commands ───────────────────────────────────────────────
        self._vessel.control.pitch = pitch_c
        self._vessel.control.yaw = yaw_c

        self._last_pitch_c = pitch_c
        self._last_yaw_c = yaw_c
