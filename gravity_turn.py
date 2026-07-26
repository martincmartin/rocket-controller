#!/usr/bin/env python3

import enum
import math
import time
from collections.abc import Callable
from typing import Any

import krpc
import krpc.client
import krpc.services.spacecenter
import numpy as np
from krpc.services.spacecenter import Vessel

from autopilot_thread import AutopilotWorker
from guidance_link import GuidanceCommand
from KSPUtils import KSPStreams, build_segments
from sim import Simulator


def engine_repr(engine: Any) -> str:
    return engine.part.title  # type: ignore[no-any-return]


krpc.services.spacecenter.Engine.__repr__ = engine_repr  # type: ignore[method-assign]


class FlightSession:
    POLL_INTERVAL = 0.25

    def __init__(self, conn: krpc.client.Client) -> None:
        self.conn = conn
        self.streams: KSPStreams  # set in __enter__
        self._ready = False  # True only strictly inside the `with` block
        self._vessel: Vessel | None = None

    @property
    def space_center(self) -> Any:
        # kRPC's Client.space_center is only Optional in its type stubs
        # because of a try/except ImportError fallback; it is always
        # populated on a real connection.
        assert self.conn.space_center is not None
        return self.conn.space_center

    def __enter__(self) -> "FlightSession":
        self._vessel = self._wait_for_vessel(self.conn)
        self._vessel.control.sas = False
        self._vessel.control.rcs = False

        if self._vessel.situation == self.space_center.VesselSituation.pre_launch:
            self._vessel.auto_pilot.stopping_time = (0.5, 0.5, 0.5)
            self._vessel.auto_pilot.reference_frame = (
                self._vessel.surface_reference_frame
            )
            self._vessel.auto_pilot.target_pitch_and_heading(90, 90)
            self._vessel.auto_pilot.target_roll = 90

            self.space_center.physics_warp_factor = 0

        self.streams = KSPStreams(self.conn)
        self._ready = True
        return self

    @property
    def vessel(self) -> Vessel:
        if not self._ready or self._vessel is None:
            raise RuntimeError(
                "FlightSession.vessel used outside an active session "
                "(before __enter__ or after __exit__)"
            )
        return self._vessel

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: Any,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Best-effort teardown: every step is attempted even if an
        earlier one raised, so one failure (e.g. the vessel already being
        gone) can't prevent the rest of cleanup."""
        if not self._ready:
            return

        vessel = self._vessel
        if vessel is not None:
            self._try(lambda: setattr(vessel.control, "throttle", 0.0))
            self._try(lambda: setattr(vessel.control, "sas", False))
            self._try(lambda: setattr(vessel.control, "rcs", False))
            self._try(lambda: setattr(vessel.auto_pilot, "target_roll", 90.0))
            self._try(lambda: vessel.auto_pilot.target_pitch_and_heading(90, 90))
            self._try(vessel.auto_pilot.disengage)
        self._try(lambda: setattr(self.space_center, "physics_warp_factor", 0))

        self._try(self.streams.close)

        self._vessel = None
        self._ready = False

    @staticmethod
    def _try(action: Callable[[], Any]) -> None:
        try:
            action()
        except Exception as e:
            print(f"  ! FlightSession cleanup warning: {e}")

    @classmethod
    def _wait_for_vessel(cls, conn: Any) -> Vessel:
        flight_scene = conn.krpc.GameScene.flight
        while True:
            try:
                # current_game_scene is a coarse, cheap guard against reading a
                # stale/leftover active_vessel while the scene itself is still
                # transitioning (e.g. mid-load()).
                if conn.krpc.current_game_scene == flight_scene:
                    return conn.space_center.active_vessel  # type: ignore[no-any-return]
            except Exception:
                pass  # scene mid-transition
            time.sleep(cls.POLL_INTERVAL)


class Phase(enum.IntEnum):
    INIT = enum.auto()
    PRELAUNCH = enum.auto()  # on the pad, engines not yet lit
    ASCENT = enum.auto()  # gravity turn, apoapsis climbing to target
    ATMOSPHERE_COAST = enum.auto()  # engines cut, waiting to clear atmosphere
    CIRCULARIZE = enum.auto()  # coast-to-burn-start *and* the burn itself
    DONE = enum.auto()  # eccentricity already within tolerance


class GravityTurn:
    TARGET_ALTITUDE = 80_000  # Desired circular orbit altitude (m)
    ENGINE_CUTOFF_ALTITUDE = 60_000  # Once apoapsis reaches this, cut engines.
    HEADING = 90  # Launch azimuth (90 = due east for equatorial orbit)
    AP_WARP_MARGIN = 0.90  # Turn off warp when Ap > this x target
    AP_THROTTLE_MARGIN = 0.95  # Start tapering throttle when Ap > this × target
    ATMOSPHERE_ALTITUDE = 25_000  # Kerbin atmosphere is 0.01 atm at 25k, 0.001 at 40k.
    ECC_TOLERANCE = 0.1  # "circular enough" eccentricity to stop the burn
    BURN_TIME_SAFETY_MARGIN = 1.5  # abort if we run this much past the plan

    STAGING_DURATION = 2.5  # seconds; measured real-world KSP staging delay

    def __init__(
        self,
        fs: FlightSession,
        turn_start_alt: float = 100,
        turn_end_alt: float = 30_000,
    ) -> None:
        self.fs = fs

        self.phase: Phase = Phase.INIT

        body = self.fs.vessel.orbit.body
        self.mu = body.gravitational_parameter
        self.body_radius = body.equatorial_radius
        self.frame = body.non_rotating_reference_frame

        # Ascent
        self.turn_start_alt = turn_start_alt
        self.turn_end_alt = turn_end_alt
        self.prev_pitch: float = 0.0  # last commanded ascent pitch

        # Circularization; created on demand in circularize(), stopped
        # when CIRCULARIZE is left (success or otherwise) -- see do_it().
        self.worker: AutopilotWorker | None = None

    def do_it(self) -> None:
        # Reference body parameters (Kerbin)
        vessel = self.fs.vessel

        # Start all streams and block until each has its first value.
        streams = self.fs.streams
        streams.add_stream("altitude", getattr, vessel.flight(), "mean_altitude")
        streams.add_stream("apoapsis", getattr, vessel.orbit, "apoapsis_altitude")
        streams.add_stream("periapsis", getattr, vessel.orbit, "periapsis_altitude")
        streams.add_stream("thrust", getattr, vessel, "thrust")
        streams.add_stream("available_thrust", getattr, vessel, "available_thrust")
        streams.add_stream("position", vessel.position, self.frame)
        streams.add_stream("velocity", vessel.velocity, self.frame)
        streams.add_stream(
            "time_to_apoapsis", getattr, vessel.orbit, "time_to_apoapsis"
        )
        streams.add_stream("direction", getattr, vessel.flight(self.frame), "direction")
        streams.add_stream("current_stage", getattr, vessel.control, "current_stage")
        streams.start()

        pre_launch_situation = self.fs.space_center.VesselSituation.pre_launch

        try:
            while True:
                streams.next()

                if vessel.situation == pre_launch_situation:
                    self.prelaunch()
                elif streams.apoapsis < self.ENGINE_CUTOFF_ALTITUDE:
                    self.ascent()
                elif streams.altitude < self.ATMOSPHERE_ALTITUDE:
                    self.atmosphere_coast()
                elif self.eccentricity_decreasing():
                    self.circularize()
                    if self.worker is not None and self.worker.error is not None:
                        print(f"\nAutopilot worker failed: {self.worker.error}")
                        vessel.control.throttle = 0.0
                        return
                else:
                    print("\nDone.")
                    vessel.control.sas = True
                    return
        finally:
            # Ensure the worker thread/connection are always torn down, even
            # on an early return or an exception unwinding out of do_it().
            if self.worker is not None:
                self.worker.stop()
                self.worker = None

    def prelaunch(self) -> None:
        if self.fs.streams.available_thrust == 0:
            vessel = self.fs.vessel
            vessel.control.sas = False
            vessel.control.rcs = False
            vessel.control.throttle = 1.0
            vessel.control.activate_next_stage()
            vessel.auto_pilot.engage()
            vessel.auto_pilot.target_pitch_and_heading(90, self.HEADING)
            vessel.auto_pilot.target_roll = 90

    def stage_if_needed(self) -> bool:
        streams = self.fs.streams
        if streams.available_thrust == 0 and streams.current_stage > 0:
            self.fs.vessel.control.throttle = 0.0

            # Disable the attitude-control worker (if one is running, i.e.
            # we're in CIRCULARIZE) for the duration of the staging event:
            # decoupler force / new engine spool-up / gimbal-torque changes
            # aren't accounted for by the controller, and the main thread is
            # about to block in time.sleep() through the event anyway. See
            # PLAN.md §8.4. We don't re-enable at the end of the staging
            # event -- the next planning loop iteration will naturally
            # publish a fresh GuidanceCommand.
            if self.worker is not None:
                self.worker.link.set(None)
            # time.sleep(0.5)  # brief pause so decouplers don't double-fire
            self.fs.vessel.control.activate_next_stage()
            print("\n  ⚡ STAGE SEPARATION")
            time.sleep(0.5)
            # Some craft designs need a second activation (e.g.
            # decouple then ignite)
            streams.next()
            if streams.available_thrust == 0 and streams.current_stage > 0:
                self.fs.vessel.control.activate_next_stage()
                print("  ⚡ ENGINE IGNITION")
                streams.next()
            return True
        return False

    def ascent(self) -> None:
        if self.phase != Phase.ASCENT:
            self.phase = Phase.ASCENT
            vessel = self.fs.vessel

            vessel.auto_pilot.engage()
            vessel.control.sas = False
            vessel.control.rcs = False
            vessel.control.throttle = 1.0
            vessel.auto_pilot.target_pitch_and_heading(90, self.HEADING)
            self.prev_pitch = 90
            vessel.auto_pilot.target_roll = 90
            # 2× physics warp during ascent
            self.fs.space_center.physics_warp_factor = 1

        altitude = self.fs.streams.altitude
        if altitude < self.turn_start_alt:
            target_pitch = 90.0
            phase_label = "Vertical ascent"
        elif altitude < self.turn_end_alt:
            frac = (altitude - self.turn_start_alt) / (
                self.turn_end_alt - self.turn_start_alt
            )
            target_pitch = 90.0 - (frac * 90.0)
            phase_label = "Gravity turn"
        else:
            target_pitch = 0.0
            phase_label = "Horizontal"

        # Only update autopilot when the angle changes meaningfully
        if abs(target_pitch - self.prev_pitch) > 0.5:
            self.prev_pitch = target_pitch
            self.fs.vessel.auto_pilot.target_pitch_and_heading(
                target_pitch, self.HEADING
            )

        if self.fs.streams.apoapsis > self.ENGINE_CUTOFF_ALTITUDE * self.AP_WARP_MARGIN:
            # 1× physics warp when close.
            self.fs.space_center.physics_warp_factor = 0

        if self.stage_if_needed():
            self.fs.vessel.control.throttle = 1.0

        apoapsis = self.fs.streams.apoapsis
        print(
            f"\rPitch {target_pitch:>5.1f}  "
            f"Apo {apoapsis:>5.0f}/{self.ENGINE_CUTOFF_ALTITUDE}  {phase_label}   ",
            end="",
            flush=True,
        )

    # Coasting out of atmosphere
    def atmosphere_coast(self) -> None:
        if self.phase != Phase.ATMOSPHERE_COAST:
            self.phase = Phase.ATMOSPHERE_COAST
            print("\nTarget apoapsis reached, coasting out of atmosphere.")
            self.fs.vessel.control.throttle = 0.0
            # 4× physics warp during coast
            self.fs.space_center.physics_warp_factor = 3

    def eccentricity_decreasing(self) -> bool:
        # This formula looks like the one for a circular orbit, but it's actually much
        # more general.  It's the sign of d eccentricity / d velocity.  In other words,
        # this tells you, as long as at least some component of thrust is in the
        # direction of velocity (thrust dot velocity is postive), whether that burn will
        # decrease eccentricity.  This formula doesn't just apply at apoapsis/periapsis
        # or for circular orbits, but for any spot on any orbit.
        streams = self.fs.streams
        speed = np.linalg.norm(streams.velocity)
        radius = np.linalg.norm(streams.position)
        return bool(speed * speed < self.mu / radius)

    def plan_circularization(self) -> None:
        """Replan the circularization burn and publish the result to the
        autopilot worker thread.

        This can safely take a long time (50 ms+, several physics ticks) --
        the worker thread keeps commanding attitude every tick from the
        last-published GuidanceCommand regardless (see PLAN.md §5).
        """
        start_wall = time.perf_counter()
        sim = Simulator(
            self.mu,
            self.body_radius,
            self.TARGET_ALTITUDE,
            self.segments,
            self.STAGING_DURATION,
        )

        streams = self.fs.streams
        r3d = np.array(streams.position)
        v3d = np.array(streams.velocity)
        tta = streams.time_to_apoapsis

        # Do we need ut0 for anything else?
        self.ut0 = streams.ut

        before_find_wall = time.perf_counter()
        self.plan = sim.find_linear_tangent_params(r3d, v3d, tta, verbose=False)
        after_find_wall = time.perf_counter()

        self.burn_start_time = self.plan.coast_time + self.ut0
        end_wall = time.perf_counter()
        print(
            f"Plan time: {(end_wall - start_wall) * 1000.0:>5.0f} "
            f"({(after_find_wall - before_find_wall) * 1000.0:>5.0f})ms  ",
            end="",
        )

        assert self.worker is not None
        self.worker.link.set(
            GuidanceCommand(
                ref_angle=self.plan.ref_angle,
                a_coeff=self.plan.a_coeff,
                b_coeff=self.plan.b_coeff,
                t0=self.burn_start_time,
                plane=self.plan.plane,
            )
        )

    def circularize(self) -> None:
        streams = self.fs.streams

        if self.phase != Phase.CIRCULARIZE:
            self.phase = Phase.CIRCULARIZE
            vessel = self.fs.vessel

            vessel.auto_pilot.disengage()
            vessel.control.sas = False

            # The attitude-control law runs on its own thread, over its own
            # kRPC connection, so replanning below (which can take multiple
            # physics ticks) never starves it. See PLAN.md.
            self.worker = AutopilotWorker()
            self.worker.start()

            # Will we get more accurate thrust direction vector if we only use reaction
            # wheel for direction, not gimballing?  Let's find out.
            for e in vessel.parts.engines:
                if e.active and e.has_fuel:
                    print(f"Locking gimbal on {e.part.title}")
                    e.gimbal_locked = True

            print("Coast and circularize!")

            self.segments = build_segments(vessel)

            streams.start()
            streams.next()

            self.plan_circularization()

            coast_time = self.burn_start_time - streams.ut

            if coast_time <= 10.0:
                self.fs.space_center.physics_warp_factor = 0
            else:
                self.fs.space_center.physics_warp_factor = 3

        # Coast or burn?
        if self.burn_start_time > streams.ut:
            coast_time = self.burn_start_time - streams.ut

            if coast_time <= 10.0:
                self.fs.space_center.physics_warp_factor = 0

            # theta is the steering angle at the very start of the burn
            # (t=0); the worker thread is already pointing there (and will
            # keep doing so until burn_start_time) via evaluate_target().
            theta = self.plan.ref_angle + math.atan(self.plan.a_coeff)
            thrust_dir_2d = np.array([math.cos(theta), math.sin(theta)])
            thrust_dir = self.plan.plane.from_plane(thrust_dir_2d)

            angle_error = math.degrees(math.acos(np.dot(streams.direction, thrust_dir)))

            print(
                f"\rcoast time: {coast_time:>5.1f}  ",
                f"Dir {angle_error:>3.2f}°  ",
                end="",
                flush=True,
            )

            # Replan. This can take a long time (50 ms+); the worker thread
            # keeps commanding attitude from the previous plan while this
            # runs, so there's no need to update it first.
            self.plan_circularization()

        else:
            self.fs.vessel.control.throttle = 1.0
            t = streams.ut - self.burn_start_time
            tan_val = self.plan.a_coeff + self.plan.b_coeff * t
            theta = self.plan.ref_angle + math.atan(tan_val)
            thrust_dir_2d = np.array([math.cos(theta), math.sin(theta)])
            thrust_dir = self.plan.plane.from_plane(thrust_dir_2d)

            if self.stage_if_needed():
                self.segments = build_segments(self.fs.vessel)
                self.plan_circularization()

            angle_error = math.degrees(math.acos(np.dot(streams.direction, thrust_dir)))
            target_apoapsis = self.plan.final_apoapsis_altitude
            target_periapsis = self.plan.final_periapsis_altitude
            print(
                f"\rDir {angle_error:>3.2f}°  "
                f"Ap {streams.apoapsis:>5.0f}/{target_apoapsis:>5.0f}  "
                f"Pe {streams.periapsis:>5.0f}/{target_periapsis:>5.0f}",
                end="",
                flush=True,
            )


def main() -> None:
    print("Connecting to kRPC server…")
    conn = krpc.connect(name="Gravity Turn")

    with FlightSession(conn) as fs:
        GravityTurn(fs, turn_start_alt=100, turn_end_alt=30_000).do_it()


if __name__ == "__main__":
    main()
