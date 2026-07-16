#!/usr/bin/env python3

"""
Flight controller for Kerbal Space Program.

This module interfaces with KSP through kRPC and executes an automated launch
from the pad through gravity turn and into orbit. It performs the gravity turn,
staging, throttle management, and circularization burn while continuously
monitoring the live vehicle state.

Trajectory planning is delegated to sim.py. This module converts the current
vessel into the simulator's abstract RocketSegment model, requests an updated
circularization plan as the trajectory evolves, and flies the resulting
steering law in real time.

`FlightSession` owns the live kRPC handles (the vessel and any streams
opened against it) for a single flight attempt, and `run_campaign()` can
run `gravity_turn()` multiple times in one script invocation -- reverting
to launch between attempts -- to compare ascent parameters or characterize
run-to-run variation. See PLAN.md for the design rationale.

Tunable parameters are grouped at the top of main() for easy adjustment.
"""

import math
import time
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Optional

import krpc
import krpc.stream
import numpy as np
from collections import namedtuple
from krpc.services.spacecenter import Vessel

from sim import RocketSegment, Simulator

# Standard gravitational acceleration (m/s²), used for Isp ↔ exhaust velocity.
G0 = 9.80665

# KSP stock resource densities (kg per unit).
# Used to convert propellant amounts → mass for fuel‐duration estimates.
RESOURCE_DENSITY = {
    "LiquidFuel": 5.0,
    "Oxidizer": 5.0,
    "SolidFuel": 7.5,
    "MonoPropellant": 4.0,
    "XenonGas": 0.1,
    "ElectricCharge": 0.0,  # massless — ignored in flow calculations
    "IntakeAir": 0.0,
}


def engine_repr(engine):
    return engine.part.title


# setattr(krpc.services.spacecenter.Engine, "__format__", engine_formatter)
krpc.services.spacecenter.Engine.__repr__ = engine_repr


# ─── Helpers ────────────────────────────────────────────────────────────────────


def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp *value* between *lo* and *hi*."""
    return max(lo, min(hi, value))


class EngineGroup:
    """Snapshot of an engine group's performance and remaining fuel."""

    __slots__ = ("name", "thrust", "flow_rate", "fuel_duration")

    def __init__(self, name, thrust, flow_rate, fuel_duration):
        self.name = name  # representative engine's part.title
        self.thrust = thrust  # N
        self.flow_rate = flow_rate  # kg/s
        self.fuel_duration = fuel_duration  # seconds until limiting propellant depletes


def _engine_group_stats(engines):
    """Compute performance stats for a group of engines sharing fuel.

    Returns a dict with thrust (N), isp (s), ve (m/s), flow_rate (kg/s),
    and fuel_duration (s), or *None* if the group cannot produce thrust.
    """
    # Engine.max_vacuum_thrust: Newtons = kg m/s^2
    thrust = sum(e.max_vacuum_thrust for e in engines)
    if thrust <= 0:
        return None

    # total mass flow for this group (kg/s)
    flow_rate = 0.0
    for e in engines:
        # Engine.vacuum_specific_impulse: seconds
        # Ve: m / sec
        ve = e.vacuum_specific_impulse * G0
        if ve > 0:
            # Engine.max_vacuum_thrust: Newtons = kg m/s^2
            flow_rate += e.max_vacuum_thrust / ve
    if flow_rate <= 0:
        return None

    # Fuel duration: find limiting propellant.
    # Use the first engine as representative — the heuristic assumes
    # engines in the same group share the same fuel tanks.
    PropSnap = namedtuple("PropSnap", ["name", "ratio", "available"])
    rep = engines[0]
    propellants = [
        PropSnap(p.name, p.ratio, p.total_resource_available)
        for p in rep.propellants
        if p.ratio > 0
    ]
    if not propellants:
        return None

    # sum(ratio_i * density_i) for normalising unit‐consumption rates.
    # kg/unit
    sum_rd = sum(p.ratio * RESOURCE_DENSITY.get(p.name, 5.0) for p in propellants)
    if sum_rd <= 0:
        return None

    # Total "volume" of propellant consumed, in "resource units" per second.
    volume_rate = flow_rate / sum_rd

    fuel_dur = float("inf")
    for p in propellants:
        density = RESOURCE_DENSITY.get(p.name, 5.0)
        if density <= 0 or p.ratio <= 0:
            print(f"!!!!! {density=}, {p.ratio=}")
            continue  # massless resources don't limit burn
        unit_rate = volume_rate * p.ratio  # units/s consumed
        if unit_rate > 0:
            fuel_dur = min(fuel_dur, p.available / unit_rate)

    if fuel_dur in (float("inf"), 0):
        return None

    return EngineGroup(rep.part.title, thrust, flow_rate, fuel_dur)


def _discover_engine_groups(vessel, active_only=True, stage_filter=None):
    """Group engines by (decouple_stage, propellant types).

    Args:
        vessel:       kRPC vessel object.
        active_only:  If True only include engines that are active with fuel.
        stage_filter: If set only include engines whose *part.stage* matches.

    Returns a list of engine-group dicts (see ``_engine_group_stats``).
    """
    by_key = {}
    for engine in vessel.parts.engines:
        if active_only and (not engine.active or not engine.has_fuel):
            continue
        if stage_filter is not None and engine.part.stage != stage_filter:
            continue
        if engine.max_thrust <= 0:
            continue

        prop_names = frozenset(p.name for p in engine.propellants if p.ratio > 0)
        if not prop_names:
            continue
        key = (engine.part.decouple_stage, prop_names)
        by_key.setdefault(key, []).append(engine)

    groups = []
    for _key, eng_list in by_key.items():
        stats = _engine_group_stats(eng_list)
        if stats is not None:
            groups.append(stats)
    return groups


def build_segments(vessel) -> list[RocketSegment]:
    """Build the list of sim.RocketSegment instances describing the
    vessel's remaining engine/fuel state, for use with Simulator.

    Accounts for:
    * **Multiple engine groups** — engines with separate fuel supplies that
      deplete at different times within the same stage.
    * **Staging** — when all groups in the current stage are exhausted,
      decoupled mass is subtracted and engines in the next stage are used.

    Engine groups are identified by the heuristic key
    ``(decouple_stage, frozenset(propellant_names))``.

    The vessel's remaining burn is walked as a sequence of *segments*, each
    with a constant set of active engine groups (and therefore constant
    total thrust and combined Isp), becoming one ``RocketSegment`` each. A segment
    ends either when the first engine group within it runs dry while
    others still have fuel left (no part separation happens, so the next
    segment's ``RocketSegment.last_segment_of_stage`` is ``False`` and it begins
    immediately with the remaining group(s)), or when all currently active
    engine groups are spent simultaneously and staging is required to
    reach the next group of engines (``last_segment_of_stage=True``,
    modeled elsewhere as a 1 second coast before the next stage).
    """
    segments: list[RocketSegment] = []

    m = vessel.mass  # kg
    current_stage = vessel.control.current_stage

    # Start with the currently active engine groups.
    groups = _discover_engine_groups(vessel, active_only=True)

    max_iterations = 50  # safety limit
    for _ in range(max_iterations):
        # ── If no groups, try to simulate the next staging event ─────
        if not groups:
            found = False
            while current_stage > 0:
                # Dry mass of parts that separate at this stage.
                # Fuel mass was already subtracted from m during segment simulation.
                drop = sum(
                    p.dry_mass
                    for p in vessel.parts.all
                    if p.decouple_stage == current_stage
                )
                m -= drop
                current_stage -= 1
                groups = _discover_engine_groups(
                    vessel, active_only=False, stage_filter=current_stage
                )
                if groups:
                    found = True
                    break
            if not found:
                break

        # Remove groups that are already empty.
        groups = [g for g in groups if g.fuel_duration > 0]
        if not groups:
            continue

        # ── Segment simulation ───────────────────────────────────────
        # Find the group that depletes first.
        min_dur = min(g.fuel_duration for g in groups)

        # Aggregate performance across all active groups.
        F_total = sum(g.thrust for g in groups)
        total_flow = sum(g.flow_rate for g in groups)
        if total_flow <= 0 or F_total <= 0:
            break
        ve = F_total / total_flow
        name = " + ".join(g.name for g in groups)

        # If any engine group in this segment still has fuel left after
        # min_dur, the next segment continues immediately with those
        # groups (same hardware stage, no decoupling event).
        remaining_after = [g for g in groups if g.fuel_duration > min_dur + 0.001]
        last_segment_of_stage = not remaining_after

        segments.append(
            RocketSegment(
                name=name,
                ve=ve,
                thrust=F_total,
                max_burn_time=min_dur,
                initial_mass=m,
                last_segment_of_stage=last_segment_of_stage,
            )
        )

        mass_consumed = total_flow * min_dur
        if mass_consumed >= m:
            print(f"!!! WTF????")
            break
        m -= mass_consumed

        groups = remaining_after
        for g in groups:
            g.fuel_duration -= min_dur

    return segments


def print_telemetry(
    altitude,
    apoapsis,
    periapsis,
    pitch,
    throttle,
    speed,
    phase: str = "",
    eccentricity: float | None = None,
):
    """Print a single-line telemetry readout to the console."""
    ecc_str = f"Ecc {eccentricity:>7.5f}  " if eccentricity is not None else ""
    print(
        f"\r  {phase:<20s}  "
        f"Alt {altitude:>8.0f} m  "
        f"Ap {apoapsis:>8.0f} m  "
        f"Pe {periapsis:>8.0f} m  "
        f"{ecc_str}"
        f"Pitch {pitch:>5.1f}°  "
        f"Thr {throttle:>3.0%}  "
        f"Spd {speed:>7.1f} m/s",
        end="",
        flush=True,
    )


def resource_mass(vessel):
    return sum(r.amount * r.density for r in vessel.resources.all)


STAGING_DURATION = 2.5  # seconds; measured real-world KSP staging delay


# ─── Flight Session ─────────────────────────────────────────────────────────────


class FlightSession:
    """One flight attempt's live kRPC handles: the active vessel, plus
    every stream opened through this session.

    Valid only strictly between __enter__() returning and __exit__()
    running. Construction does no I/O and blocks on nothing --
    __enter__() is what waits for a fresh, pre-launch vessel and hands it
    out. `.vessel` and `.add_stream(...)` both raise immediately outside
    that window (i.e. before entering, or after exiting), rather than
    returning a stale or not-yet-ready handle.

    Does NOT revert/load the game. That's the caller's decision, made
    after the `with` block exits -- see `run_campaign()`.
    """

    READY_TIMEOUT = 60.0
    POLL_INTERVAL = 0.25

    # `conn` is deliberately typed as `Any`, not `krpc.client.Client`: the
    # vendored kRPC stub's own `Client.__init__` assigns `self.space_center`
    # etc. from a try/except ImportError fallback (`lambda _: None`), so
    # pyright infers those attributes as `X | None` and flags every
    # `conn.space_center.foo` access below as "possibly None" -- a false
    # positive from the stub's typing, not a real bug (see PLAN.md).
    def __init__(self, conn: Any) -> None:
        self.conn = conn
        self._streams: list[krpc.stream.Stream] = []
        self._ready = False  # True only strictly inside the `with` block
        self._vessel: Optional[Vessel] = None

    def __enter__(self) -> "FlightSession":
        self._vessel = self._wait_for_prelaunch(self.conn)
        self._vessel.control.sas = False
        self._vessel.control.rcs = False
        self._vessel.auto_pilot.target_pitch_and_heading(90, 90)
        self._vessel.auto_pilot.target_roll = 90
        self.conn.space_center.physics_warp_factor = 0

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

    def add_stream(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Like conn.add_stream(...), but the returned stream is removed
        automatically when this session closes.

        Return type is deliberately `Any`, not `krpc.stream.Stream`: the
        vendored kRPC stub's own `Stream.__call__` returns bare `object`
        (not a generic/TypeVar), so a precisely-typed `Stream` here would
        make every `some_stream()` read downstream (e.g. `altitude()`,
        `position()`) infer as `object` and fail comparisons/arithmetic
        against it -- a false positive from the stub's typing, not a real
        bug (see PLAN.md).
        """
        if not self._ready:
            raise RuntimeError(
                "FlightSession.add_stream(...) used outside an active session "
                "(before __enter__ or after __exit__)"
            )
        s = self.conn.add_stream(func, *args, **kwargs)
        self._streams.append(s)
        return s

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
        # No return value -- falsy, so an exception raised in the `with`
        # body is never swallowed.

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
        self._try(lambda: setattr(self.conn.space_center, "physics_warp_factor", 0))

        for s in self._streams:
            self._try(s.remove)
        self._streams.clear()

        self._vessel = None
        self._ready = False

    @staticmethod
    def _try(action: Callable[[], Any]) -> None:
        try:
            action()
        except Exception as e:  # noqa: BLE001 - teardown must not raise
            print(f"  ! FlightSession cleanup warning: {e}")

    @classmethod
    def _wait_for_prelaunch(cls, conn: Any) -> Vessel:
        """Poll until a vessel exists, in the flight scene, sitting on
        the pad. Needed because revert_to_launch()/load() return before
        the scene has actually finished reloading."""
        deadline = time.monotonic() + cls.READY_TIMEOUT
        pre_launch = conn.space_center.VesselSituation.pre_launch
        flight_scene = conn.krpc.GameScene.flight
        while True:
            try:
                # current_game_scene is a coarse, cheap guard against
                # reading a stale/leftover active_vessel while the scene
                # itself is still transitioning (e.g. mid-load()).
                if conn.krpc.current_game_scene == flight_scene:
                    vessel = conn.space_center.active_vessel
                    if vessel.situation == pre_launch:
                        return vessel
            except Exception:
                pass  # scene mid-transition; retry until the timeout
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "Timed out waiting for the vessel to reach pre-launch"
                )
            time.sleep(cls.POLL_INTERVAL)


@dataclass
class FlightResult:
    """Outcome of one gravity_turn() attempt."""

    turn_start_alt: float
    turn_end_alt: float
    final_apoapsis: float
    final_periapsis: float
    final_mass: float
    error: str | None = None


# Main gravity turn implementation.
# TURN_START_ALT = 100  # Altitude to begin pitching over (m)
# TURN_END_ALT = 35_000  # Altitude at which pitch reaches 0° (horizontal)
def gravity_turn(
    fs: FlightSession, turn_start_alt: float, turn_end_alt: float
) -> FlightResult:
    # ── Tunable Parameters ──────────────────────────────────────────────────
    TARGET_ALTITUDE = 80_000  # Desired circular orbit altitude (m)
    ENGINE_CUTOFF_ALTITUDE = 60_000  # Once apopasis reaches this, cut engines.
    HEADING = 90  # Launch azimuth (90 = due east for equatorial orbit)
    # MAX_Q_THROTTLE = 0.75  # Throttle limit during max-Q region
    MAX_Q_THROTTLE = 1.0  # Throttle limit during max-Q region
    MAX_Q_LOW = 10_000  # Start of max-Q throttle-down band (m)
    MAX_Q_HIGH = 30_000  # End of max-Q throttle-down band (m)
    AP_WARP_MARGIN = 0.90  # Turn off warp when Ap > this x target
    AP_THROTTLE_MARGIN = 0.95  # Start tapering throttle when Ap > this × target
    ATMOSPHERE_ALTITUDE = 25_000  # Kerbin atmosphere is 0.01 atm at 25k, 0.001 at 40k.

    conn = fs.conn
    vessel = fs.vessel
    print(f"  Vessel: {vessel.name}")

    mass = resource_mass(vessel)
    print(f"Starting resource mass: {mass} kg")

    # Reference body parameters (Kerbin)
    body = vessel.orbit.body
    body_radius = body.equatorial_radius
    mu = body.gravitational_parameter
    frame = body.non_rotating_reference_frame
    print(f"  Body: {body.name}  (R = {body_radius:.0f} m, μ = {mu:.3e} m³/s²)")

    # ── Telemetry Streams ───────────────────────────────────────────────────
    # Streams are much faster than polling properties repeatedly.
    ut = fs.add_stream(getattr, conn.space_center, "ut")
    altitude = fs.add_stream(getattr, vessel.flight(), "mean_altitude")
    apoapsis = fs.add_stream(getattr, vessel.orbit, "apoapsis_altitude")
    periapsis = fs.add_stream(getattr, vessel.orbit, "periapsis_altitude")
    eccentricity = fs.add_stream(getattr, vessel.orbit, "eccentricity")
    speed = fs.add_stream(
        getattr, vessel.flight(vessel.orbit.body.reference_frame), "speed"
    )
    # Used by the Coast & Replan loop below. Streamed (rather than the
    # blocking vessel.position(frame)/vessel.velocity(frame) calls) so
    # repeated reads don't each cost a network round trip.
    position = fs.add_stream(vessel.position, frame)
    velocity = fs.add_stream(vessel.velocity, frame)
    # Auto-staging checks (ascent + circularization loops) and the Coast
    # & Replan loop's replanning input -- all read every iteration of a
    # tight loop, so streamed for the same reason as the telemetry above.
    available_thrust = fs.add_stream(getattr, vessel, "available_thrust")
    current_stage = fs.add_stream(getattr, vessel.control, "current_stage")
    thrust = fs.add_stream(getattr, vessel, "thrust")
    time_to_apoapsis = fs.add_stream(getattr, vessel.orbit, "time_to_apoapsis")

    # ── Pre-Launch Setup ────────────────────────────────────────────────────
    vessel.control.sas = False
    vessel.control.rcs = False
    vessel.control.throttle = 1.0

    # ── Ignition ────────────────────────────────────────────────────────────
    vessel.control.activate_next_stage()
    vessel.auto_pilot.engage()
    vessel.auto_pilot.target_pitch_and_heading(90, HEADING)
    vessel.auto_pilot.target_roll = 90

    # ═══════════════════════════════════════════════════════════════════════
    #  Ascent & Gravity Turn
    # ═══════════════════════════════════════════════════════════════════════
    print("\n── Ascent & Gravity Turn ──")
    conn.space_center.physics_warp_factor = 1  # 2× physics warp during ascent
    turn_angle = 0.0

    while True:
        alt = altitude()
        ap = apoapsis()

        # ── Gravity turn pitch profile ──────────────────────────────────
        if alt < turn_start_alt:
            # Vertical ascent
            target_pitch = 90.0
            phase = "Vertical ascent"
        elif alt < turn_end_alt:
            # Smooth sinusoidal pitch-over
            frac = (alt - turn_start_alt) / (turn_end_alt - turn_start_alt)
            target_pitch = 90.0 - (frac * 90.0)
            phase = "Gravity turn"
        else:
            target_pitch = 0.0
            phase = "Horizontal"

        # Only update autopilot when the angle changes meaningfully
        if abs(target_pitch - turn_angle) > 0.5:
            turn_angle = target_pitch
            vessel.auto_pilot.target_pitch_and_heading(turn_angle, HEADING)

        # ── Throttle management ─────────────────────────────────────────
        # if MAX_Q_LOW < alt < MAX_Q_HIGH:
        #     # Reduce throttle through max-Q to limit aerodynamic stress
        #     throttle = MAX_Q_THROTTLE
        if ap > ENGINE_CUTOFF_ALTITUDE * AP_THROTTLE_MARGIN:
            # Taper throttle as apoapsis approaches the target
            remaining_frac = (ENGINE_CUTOFF_ALTITUDE - ap) / (
                ENGINE_CUTOFF_ALTITUDE * (1 - AP_THROTTLE_MARGIN)
            )
            print(f"{remaining_frac=}")
            throttle = clamp(remaining_frac, 0.05, 1.0)
        else:
            throttle = 1.0
        vessel.control.throttle = throttle

        if ap > ENGINE_CUTOFF_ALTITUDE * AP_WARP_MARGIN:
            conn.space_center.physics_warp_factor = 0  # 1× physics warp when close.

        # ── Auto-staging (fuel depletion check) ─────────────────────────
        if available_thrust() == 0 and current_stage() > 0:
            time.sleep(0.5)  # brief pause so decouplers don't double-fire
            vessel.control.activate_next_stage()
            print("\n  ⚡ STAGE SEPARATION")
            time.sleep(0.5)
            # Some craft designs need a second activation (e.g. decouple then ignite)
            if available_thrust() == 0 and current_stage() > 0:
                vessel.control.activate_next_stage()
                print("  ⚡ ENGINE IGNITION")
                time.sleep(0.3)

        # ── Telemetry ───────────────────────────────────────────────────
        print_telemetry(alt, ap, periapsis(), turn_angle, throttle, speed(), phase)

        # ── Exit condition: apoapsis reached ────────────────────────────
        if ap >= ENGINE_CUTOFF_ALTITUDE:
            break

        time.sleep(0.05)

    # Cut throttle once target apoapsis is reached
    vessel.control.throttle = 0.0
    conn.space_center.physics_warp_factor = 3  # 4× physics warp during coast
    print(
        f"\n  ✓ Target apoapsis reached: {apoapsis():.0f} m, waiting until out of atmosphere."
    )

    # Wait for solid boosters to burn out, and to be (mostly) out of the atmosphere
    while thrust() > 0 or altitude() < ATMOSPHERE_ALTITUDE:
        time.sleep(0.1)

    # ═══════════════════════════════════════════════════════════════════════
    #  Coast & Replan to Burn Start
    # ═══════════════════════════════════════════════════════════════════════
    print("\n── Coast & Replan to Burn Start ──")

    vessel.auto_pilot.reference_frame = frame
    vessel.auto_pilot.stopping_time = (
        2,
        2,
        2,
    )  # gentler corrections to avoid oscillation

    # Engine/fuel state doesn't change while coasting (no thrust, no
    # staging), so this is computed once here rather than re-derived
    # (many blocking per-engine/per-part round trips) on every iteration
    # of the loop below.
    segments = build_segments(vessel)

    first_iteration = True
    while True:
        # Snapshot position/velocity once per iteration, under both
        # streams' condition locks so they're guaranteed to come from the
        # same physics tick as each other (see PLAN.md §2.4). Read once
        # and reuse for the rest of this iteration: find_linear_tangent_
        # params() below can take 50 ms+, during which the streamed
        # values would otherwise advance past what the plan was based on.
        r3d = np.array(position())
        v3d = np.array(velocity())
        tta = time_to_apoapsis()
        ut0 = ut()  # pre-call snapshot -- see burn_start_time below

        # Replan on every iteration against the vessel's live state: fuel
        # burned, drag-induced orbital changes, and elapsed time all shift
        # the optimal (coast_time, a_coeff, b_coeff, burn_time) solution.
        sim = Simulator(mu, body_radius, TARGET_ALTITUDE, segments, STAGING_DURATION)
        plan = sim.find_linear_tangent_params(r3d, v3d, tta, verbose=False)

        # plan.coast_time is "seconds from ut0 until the burn should
        # start", not "seconds from now" -- find_linear_tangent_params()
        # above can take 50 ms+, so anchor it to ut0 (read before the
        # call) rather than a fresh ut() (read after), which would
        # silently double-count however long the solve just took.
        burn_start_time = plan.coast_time + ut0

        # Drop out of physics warp shortly before the burn so the autopilot
        # has full control authority to settle into the burn attitude.
        if plan.coast_time <= 10.0:
            conn.space_center.physics_warp_factor = 0

        # Point toward the burn's initial attitude while coasting, so the
        # craft has time to rotate into position before ignition.
        theta0 = plan.ref_angle + math.atan(plan.a_coeff)
        initial_dir = plan.plane.from_plane(
            np.array([math.cos(theta0), math.sin(theta0)])
        )
        vessel.auto_pilot.target_direction = tuple(initial_dir)

        print(
            f"\r  Coast {plan.coast_time:>6.1f} s  "
            f"a {plan.a_coeff:>8.5f}  b {plan.b_coeff:>9.6f}  "
            f"Burn {plan.burn_time:>6.1f} s  "
            f"Ap {apoapsis():>7.0f}/{plan.final_apoapsis_altitude:<7.0f} m  "
            f"Pe {periapsis():>7.0f}/{plan.final_periapsis_altitude:<7.0f} m  "
            f"Ecc {eccentricity():>7.5f}   ",
            end="",
            flush=True,
        )
        if first_iteration:
            print()  # preserve the initial values in scrollback for comparison
            first_iteration = False

        if ut() >= burn_start_time:
            break
    print()

    # ═══════════════════════════════════════════════════════════════════════
    #  Circularization Burn
    # ═══════════════════════════════════════════════════════════════════════
    conn.space_center.physics_warp_factor = 0  # back to 1× for the burn
    print("\n── Circularization Burn ──")
    throttle = 1.0
    vessel.control.throttle = throttle
    burn_start_ut = ut()
    prev_ecc = eccentricity()
    ECC_TOLERANCE = 0.1  # "circular enough" eccentricity to stop the burn
    BURN_TIME_SAFETY_MARGIN = 1.5  # abort if we run this much past the plan

    while True:
        t = ut() - burn_start_ut
        theta = plan.ref_angle + math.atan(plan.a_coeff + plan.b_coeff * t)
        thrust_dir = plan.plane.from_plane(np.array([math.cos(theta), math.sin(theta)]))
        vessel.auto_pilot.target_direction = tuple(thrust_dir)

        # Auto-staging during burn
        if available_thrust() == 0 and current_stage() > 0:
            separation_start = ut()
            throttle = 0.0
            vessel.control.throttle = throttle
            vessel.control.activate_next_stage()
            print("\n  ⚡ STAGE SEPARATION")
            time.sleep(0.5)
            if available_thrust() == 0 and current_stage() > 0:
                vessel.control.activate_next_stage()
                print("  ⚡ ENGINE IGNITION")
                time.sleep(0.3)
            throttle = 1.0
            vessel.control.throttle = throttle
            print(f"Staging took: {ut() - separation_start} sec")

        ecc = eccentricity()
        print_telemetry(
            altitude(),
            apoapsis(),
            periapsis(),
            math.degrees(theta),
            throttle,
            speed(),
            "Circularizing",
            eccentricity=ecc,
        )

        # Stop once eccentricity is close enough to zero (apsides equal)
        # AND has just started rising again (we've passed the minimum).
        if ecc <= ECC_TOLERANCE and ecc > prev_ecc:
            break
        if t > plan.burn_time * BURN_TIME_SAFETY_MARGIN:
            print("\n  ⚠ Burn exceeded planned duration; stopping.")
            break
        prev_ecc = ecc

    vessel.control.throttle = 0.0
    print(f"\n  ✓ Circularization complete!")

    # ── Final Orbit Summary ─────────────────────────────────────────────
    time.sleep(1)
    final_ap = vessel.orbit.apoapsis_altitude
    final_pe = vessel.orbit.periapsis_altitude
    final_inc = vessel.orbit.inclination
    print("\n══════════════════════════════════════════════")
    print("  ORBIT ACHIEVED")
    print(f"  Apoapsis:     {final_ap:>10,.0f} m")
    print(f"  Periapsis:    {final_pe:>10,.0f} m")
    print(f"  Inclination:  {math.degrees(final_inc):>10.2f}°")
    print(f"  Eccentricity: {vessel.orbit.eccentricity:>10.6f}")
    print("══════════════════════════════════════════════\n")

    vessel.auto_pilot.disengage()
    vessel.control.sas = True
    print("Autopilot disengaged. SAS enabled. Have a safe flight! 🚀")

    final_mass = resource_mass(vessel)
    print(
        f"Remaining resource mass: {final_mass} kg, used mass: {mass - final_mass} kg"
    )

    return FlightResult(
        turn_start_alt=turn_start_alt,
        turn_end_alt=turn_end_alt,
        final_apoapsis=final_ap,
        final_periapsis=final_pe,
        final_mass=final_mass,
    )


# ─── Multi-run harness ──────────────────────────────────────────────────────────


def _run_one_attempt(conn: Any, params: dict[str, Any]) -> FlightResult:
    """Run a single gravity_turn() attempt, converting any exception into
    a FlightResult(error=...) instead of letting it escape. This is the
    one place a per-attempt flight-logic failure is caught -- see
    run_campaign()'s docstring for why that's different from a
    harness-plumbing failure."""
    try:
        # FlightSession(conn) itself is side-effect-free; entering the
        # `with` block is what waits for pre-launch, so the wait is
        # inside the try along with the flight itself.
        with FlightSession(conn) as fs:
            return gravity_turn(fs, **params)
    except Exception as e:
        traceback.print_exc()
        return FlightResult(
            turn_start_alt=params["turn_start_alt"],
            turn_end_alt=params["turn_end_alt"],
            final_apoapsis=math.nan,
            final_periapsis=math.nan,
            final_mass=math.nan,
            error=str(e),
        )


def run_campaign(
    conn: Any,
    param_sets: list[dict[str, Any]],
    revert_after_last: bool = False,
) -> list[FlightResult]:
    """Run gravity_turn() once per entry in param_sets, reverting to
    launch between attempts.

    A failing attempt (an exception raised anywhere in that attempt's
    FlightSession/gravity_turn() call) is caught, recorded as a
    FlightResult(error=...), and the sweep continues. Failing to revert
    (or the game reporting it can't) aborts the remaining sweep instead --
    that's a harness-plumbing failure rather than a flight-logic failure,
    and silently continuing after it would mean re-flying an
    already-flown vessel and mislabeling the result as a fresh attempt.

    `revert_after_last` controls whether the *final* attempt is reverted
    too. Default False, so a single run (or the last run of a sweep)
    simply leaves the vessel wherever it ended up -- e.g. in orbit -- and
    the script can just exit without touching the save.
    """
    results: list[FlightResult] = []

    for i, params in enumerate(param_sets):
        is_last = i == len(param_sets) - 1
        results.append(_run_one_attempt(conn, params))

        if not is_last or revert_after_last:
            if not conn.space_center.can_revert_to_launch():
                raise RuntimeError(
                    "Cannot revert to launch; aborting the remaining sweep"
                )
            conn.space_center.revert_to_launch()

    return results


# ─── Main ───────────────────────────────────────────────────────────────────────


def main():
    # ── Connect ─────────────────────────────────────────────────────────────
    print("Connecting to kRPC server…")
    conn = krpc.connect(name="Gravity Turn")

    results = run_campaign(
        conn,
        [dict(turn_start_alt=100, turn_end_alt=30_000)],
    )
    for result in results:
        print(result)

    conn.close()


if __name__ == "__main__":
    main()
