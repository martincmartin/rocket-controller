#!/usr/bin/env python3
"""
Gravity Turn Launch Script for Kerbin (KSP + kRPC)

Performs an automated launch from the pad through gravity turn and into
a circular orbit at the specified target altitude.

Phases:
  1. Vertical ascent
  2. Gravity turn (smooth pitch-over as a function of altitude)
  3. Coast to target apoapsis with throttle tapering
  4. Coast to apoapsis for circularization
  5. Circularization burn via maneuver node

Usage:
  1. Place your rocket on the launch pad in KSP
  2. Make sure the kRPC server is running (toolbar → kRPC → Start Server)
  3. Run:  python gravity_turn.py

Tunable parameters are grouped at the top of main() for easy adjustment.
"""

import math
import time
import krpc
import numpy as np
from collections import namedtuple

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

    __slots__ = ("thrust", "flow_rate", "fuel_duration")

    def __init__(self, thrust, flow_rate, fuel_duration):
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

    print(f"{rep.part.title}: {thrust=}, {flow_rate=}, {fuel_dur=}")
    return EngineGroup(thrust, flow_rate, fuel_dur)


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


def burn_time(vessel, delta_v: float) -> float:
    """Estimate burn duration for *delta_v* m/s.

    Accounts for:
    * **Multiple engine groups** — engines with separate fuel supplies that
      deplete at different times within the same stage.
    * **Staging** — when all groups in the current stage are exhausted,
      decoupled mass is subtracted and engines in the next stage are used.

    Engine groups are identified by the heuristic key
    ``(decouple_stage, frozenset(propellant_names))``.

    The burn is simulated as a sequence of *segments*.  Each segment has a
    constant set of active engine groups (and therefore constant total
    thrust and combined Isp).  A segment ends when the first group runs
    out of fuel, at which point that group is removed and a new segment
    begins with the remaining groups.
    """
    remaining_dv = delta_v
    total_time = 0.0
    m = vessel.mass  # kg
    current_stage = vessel.control.current_stage

    # Start with the currently active engine groups.
    groups = _discover_engine_groups(vessel, active_only=True)

    max_iterations = 50  # safety limit
    for _ in range(max_iterations):
        if remaining_dv <= 0:
            break

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

        # Δv available in this segment (Tsiolkovsky).
        mass_consumed = total_flow * min_dur
        if mass_consumed >= m:
            print(f"!!! WTF????")
            break
        dv_segment = ve * math.log(m / (m - mass_consumed))

        if dv_segment >= remaining_dv:
            # We finish the burn inside this segment.
            m_after = m * math.exp(-remaining_dv / ve)
            t = (m - m_after) / total_flow
            total_time += t
            remaining_dv = 0
        else:
            # Entire segment is consumed — remove the depleted group.
            total_time += min_dur
            remaining_dv -= dv_segment
            m -= mass_consumed
            groups = [g for g in groups if g.fuel_duration > min_dur + 0.001]
            for g in groups:
                g.fuel_duration -= min_dur

    return total_time


def print_telemetry(
    altitude, apoapsis, periapsis, pitch, throttle, speed, phase: str = ""
):
    """Print a single-line telemetry readout to the console."""
    print(
        f"\r  {phase:<20s}  "
        f"Alt {altitude:>8.0f} m  "
        f"Ap {apoapsis:>8.0f} m  "
        f"Pe {periapsis:>8.0f} m  "
        f"Pitch {pitch:>5.1f}°  "
        f"Thr {throttle:>3.0%}  "
        f"Spd {speed:>7.1f} m/s",
        end="",
        flush=True,
    )


def resource_mass(vessel):
    return sum(r.amount * r.density for r in vessel.resources.all)


def plan_circularization(vessel):
    body = vessel.orbit.body
    frame = body.non_rotating_reference_frame

    # Position (m)
    r3d = np.array(vessel.position(frame))

    # Velocity (m/s)
    v3d = np.array(vessel.velocity(frame))

    print(
        f"{r3d=}, {v3d=}, {vessel.mass=}, mu = {body.gravitational_parameter}, time to apopasis={vessel.orbit.time_to_apoapsis}"
    )

    # Reduce to 2D in the orbital plane.  r_hat will be our new x axis, w_hat
    # our new y.

    # r_hat, w_hat, r, v = project(r3d, v3d)

    # mu = body.gravitational_parameter
    # m0 = vessel.mass


"""
To plan circularization burn, consider using solve_ivp function from the SciPy library
with the 4 systems of equations:

d^2 r_ / d t^2 =−μ/r^3 ​r_ + (m0 - T​/ve t) v_hat
where r_ and v_ are 2D vectors, and v_ = d r_ / d t

Could use scipy.optimize.minimize_scalar to then find the best time to start the burn.

"For spacecraft dynamics, solve_ivp with the "DOP853" method (high accuracy,
non-stiff) is often an excellent choice:"

For when to start the burn, choose instead the universal variable, and determine
time, x_vec and v_vec from that.  https://en.wikipedia.org/wiki/Universal_variable_formulation
and Bate, Mueller & Whites Fundamentals of Astrodynamics section 4.3.  Or, during
the integration, in the timestep where the thrust is turned on, just assume the thrust
is proportional to the % time on?  In other words, if youre doing full thrust for 60%
of the time step, assume 60% thrust for the whole time step?  Would be a lot easier
and probably close enough...

Could also consider poliastro: https://docs.poliastro.space/en/stable/
"""


# Main gravity turn implementation.
# TURN_START_ALT = 100  # Altitude to begin pitching over (m)
# TURN_END_ALT = 35_000  # Altitude at which pitch reaches 0° (horizontal)
def gravity_turn(conn, turn_start_alt, turn_end_alt):
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

    vessel = conn.space_center.active_vessel
    print(f"  Vessel: {vessel.name}")

    mass = resource_mass(vessel)
    print(f"Starting resource mass: {mass} kg")

    # Reference body parameters (Kerbin)
    body = vessel.orbit.body
    body_radius = body.equatorial_radius
    mu = body.gravitational_parameter
    print(f"  Body: {body.name}  (R = {body_radius:.0f} m, μ = {mu:.3e} m³/s²)")

    # ── Telemetry Streams ───────────────────────────────────────────────────
    # Streams are much faster than polling properties repeatedly.
    ut = conn.add_stream(getattr, conn.space_center, "ut")
    altitude = conn.add_stream(getattr, vessel.flight(), "mean_altitude")
    apoapsis = conn.add_stream(getattr, vessel.orbit, "apoapsis_altitude")
    periapsis = conn.add_stream(getattr, vessel.orbit, "periapsis_altitude")
    speed = conn.add_stream(
        getattr, vessel.flight(vessel.orbit.body.reference_frame), "speed"
    )
    stage_fuel = None  # set up after first staging event

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
    #  PHASE 1 — Ascent & Gravity Turn
    # ═══════════════════════════════════════════════════════════════════════
    print("\n── Phase 1: Ascent & Gravity Turn ──")
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
            print("Turning off warp!")
            conn.space_center.physics_warp_factor = 0  # 1× physics warp when close.

        # ── Auto-staging (fuel depletion check) ─────────────────────────
        if vessel.available_thrust == 0 and vessel.control.current_stage > 0:
            time.sleep(0.5)  # brief pause so decouplers don't double-fire
            vessel.control.activate_next_stage()
            print("\n  ⚡ STAGE SEPARATION")
            time.sleep(0.5)
            # Some craft designs need a second activation (e.g. decouple then ignite)
            if vessel.available_thrust == 0 and vessel.control.current_stage > 0:
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
    print(f"\n  ✓ Target apoapsis reached: {apoapsis():.0f} m")

    # Wait for solid boosters to burn out, and to be (mostly) out of the atmosphere
    while vessel.thrust > 0 or altitude() < ATMOSPHERE_ALTITUDE:
        time.sleep(0.1)

    # ═══════════════════════════════════════════════════════════════════════
    #  PHASE 2 — Coast to Apoapsis & Circularization Burn
    # ═══════════════════════════════════════════════════════════════════════
    print("\n── Phase 2: Planning Circularization Burn ──")

    plan_circularization(vessel)

    # This should be a helper function.
    # Calculate the required Δv to raise periapsis to desired value.
    r_target = body_radius + TARGET_ALTITUDE
    r_apoapsis = body_radius + apoapsis()
    v_now = math.sqrt(mu * (2 / r_apoapsis - 1 / vessel.orbit.semi_major_axis))
    v_goal = math.sqrt(mu * (2 / r_apoapsis - 2 / (r_target + r_apoapsis)))

    delta_v = v_goal - v_now
    print(f"  Δv for circularization: {delta_v:.1f} m/s")

    # Compute burn time
    burn_dur = burn_time(vessel, delta_v)
    print(f"  Estimated burn time:    {burn_dur:.1f} s")

    # Add a maneuver node at apoapsis
    node = vessel.control.add_node(
        ut() + vessel.orbit.time_to_apoapsis,
        prograde=delta_v,
    )
    print(f"  Maneuver node placed at T+{vessel.orbit.time_to_apoapsis:.0f} s")

    # Point prograde for coast and circularization burn
    vessel.auto_pilot.reference_frame = vessel.orbital_reference_frame
    vessel.auto_pilot.target_direction = (0, 1, 0)  # prograde in orbital frame
    vessel.auto_pilot.stopping_time = (
        2,
        2,
        2,
    )  # gentler corrections to avoid oscillation

    # Wait until pointing within 5° (auto_pilot.wait() demands too-tight tolerance)
    alignment_timeout = 60  # seconds
    t0 = time.time()
    while time.time() - t0 < alignment_timeout:
        if vessel.auto_pilot.error < 5.0:
            break
        time.sleep(0.25)
    print(f"  Autopilot aligned to prograde (error: {vessel.auto_pilot.error:.1f}°)")

    # ── Wait until burn start (lead by half the burn duration) ──────────
    print("\n── Phase 3: Coasting to Burn Start ──")
    burn_ut = ut() + vessel.orbit.time_to_apoapsis - (burn_dur / 2.0)

    # Fine wait
    while ut() < burn_ut:
        time_remaining = burn_ut - ut()
        print(
            f"\r  Burn in {time_remaining:>6.1f} s, apopasis={apoapsis()}   ",
            end="",
            flush=True,
        )
        time.sleep(0.1)
    print()

    # ═══════════════════════════════════════════════════════════════════════
    #  PHASE 4 — Circularization Burn
    # ═══════════════════════════════════════════════════════════════════════
    conn.space_center.physics_warp_factor = 0  # back to 1× for the burn
    print("\n── Phase 4: Circularization Burn ──")
    vessel.control.throttle = 1.0

    # Burn until periapsis reaches the target altitude
    prev_pe = periapsis()

    while True:
        pe = periapsis()

        # Throttle down as periapsis approaches the target for precision
        pe_remaining = TARGET_ALTITUDE - pe
        if pe_remaining < 2000:
            vessel.control.throttle = clamp(pe_remaining / 2000.0, 0.02, 1.0)

        # Auto-staging during burn
        if vessel.available_thrust == 0 and vessel.control.current_stage > 0:
            vessel.control.throttle = 0.0
            time.sleep(0.5)
            vessel.control.activate_next_stage()
            print("\n  ⚡ STAGE SEPARATION")
            time.sleep(0.5)
            if vessel.available_thrust == 0 and vessel.control.current_stage > 0:
                vessel.control.activate_next_stage()
                print("  ⚡ ENGINE IGNITION")
                time.sleep(0.3)
            vessel.control.throttle = 1.0

        # Stop when periapsis reaches (or overshoots) the target
        if pe >= TARGET_ALTITUDE * 0.99:
            break
        # Stop if periapsis starts dropping (we've passed apoapsis)
        if pe < prev_pe - 100 and pe > TARGET_ALTITUDE * 0.5:
            break

        prev_pe = pe
        print(
            f"\r  Periapsis: {pe:>10,.0f} m  (target: {TARGET_ALTITUDE:,} m)   ",
            end="",
            flush=True,
        )
        time.sleep(0.05)

    vessel.control.throttle = 0.0
    node.remove()
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
    return final_mass


# ─── Main ───────────────────────────────────────────────────────────────────────


def main():
    # ── Connect ─────────────────────────────────────────────────────────────
    print("Connecting to kRPC server…")
    conn = krpc.connect(name="Gravity Turn")

    gravity_turn(conn, 100, 15_000)

    conn.close()


if __name__ == "__main__":
    main()
