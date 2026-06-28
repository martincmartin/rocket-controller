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

    return {
        "thrust": thrust,
        "flow_rate": flow_rate,
        "fuel_duration": fuel_dur,
    }


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
    print(f"{vessel.mass=}")
    current_stage = vessel.control.current_stage

    # Start with the currently active engine groups.
    groups = _discover_engine_groups(vessel, active_only=True)

    max_iterations = 50  # safety limit
    for _ in range(max_iterations):
        if remaining_dv <= 0:
            break

        print(f"{remaining_dv=}, {groups=}")

        # ── If no groups, try to simulate the next staging event ─────
        if not groups:
            found = False
            while current_stage > 0:
                print(f"Simulating staging of stage {current_stage}.")
                for p in vessel.parts.all:
                    if p.decouple_stage == current_stage:
                        print(f"decoupling {p.title}, mass {p.mass}")

                # Mass of parts that separate at this stage.
                drop = sum(
                    p.mass
                    for p in vessel.parts.all
                    if p.decouple_stage == current_stage
                )
                print("Total mass dropping in this decoupling: {drop}")
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
        groups = [g for g in groups if g["fuel_duration"] > 0]
        if not groups:
            continue

        for g in groups:
            print(g)

        # ── Segment simulation ───────────────────────────────────────
        # Find the group that depletes first.
        min_dur = min(g["fuel_duration"] for g in groups)

        # Aggregate performance across all active groups.
        F_total = sum(g["thrust"] for g in groups)
        total_flow = sum(g["flow_rate"] for g in groups)
        if total_flow <= 0 or F_total <= 0:
            break
        ve = F_total / total_flow
        print(f"{ve=}")

        # Δv available in this segment (Tsiolkovsky).
        mass_consumed = total_flow * min_dur
        print(f"{mass_consumed=}, {m=}")
        if mass_consumed >= m:
            print(f"!!! WTF????")
            break
        dv_segment = ve * math.log(m / (m - mass_consumed))
        print(f"{dv_segment=}")

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
            print(f"mass of entire vessel after burn but before staging: {m} kg")
            groups = [g for g in groups if g["fuel_duration"] > min_dur + 0.001]
            for g in groups:
                g["fuel_duration"] -= min_dur

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


# ─── Main ───────────────────────────────────────────────────────────────────────


def main():
    # ── Tunable Parameters ──────────────────────────────────────────────────
    TARGET_ALTITUDE = 80_000  # Desired circular orbit altitude (m)
    TURN_START_ALT = 100  # Altitude to begin pitching over (m)
    TURN_END_ALT = 35_000  # Altitude at which pitch reaches 0° (horizontal)
    HEADING = 90  # Launch azimuth (90 = due east for equatorial orbit)
    # MAX_Q_THROTTLE = 0.75  # Throttle limit during max-Q region
    MAX_Q_THROTTLE = 1.0  # Throttle limit during max-Q region
    MAX_Q_LOW = 10_000  # Start of max-Q throttle-down band (m)
    MAX_Q_HIGH = 30_000  # End of max-Q throttle-down band (m)
    AP_THROTTLE_MARGIN = 0.95  # Start tapering throttle when Ap > this × target

    # ── Connect ─────────────────────────────────────────────────────────────
    print("Connecting to kRPC server…")
    conn = krpc.connect(name="Gravity Turn")
    vessel = conn.space_center.active_vessel
    print(f"  Vessel: {vessel.name}")

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

    # print("***** At launch")
    # compute_shit(vessel)

    # ═══════════════════════════════════════════════════════════════════════
    #  PHASE 1 — Ascent & Gravity Turn
    # ═══════════════════════════════════════════════════════════════════════
    print("\n── Phase 1: Ascent & Gravity Turn ──")
    conn.space_center.physics_warp_factor = 1  # 2× physics warp during ascent
    turn_angle = 0.0

    _discover_engine_groups(vessel)

    while True:
        alt = altitude()
        ap = apoapsis()

        # ── Gravity turn pitch profile ──────────────────────────────────
        if alt < TURN_START_ALT:
            # Vertical ascent
            target_pitch = 90.0
            phase = "Vertical ascent"
        elif alt < TURN_END_ALT:
            # Smooth sinusoidal pitch-over
            frac = (alt - TURN_START_ALT) / (TURN_END_ALT - TURN_START_ALT)
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
        if MAX_Q_LOW < alt < MAX_Q_HIGH:
            # Reduce throttle through max-Q to limit aerodynamic stress
            throttle = MAX_Q_THROTTLE
        elif ap > TARGET_ALTITUDE * AP_THROTTLE_MARGIN:
            # Taper throttle as apoapsis approaches the target
            remaining_frac = (TARGET_ALTITUDE - ap) / (
                TARGET_ALTITUDE * (1 - AP_THROTTLE_MARGIN)
            )
            throttle = clamp(remaining_frac, 0.05, 1.0)
        else:
            throttle = 1.0
        vessel.control.throttle = throttle

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
        if ap >= TARGET_ALTITUDE * 0.99:
            break

        time.sleep(0.05)

    # conn.space_center.physics_warp_factor = 0

    # print("\n***** While thrusting")
    # compute_shit(vessel)

    # vessel.control.throttle = 0.0
    # print("***** No thrust")
    # compute_shit(vessel)

    # while altitude() <= 70_000:
    #     time.sleep(1)

    # vessel.control.throttle = 1.0
    # print("\n***** While thrusting in vacuum")
    # compute_shit(vessel)

    # Cut throttle once target apoapsis is reached
    vessel.control.throttle = 0.0
    conn.space_center.physics_warp_factor = 3  # 4× physics warp during coast
    print(f"\n  ✓ Target apoapsis reached: {apoapsis():.0f} m")

    # ═══════════════════════════════════════════════════════════════════════
    #  PHASE 2 — Coast to Apoapsis & Circularization Burn
    # ═══════════════════════════════════════════════════════════════════════
    print("\n── Phase 2: Planning Circularization Burn ──")

    # Calculate the required Δv for a circular orbit at the target altitude
    r_target = body_radius + TARGET_ALTITUDE
    v_circular = math.sqrt(mu / r_target)
    # Vis-viva: current velocity at apoapsis of the transfer orbit
    a_transfer = (body_radius + vessel.orbit.periapsis_altitude + 2 * r_target) / 2
    # More precisely, use the actual semi-major axis
    a_transfer = vessel.orbit.semi_major_axis
    v_apoapsis = math.sqrt(mu * (2.0 / r_target - 1.0 / a_transfer))
    delta_v = v_circular - v_apoapsis
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

    # Point at the maneuver node burn vector
    vessel.auto_pilot.reference_frame = node.reference_frame
    vessel.auto_pilot.target_direction = (0, 1, 0)  # node's burn vector
    vessel.auto_pilot.wait()
    print("  Autopilot locked to maneuver node")

    # ── Wait until burn start (lead by half the burn duration) ──────────
    print("\n── Phase 3: Coasting to Burn Start ──")
    burn_ut = ut() + vessel.orbit.time_to_apoapsis - (burn_dur / 2.0)

    # Fine wait
    while ut() < burn_ut:
        time_remaining = burn_ut - ut()
        print(f"\r  Burn in {time_remaining:>6.1f} s   ", end="", flush=True)
        time.sleep(0.1)
    print()

    # ═══════════════════════════════════════════════════════════════════════
    #  PHASE 4 — Circularization Burn
    # ═══════════════════════════════════════════════════════════════════════
    conn.space_center.physics_warp_factor = 0  # back to 1× for the burn
    print("\n── Phase 4: Circularization Burn ──")
    vessel.control.throttle = 1.0

    # Burn until the remaining Δv in the node is very small
    remaining_dv = conn.add_stream(getattr, node, "remaining_delta_v")
    prev_remaining = remaining_dv()

    while True:
        dv = remaining_dv()

        # Throttle down as we approach the target to avoid overshooting
        if dv < 10:
            vessel.control.throttle = clamp(dv / 10.0, 0.02, 1.0)

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

        # If Δv starts increasing, we've overshot — stop immediately
        if dv > prev_remaining + 0.1 and dv < 5:
            break
        # If close enough, stop
        if dv < 0.2:
            break

        prev_remaining = dv
        print(f"\r  Remaining Δv: {dv:>7.2f} m/s   ", end="", flush=True)
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

    conn.close()


if __name__ == "__main__":
    main()
