#!/usr/bin/env python3

import time

import krpc


def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp *value* between *lo* and *hi*."""
    return max(lo, min(hi, value))


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


HEADING = 90  # Launch azimuth (90 = due east for equatorial orbit)


def print_fuel(vessel):
    for p in vessel.parts.all:
        if "Fuel Tank" in p.title:
            print(f"mass={p.mass}, dry={p.dry_mass}, diff={p.mass - p.dry_mass}")


# ── Connect ─────────────────────────────────────────────────────────────
print("Connecting to kRPC server…")
conn = krpc.connect(name="Explore")
vessel = conn.space_center.active_vessel
print(f"  Vessel: {vessel.name}")

# ── Pre-Launch Setup ────────────────────────────────────────────────────
vessel.control.sas = False
vessel.control.rcs = False
vessel.control.throttle = 1.0

# ── Ignition ────────────────────────────────────────────────────────────
vessel.control.activate_next_stage()
vessel.auto_pilot.engage()
vessel.auto_pilot.target_pitch_and_heading(90, HEADING)
vessel.auto_pilot.target_roll = 90

time.sleep(0.5)
print_fuel(vessel)

while True:
    print("**********")
    print(f"{vessel.met=}")
    print_fuel(vessel)
    print("")
    for e in vessel.parts.engines:
        mass_rate = e.thrust / (e.specific_impulse * G0)
        print(f"{e.part.title}, {e.active=}, {e.max_vacuum_thrust=}, {mass_rate=}")
        for p in e.propellants:
            print(
                f"  {p.name}, {p.current_amount=}, {p.current_requirement=}, "
                f"{p.total_resource_available=}, {p.total_resource_capacity=}, "
                f"{p.ignore_for_isp=}, {p.ignore_for_thrust_curve=}, "
                f"{p.draw_stack_gauge=}, {p.is_deprived=}, {p.ratio=}"
            )
    time.sleep(0.5)
