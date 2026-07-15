KSP Ascent Autopilot


# Overview
An ascent optimizier and autopilot for Kerbal Space Program.

# Architecture

The project separates planning from execution. sim.py computes the optimal
maneuver using only physical quantities, making it independent of Kerbal Space
Program. gravity_turn.py is responsible for interacting with KSP via kRPC and
executing those plans on a live vehicle.

## `gravity_turn.py` — Flight Controller

Interfaces with Kerbal Space Program through kRPC and flies the vehicle in
real time. It is responsible for launch, gravity turn, staging, throttle
control, telemetry, and executing orbital maneuvers.

It converts the current vehicle state into an abstract rocket model,
continuously replans the circularization burn as conditions change, and
commands the autopilot to follow the resulting guidance law.

## `sim.py` — Trajectory Planner

A game-independent astrodynamics library used to plan orbital maneuvers. It
contains the physics simulation, orbital mechanics, trajectory propagation,
and numerical optimization routines.

Given the vehicle's physical characteristics and current state, it computes
the steering law and burn timing required to achieve the target orbit. It has
no dependency on kRPC or any other KSP-specific APIs.
