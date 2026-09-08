We currently have an oracle finder in experiments/two_burn_reference.py .
However, this is experimental code mixed in with other experimental code.

This task is to create a separate oracle finder that will be used going forward,
whenever we encounter a rocket or state (position, velocity, fuel used, etc)
that's giving the solver problems.  The first thing we'll do is run the oracle
to find out exactly what the optimal, minimal fuel/maximum final mass trajectory
really is.

It should take in a 3D position, 3D velocity, and a segments list.  For now, it
should only pay attention to the first segment, and pretend the max_burn_time is
150 seconds.  Supporting staging and multiple segments is explicitly out of
scope/future work.

Look at the existing experiments/two_burn_reference.py as a reference but I want
a clean implementation without any baggage that's easy to understand and
maintain.  You can reuse code as long as it follows those goals.

The idea is to be offline, it can take minutes to run but not e.g. more than 10
minutes.

It should do:

- Pure lawden primer solver with the sigmoid annealing stuff.
- Burn/coast/burn
- coast/bun
- pure burn

It should take whatever one uses the least fuel/least delta-v/highest end mass.

The target is a circular orbit of radius given by the user, defaults to 80km
altitude.

The output should be initial costate values plus final mass, mass of fuel
consumed, and delta-v consumed.  It should also contain the control inputs,
namely thrust direction over time and also thrust start/stop times, along with
state (position, velocity, mass) and costate values over time of both burn and
coast phases.  I'm not sure the best format for things that vary over time, but
look at what solve_ivp returns, that's a good starting point.  Output in the 2D
orbital plane, but also output the plane definition so we can translate back to
3D if needed.

Use the 2D orbital plane based dynamics.  Polar coordinates, you can drop the
tangential position from the solver and costates.  For the Lawden solver,
explicitly represent mass and have a mass costate.  For the other solvers that
take burn/coast start times as parameters, follow John E. Prussing's convention
where the mass is implicit, calculated from the current time, so there's no mass
costate.

Use the three Kerbin test cases as examples.

Create a PLAN.md in this directory and wait for feedback before implementing.
Again, no explicit dependencies on anything in the examples/ directory as we'll
be deleting that.
