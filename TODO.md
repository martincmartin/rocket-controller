# Sim.py
- Save a dimension in search by doing a 1D search for best burn time, that minimizes some error?
  - Like delta semi-major axis divided by target semi-major axis (= target radius) + absolute eccentricity.
  
- Thrust profile?
  - Start with just constant % thrust per stage, or linearly interpolate between input thrust values.
  
- General clean up and review.

# Gravity_turn.py

- Recompute again after burn starts, maybe lowering target altitude threshold a little.

## Multi-run / FlightSession follow-ups

- If a second reader thread is ever introduced anywhere in this codebase,
  remember the lock-ordering deadlock risk from holding multiple streams'
  `.condition`s at once (PLAN.md §2.4, confirmed empirically both ways).
  Acquire them in one consistent order (e.g. sorted by `id()`) rather than
  however each caller happens to list them, or route multi-condition reads
  through a single gatekeeper lock.

- Group the ascent-loop telemetry streams (`ut`, `altitude`, `apoapsis`,
  `periapsis`, `eccentricity`, `speed`) with `types.SimpleNamespace` for
  readability
