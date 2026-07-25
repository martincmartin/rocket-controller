"""KSP stream management utilities.

KSPStreams centralises kRPC stream lifecycle (creation, start-up, per-tick
atomic snapshotting, and teardown) so callers never need to manage individual
stream condition variables or handle the StreamError retry dance themselves.

Usage pattern
-------------
    ks = KSPStreams(conn)
    ks.add_stream("position", vessel.position, frame)
    ks.add_stream("velocity", vessel.velocity, frame)
    ks.start()          # blocks until every stream has its first value

    while True:
        ks.next()       # blocks until the next physics tick, then snapshots
        pos = ks.position
        vel = ks.velocity

Notes
-----
- ``next()`` must not be called from multiple threads concurrently; it holds
  all stream condition locks simultaneously.
- The ``ut`` stream is created automatically in ``__init__``; callers must not
  call ``add_stream("ut", ...)`` themselves.
- ``add_stream`` with an already-registered name removes the old stream before
  registering the new one, so it is safe to call mid-loop when e.g. the active
  engine changes and thrust-direction streams must be rebuilt.
"""

import contextlib
from collections import namedtuple
from typing import Any

import krpc
import krpc.client
import krpc.error

from sim import RocketSegment


class KSPStreams:
    """Manages a set of kRPC streams, snapshotting them atomically each tick.

    Parameters
    ----------
    conn:
        Live kRPC connection object (``krpc.connect(...)`` return value).
    """

    def __init__(self, conn: krpc.client.Client) -> None:
        self._conn = conn
        # Insertion-ordered dict of name → stream object.
        # ``ut`` is always the first entry.
        self._streams: dict[str, Any] = {}
        # Latest values, updated atomically by next().
        self._values: dict[str, Any] = {}

        # Create the universal-time stream automatically.
        ut_stream = conn.add_stream(getattr, conn.space_center, "ut")
        self._streams["ut"] = ut_stream
        # None until the first next() call; signals "no previous tick seen yet".
        self._prev_ut: float | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_stream(self, name: str, func: Any, *args: Any, **kwargs: Any) -> None:
        """Register a named stream.

        If a stream with *name* already exists it is removed first, so this
        method is safe to call mid-loop to replace a stream (e.g. after an
        engine change).

        Parameters
        ----------
        name:
            Attribute name callers will use to read the stream value after
            ``next()`` (e.g. ``ks.position``).  Must not be ``"ut"``.
        func, args, kwargs:
            Forwarded verbatim to ``conn.add_stream``.
        """
        if name == "ut":
            raise ValueError(
                "Cannot replace the 'ut' stream; it is managed by KSPStreams."
            )
        if name in self._streams:
            self._streams[name].remove()
        self._streams[name] = self._conn.add_stream(func, *args, **kwargs)

    def start(self) -> None:
        """Start all streams and block until every stream has its first value.

        Calls ``stream.start(False)`` on every registered stream (including
        ``ut``).  ``start(False)`` is idempotent, so streams that were already
        started (e.g. because they were registered in a previous call to
        ``start()``) are unaffected.

        Then loops until ``stream()`` can be called on every stream without
        raising a ``StreamError`` (the error kRPC raises when a stream has not
        yet received its first value from the server).
        """
        for stream in self._streams.values():
            stream.start(False)
        while True:
            try:
                for stream in self._streams.values():
                    stream()
                break
            except krpc.error.StreamError:
                pass

    def next(self) -> None:
        """Block until the next physics tick, then snapshot all stream values.

        After this returns, every registered stream's latest value is available
        as an attribute: ``ks.<name>``.

        Must not be called from multiple threads concurrently.
        """
        # Wait until ut advances past the value we last saw.  Check inside
        # the condition lock and loop to handle spurious wakeups, and to
        # avoid the race where ut advances between the check and the wait().
        ut_stream = self._streams["ut"]
        with ut_stream.condition:
            while ut_stream() == self._prev_ut:
                ut_stream.wait()

        # Atomically snapshot every stream under all their condition locks so
        # all values come from the same physics tick.
        with contextlib.ExitStack() as stack:
            for stream in self._streams.values():
                stack.enter_context(stream.condition)
            for name, stream in self._streams.items():
                self._values[name] = stream()

        self._prev_ut = self._values["ut"]

    def close(self) -> None:
        """Remove all streams and discard their values.

        Best-effort: a failure removing one stream does not prevent the others
        from being removed.  Safe to call more than once.
        """
        for stream in self._streams.values():
            with contextlib.suppress(Exception):
                stream.remove()
        self._streams.clear()
        self._values.clear()
        self._prev_ut = None

    def __getattr__(self, name: str) -> Any:
        # Guard against accidental recursion during __init__ before _values
        # exists.  Normal attribute lookup (via __getattribute__) handles
        # _streams, _values, _conn, _prev_ut, etc.
        if name.startswith("_"):
            raise AttributeError(name)
        values = object.__getattribute__(self, "_values")
        try:
            return values[name]
        except KeyError:
            pass

        streams = object.__getattribute__(self, "_streams")
        if name in streams:
            raise AttributeError(
                f"Stream {name!r} is registered but has no value yet -- "
                "call KSPStreams.next() at least once after registering it "
                "(start() only guarantees the underlying kRPC stream has a "
                "value server-side; next() is what populates this local "
                "snapshot cache)."
            ) from None
        raise AttributeError(
            f"KSPStreams has no stream named {name!r}. "
            f"Registered streams: {list(streams)}"
        ) from None


# ***************   Engine Groups And Segments   ***********

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


class EngineGroup:
    """Snapshot of an engine group's performance and remaining fuel."""

    __slots__ = ("flow_rate", "fuel_duration", "name", "thrust")

    def __init__(
        self, name: str, thrust: float, flow_rate: float, fuel_duration: float
    ) -> None:
        self.name = name  # representative engine's part.title
        self.thrust = thrust  # N
        self.flow_rate = flow_rate  # kg/s
        self.fuel_duration = fuel_duration  # seconds until limiting propellant depletes


def _engine_group_stats(engines: list[Any]) -> EngineGroup | None:
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


def _discover_engine_groups(
    vessel: Any, active_only: bool = True, stage_filter: int | None = None
) -> list[EngineGroup]:
    """Group engines by (decouple_stage, propellant types).

    Args:
        vessel:       kRPC vessel object.
        active_only:  If True only include engines that are active with fuel.
        stage_filter: If set only include engines whose *part.stage* matches.

    Returns a list of engine-group dicts (see ``_engine_group_stats``).
    """
    by_key: dict[tuple[int, frozenset[str]], list[Any]] = {}
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
    for eng_list in by_key.values():
        stats = _engine_group_stats(eng_list)
        if stats is not None:
            groups.append(stats)
    return groups


def build_segments(vessel: Any) -> list[RocketSegment]:
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
            print("!!! WTF????")
            break
        m -= mass_consumed

        groups = remaining_after
        for g in groups:
            g.fuel_duration -= min_dur

    return segments
