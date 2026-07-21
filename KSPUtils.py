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
from typing import Any


class KSPStreams:
    """Manages a set of kRPC streams, snapshotting them atomically each tick.

    Parameters
    ----------
    conn:
        Live kRPC connection object (``krpc.connect(...)`` return value).
    """

    def __init__(self, conn: Any) -> None:
        self._conn = conn
        # Insertion-ordered dict of name → stream object.
        # ``ut`` is always the first entry.
        self._streams: dict[str, Any] = {}
        # Latest values, updated atomically by next().
        self._values: dict[str, Any] = {}

        # Create the universal-time stream automatically.
        ut_stream = conn.add_stream(getattr, conn.space_center, "ut")
        self._streams["ut"] = ut_stream
        # Read the initial ut value; used by next() to detect a new tick.
        self._prev_ut: float | None = ut_stream()

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
            with contextlib.suppress(Exception):
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
            except Exception:
                pass

    def next(self) -> None:
        """Block until the next physics tick, then snapshot all stream values.

        After this returns, every registered stream's latest value is available
        as an attribute: ``ks.<name>``.

        Must not be called from multiple threads concurrently.
        """
        ut_stream = self._streams["ut"]

        # Wait until ut advances past the value we last saw.
        if self._prev_ut is None or ut_stream() == self._prev_ut:
            with ut_stream.condition:
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
        try:
            values = object.__getattribute__(self, "_values")
            return values[name]
        except KeyError:
            streams = object.__getattribute__(self, "_streams")
            raise AttributeError(
                f"KSPStreams has no stream named {name!r}. "
                f"Registered streams: {list(streams)}"
            ) from None
