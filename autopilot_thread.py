"""Runs CustomAutopilot on its own thread, over its own kRPC connection.

The attitude control law (CustomAutopilot.update()) must run every physics tick, but the
guidance-law replanning that produces its target (find_linear_ tangent_params in sim.py)
can take 50ms+ -- multiple physics ticks. Running the autopilot on its own thread, with
its own kRPC connection (so its control-writing RPCs are never queued up behind whatever
the main thread's connection happens to be doing), decouples the two.

Communication between the main (planning) thread and this worker thread is via a
GuidanceLink (guidance_link.py): the main thread publishes a fresh GuidanceCommand
whenever it finishes replanning (or wants to enable/disable attitude commanding,
e.g. around a staging event); the worker thread reads the latest command every physics
tick and evaluates it with evaluate_target().

"""

import threading
from collections.abc import Callable
from typing import Any

import krpc
import krpc.client

from autopilot import CustomAutopilot
from guidance_link import GuidanceCommand, GuidanceLink, evaluate_target
from krpc_batch import ControlSender, make_batched_control_sender
from KSPUtils import KSPStreams

DEFAULT_NAME = "Gravity Turn (Autopilot)"


class AutopilotWorker:
    """Owns a dedicated kRPC connection and thread running CustomAutopilot.

    Usage::

        worker = AutopilotWorker()
        worker.start()
        try:
            worker.link.set(GuidanceCommand(...))
            ...
            worker.link.set(None)   # e.g. bracketing a staging event
            ...
            worker.link.set(GuidanceCommand(...))
        finally:
            worker.stop()

    or as a context manager::

        with AutopilotWorker() as worker:
            worker.link.set(GuidanceCommand(...))
            ...

    Never touches the caller's ``FlightSession``/``conn``/``vessel`` -- it makes its own
    connection and fetches its own vessel/frame handles.  kRPC RPCs on one connection
    are serialized by a single lock held across the full round trip, so sharing a
    connection risks the control-writing RPCs queuing up behind whatever the main
    thread's connection is doing.

    """

    def __init__(
        self,
        name: str = DEFAULT_NAME,
        connect: Callable[..., krpc.client.Client] = krpc.connect,
        connect_kwargs: dict[str, Any] | None = None,
        control_sender_factory: Callable[
            [Any, Any], ControlSender
        ] = make_batched_control_sender,
    ) -> None:
        self._name = name
        self._connect = connect
        self._connect_kwargs = connect_kwargs or {}
        self._control_sender_factory = control_sender_factory

        self._conn: krpc.client.Client | None = None
        self._vessel: Any | None = None
        self._streams: KSPStreams | None = None
        self._autopilot: CustomAutopilot | None = None

        self._link = GuidanceLink()
        self._stop_event = threading.Event()
        self._started_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None

    @property
    def link(self) -> GuidanceLink:
        return self._link

    @property
    def error(self) -> BaseException | None:
        """Set if the worker thread died from an unhandled exception. The
        main loop must poll this once per iteration while the worker is
        expected to be running, and abort (see PLAN.md §8.3) if it's set."""
        return self._error

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, ready_timeout: float = 10.0) -> None:
        """Connect, build the autopilot, and spawn the worker thread.

        Blocks until the thread has processed at least one physics tick (so
        the caller can safely disengage vessel.auto_pilot/SAS immediately
        after this returns without a gap where nothing is commanding
        attitude), or ``ready_timeout`` seconds elapse.
        """
        conn = self._connect(name=self._name, **self._connect_kwargs)
        self._conn = conn

        assert conn.space_center is not None
        vessel = conn.space_center.active_vessel
        self._vessel = vessel
        frame = vessel.orbit.body.non_rotating_reference_frame

        streams = KSPStreams(conn)
        self._streams = streams
        send_controls = self._control_sender_factory(conn, vessel.control)
        self._autopilot = CustomAutopilot(
            streams,
            vessel,
            frame,
            send_controls=send_controls,
        )
        streams.start()

        self._stop_event.clear()
        self._started_event.clear()
        self._error = None
        self._thread = threading.Thread(
            target=self._run, name="AutopilotWorker", daemon=True
        )
        self._thread.start()

        if not self._started_event.wait(timeout=ready_timeout):
            raise TimeoutError(
                "AutopilotWorker thread did not process its first physics "
                f"tick within {ready_timeout}s"
            )

    def stop(self, timeout: float = 2.0) -> None:
        """Signal the thread to stop, join with a bounded timeout, zero
        pitch/roll/yaw best-effort, then close the connection/streams.

        Never hangs the caller: if the thread doesn't stop within
        ``timeout``, this logs a warning and proceeds with cleanup anyway.
        Safe to call more than once.
        """
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                print(f"  ! AutopilotWorker thread did not stop within {timeout}s")
            self._thread = None

        self._zero_controls_best_effort()

        if self._streams is not None:
            try:
                self._streams.close()
            except Exception as e:
                print(f"  ! AutopilotWorker cleanup warning: {e}")

        if self._conn is not None:
            try:
                self._conn.close()
            except Exception as e:
                print(f"  ! AutopilotWorker cleanup warning: {e}")
            self._conn = None

        self._vessel = None
        self._streams = None
        self._autopilot = None

    def __enter__(self) -> "AutopilotWorker":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Worker thread body
    # ------------------------------------------------------------------

    def _run(self) -> None:
        assert self._streams is not None
        assert self._autopilot is not None
        streams = self._streams
        autopilot = self._autopilot

        was_enabled = False
        try:
            while not self._stop_event.is_set():
                streams.next()
                self._started_event.set()

                match self._link.get():
                    case GuidanceCommand() as cmd:
                        target_dir, target_dir_dot = evaluate_target(cmd, streams.ut)
                        autopilot.update(target_dir, target_dir_dot)
                        was_enabled = True
                    case None:
                        if was_enabled:
                            # Falling edge: zero controls exactly once, then
                            # leave them alone on every subsequent disabled
                            # tick -- see PLAN.md §6, "Idle behavior".
                            # Something else (the stock vessel.auto_pilot on
                            # the main thread, or a future scheme) may be
                            # commanding attitude while this worker is
                            # disabled, and writing every tick -- even
                            # zeros -- would fight it.
                            self._zero_controls()
                            was_enabled = False
        except Exception as e:
            self._error = e
            self._zero_controls_best_effort()

    def _zero_controls(self) -> None:
        assert self._vessel is not None
        self._vessel.control.pitch = 0.0
        self._vessel.control.roll = 0.0
        self._vessel.control.yaw = 0.0

    def _zero_controls_best_effort(self) -> None:
        try:
            self._zero_controls()
        except Exception as e:
            print(f"  ! AutopilotWorker failed to zero controls after error: {e}")
