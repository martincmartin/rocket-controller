"""Unit tests for autopilot_thread.py.

Uses a fake kRPC connection/vessel -- no live KSP server needed. The fake
connection's ``ut`` stream only advances when the test explicitly calls
``tick()``, which makes the worker thread's progress fully
deterministic/controllable from the test.
"""

import threading
import time
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from autopilot_thread import AutopilotWorker
from guidance_link import GuidanceCommand
from sim import OrbitalPlane

PLANE = OrbitalPlane(np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))


# ─── Fakes ─────────────────────────────────────────────────────────────────


class _FakeStream:
    """Minimal fake of a kRPC Stream: a value-fetching callable plus a real
    ``threading.Condition`` (only the ``ut`` stream is ever actually waited
    on; the rest just need to support being entered as a context manager by
    KSPStreams.next()'s ExitStack)."""

    def __init__(self, value_fn: Callable[[], Any]) -> None:
        self._value_fn = value_fn
        self.condition = threading.Condition()
        self.started = False

    def start(self, wait: bool = True) -> None:
        self.started = True

    def __call__(self) -> Any:
        return self._value_fn()

    def wait(self, timeout: float | None = None) -> None:
        self.condition.wait(timeout=timeout)

    def remove(self) -> None:
        pass


class FakeControl:
    def __init__(self) -> None:
        self.pitch = 0.0
        self.roll = 0.0
        self.yaw = 0.0


class FakeVessel:
    def __init__(self) -> None:
        self.control = FakeControl()
        self.available_torque = ((10.0, 10.0, 10.0), (-10.0, -10.0, -10.0))
        self.inertia_tensor = (2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0)
        frame = object()
        self.orbit = SimpleNamespace(
            body=SimpleNamespace(non_rotating_reference_frame=frame)
        )

    def rotation(self, frame: Any) -> tuple[float, float, float, float]:
        return (0.0, 0.0, 0.0, 1.0)

    def angular_velocity(self, frame: Any) -> tuple[float, float, float]:
        return (0.0, 0.0, 0.0)


class FakeConnection:
    """Fake kRPC ``Client``: supports exactly what KSPStreams/AutopilotWorker
    need (``space_center.active_vessel``, ``add_stream``, ``close``)."""

    def __init__(self) -> None:
        self.vessel = FakeVessel()
        self.space_center = SimpleNamespace(active_vessel=self.vessel)
        self._ut = 0.0
        self.ut_stream = _FakeStream(lambda: self._ut)
        self.closed = False

    def add_stream(self, func: Any, *args: Any, **kwargs: Any) -> _FakeStream:
        if func is getattr and args[0] is self.space_center and args[1] == "ut":
            return self.ut_stream
        if func is getattr:
            obj, name = args[0], args[1]
            return _FakeStream(lambda o=obj, n=name: getattr(o, n))
        return _FakeStream(lambda f=func, a=args, kw=kwargs: f(*a, **kw))

    def tick(self, dt: float = 0.02) -> None:
        with self.ut_stream.condition:
            self._ut += dt
            self.ut_stream.condition.notify_all()

    def close(self) -> None:
        self.closed = True


def _make_worker() -> tuple[AutopilotWorker, FakeConnection]:
    conn = FakeConnection()

    def fake_connect(**kwargs: Any) -> FakeConnection:
        return conn

    worker = AutopilotWorker(connect=fake_connect)
    return worker, conn


def _wait_until(predicate: Callable[[], bool], timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("condition not met within timeout")


# ─── Tests ─────────────────────────────────────────────────────────────────


class TestLifecycle:
    def test_start_returns_after_first_tick_without_hanging(self) -> None:
        worker, _conn = _make_worker()
        worker.start(ready_timeout=2.0)
        try:
            assert worker.error is None
        finally:
            worker.stop()

    def test_stop_joins_thread_and_closes_connection(self) -> None:
        worker, conn = _make_worker()
        worker.start()
        worker.stop()
        assert conn.closed is True

    def test_stop_is_safe_to_call_more_than_once(self) -> None:
        worker, _conn = _make_worker()
        worker.start()
        worker.stop()
        worker.stop()  # must not raise

    def test_disabled_by_default_never_writes_controls(self) -> None:
        worker, conn = _make_worker()
        worker.start()
        try:
            conn.vessel.control.pitch = 42.0
            conn.vessel.control.roll = 42.0
            conn.vessel.control.yaw = 42.0
            for _ in range(5):
                conn.tick()
            time.sleep(0.05)
            assert conn.vessel.control.pitch == 42.0
            assert conn.vessel.control.roll == 42.0
            assert conn.vessel.control.yaw == 42.0
        finally:
            worker.stop()


class TestEnableDisable:
    def test_enabling_starts_writing_controls(self) -> None:
        worker, conn = _make_worker()
        worker.start()
        try:
            worker.link.set(
                GuidanceCommand(
                    ref_angle=0.0,
                    a_coeff=0.0,
                    b_coeff=0.0,
                    t0=-1000.0,
                    plane=PLANE,
                )
            )
            conn.vessel.control.pitch = 999.0
            for _ in range(3):
                conn.tick()
            _wait_until(lambda: conn.vessel.control.pitch != 999.0)
        finally:
            worker.stop()

    def test_falling_edge_zeros_once_then_leaves_controls_alone(self) -> None:
        worker, conn = _make_worker()
        worker.start()
        try:
            worker.link.set(
                GuidanceCommand(
                    ref_angle=0.0,
                    a_coeff=0.0,
                    b_coeff=0.0,
                    t0=-1000.0,
                    plane=PLANE,
                )
            )
            # With this plane/ref_angle, the target direction has no
            # z-component, so `pitch` legitimately stays 0 -- use `yaw`
            # (guaranteed nonzero here) to confirm the first enabled tick
            # actually ran before proceeding.
            conn.vessel.control.yaw = 999.0
            conn.tick()
            _wait_until(lambda: conn.vessel.control.yaw != 999.0)

            worker.link.set(None)
            conn.vessel.control.pitch = 7.0
            conn.vessel.control.roll = 7.0
            conn.vessel.control.yaw = 7.0
            conn.tick()  # falling-edge tick: should zero exactly once
            _wait_until(lambda: conn.vessel.control.yaw == 0.0)
            assert conn.vessel.control.pitch == 0.0
            assert conn.vessel.control.roll == 0.0

            # Subsequent disabled ticks must not touch controls at all.
            conn.vessel.control.pitch = 42.0
            conn.vessel.control.roll = 42.0
            conn.vessel.control.yaw = 42.0
            for _ in range(5):
                conn.tick()
            time.sleep(0.05)
            assert conn.vessel.control.pitch == 42.0
            assert conn.vessel.control.roll == 42.0
            assert conn.vessel.control.yaw == 42.0
        finally:
            worker.stop()

    def test_republishing_command_resumes_commanding(self) -> None:
        worker, conn = _make_worker()
        worker.start()
        try:
            worker.link.set(None)
            conn.vessel.control.pitch = 13.0
            conn.tick()
            time.sleep(0.02)
            assert conn.vessel.control.pitch == 13.0  # untouched while disabled

            worker.link.set(
                GuidanceCommand(
                    ref_angle=0.0,
                    a_coeff=0.0,
                    b_coeff=0.0,
                    t0=-1000.0,
                    plane=PLANE,
                )
            )
            for _ in range(3):
                conn.tick()
            _wait_until(lambda: conn.vessel.control.pitch != 13.0)
        finally:
            worker.stop()


class TestErrorPropagation:
    def test_unhandled_exception_sets_error_and_stops_thread(self) -> None:
        worker, conn = _make_worker()
        worker.start()
        try:
            # An invalid `plane` (None) makes evaluate_target() raise
            # AttributeError when it calls cmd.plane.from_plane(...).
            worker.link.set(
                GuidanceCommand(
                    ref_angle=0.0,
                    a_coeff=0.0,
                    b_coeff=0.0,
                    t0=-1000.0,
                    plane=None,  # type: ignore[arg-type]
                )
            )
            conn.tick()
            _wait_until(lambda: worker.error is not None)
            assert isinstance(worker.error, AttributeError)
            _wait_until(lambda: worker._thread is None or not worker._thread.is_alive())
        finally:
            worker.stop()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
