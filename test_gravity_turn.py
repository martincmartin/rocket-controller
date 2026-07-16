"""Unit tests for gravity_turn.py.

Covers `FlightSession`'s cleanup/lifecycle guarantees, using a small fake
kRPC connection -- no live KSP/kRPC server needed.
"""

import pytest

from gravity_turn import FlightSession


# ─── FlightSession ────────────────────────────────────────────────────────────
#
# A minimal fake kRPC connection, just enough to exercise FlightSession's
# lifecycle/cleanup logic without a live KSP/kRPC server.


class FakeStream:
    def __init__(self, name: str, removed_log: list[str]) -> None:
        self.name = name
        self._removed_log = removed_log
        self.removed = False
        self.fail_on_remove = False

    def __call__(self) -> float:
        return 0.0

    def remove(self) -> None:
        if self.fail_on_remove:
            raise RuntimeError(f"boom removing {self.name}")
        self.removed = True
        self._removed_log.append(self.name)


class FakeAutoPilot:
    def __init__(self) -> None:
        self.disengaged = False

    def disengage(self) -> None:
        self.disengaged = True


class FakeControl:
    def __init__(self) -> None:
        self.throttle = 1.0


class FakeVessel:
    def __init__(self) -> None:
        self.situation = "pre_launch"
        self.control = FakeControl()
        self.auto_pilot = FakeAutoPilot()


class FakeVesselSituation:
    pre_launch = "pre_launch"


class FakeSpaceCenter:
    def __init__(self, vessel: FakeVessel) -> None:
        self.active_vessel = vessel
        self.physics_warp_factor = 1
        self.VesselSituation = FakeVesselSituation


class FakeGameScene:
    flight = "flight"


class FakeKrpc:
    def __init__(self) -> None:
        self.GameScene = FakeGameScene
        self.current_game_scene = "flight"


class FakeConn:
    """Just enough of krpc.client.Client for FlightSession."""

    def __init__(self) -> None:
        self.krpc = FakeKrpc()
        self.space_center = FakeSpaceCenter(FakeVessel())
        self.removed_stream_names: list[str] = []
        self._next_id = 0

    def add_stream(self, func, *args, **kwargs) -> FakeStream:
        self._next_id += 1
        return FakeStream(f"stream-{self._next_id}", self.removed_stream_names)


@pytest.fixture
def conn() -> FakeConn:
    return FakeConn()


def test_enter_waits_for_prelaunch_and_returns_self(conn):
    with FlightSession(conn) as fs:
        assert fs.vessel is conn.space_center.active_vessel


def test_normal_exit_removes_every_stream(conn):
    with FlightSession(conn) as fs:
        s1 = fs.add_stream(lambda: 1)
        s2 = fs.add_stream(lambda: 2)

    assert s1.removed
    assert s2.removed
    assert conn.removed_stream_names == ["stream-1", "stream-2"]


def test_normal_exit_resets_throttle_autopilot_and_warp(conn):
    vessel = conn.space_center.active_vessel
    vessel.control.throttle = 1.0
    conn.space_center.physics_warp_factor = 3

    with FlightSession(conn):
        pass

    assert vessel.control.throttle == 0.0
    assert vessel.auto_pilot.disengaged
    assert conn.space_center.physics_warp_factor == 0


def test_exceptional_exit_still_cleans_up_and_propagates(conn):
    with pytest.raises(ValueError, match="boom"):
        with FlightSession(conn) as fs:
            s1 = fs.add_stream(lambda: 1)
            raise ValueError("boom")

    assert s1.removed


def test_vessel_raises_before_enter_and_after_exit(conn):
    fs = FlightSession(conn)

    with pytest.raises(RuntimeError):
        _ = fs.vessel

    with fs:
        _ = fs.vessel  # does not raise while entered

    with pytest.raises(RuntimeError):
        _ = fs.vessel


def test_add_stream_raises_before_enter_and_after_exit(conn):
    fs = FlightSession(conn)

    with pytest.raises(RuntimeError):
        fs.add_stream(lambda: 1)

    with fs:
        fs.add_stream(lambda: 1)  # does not raise while entered

    with pytest.raises(RuntimeError):
        fs.add_stream(lambda: 1)


def test_one_streams_remove_failure_does_not_block_the_others(conn):
    with FlightSession(conn) as fs:
        s1 = fs.add_stream(lambda: 1)
        s2 = fs.add_stream(lambda: 2)
        s1.fail_on_remove = True

    assert not s1.removed
    assert s2.removed


def test_close_is_idempotent(conn):
    fs = FlightSession(conn)
    with fs:
        fs.add_stream(lambda: 1)

    fs.close()  # second close() should be a no-op, not raise
    assert conn.removed_stream_names == ["stream-1"]
