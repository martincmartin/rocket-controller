"""Unit tests for KSPUtils.KSPStreams.

Covers the __getattr__ error messages -- specifically that "registered but
no value yet" (start() was called, but next() hasn't run since) is
distinguished from "no such stream" (never registered at all).
"""

import pytest

from KSPUtils import KSPStreams


class FakeStream:
    def __init__(self, value: float) -> None:
        self._value = value
        self.removed = False

    def __call__(self) -> float:
        return self._value

    def start(self, block: bool = True) -> None:
        pass

    def remove(self) -> None:
        self.removed = True

    class _Condition:
        def __enter__(self) -> "FakeStream._Condition":
            return self

        def __exit__(self, *_: object) -> None:
            pass

        def wait(self) -> None:
            pass

    condition = _Condition()


class FakeSpaceCenter:
    ut = 0.0


class FakeConn:
    def __init__(self) -> None:
        self.space_center = FakeSpaceCenter()
        self._next_value = 1.0

    def add_stream(self, func, *args, **kwargs) -> FakeStream:
        value = self._next_value
        self._next_value += 1.0
        return FakeStream(value)


@pytest.fixture
def conn() -> FakeConn:
    return FakeConn()


def test_never_registered_raises_no_stream_named(conn):
    ks = KSPStreams(conn)
    with pytest.raises(AttributeError, match="has no stream named 'bogus'"):
        _ = ks.bogus


def test_registered_but_next_not_called_raises_no_value_yet(conn):
    ks = KSPStreams(conn)
    ks.add_stream("altitude", lambda: 1)
    ks.start()

    with pytest.raises(AttributeError, match="registered but has no value yet"):
        _ = ks.altitude


def test_registered_and_next_called_returns_value(conn):
    ks = KSPStreams(conn)
    ks.add_stream("altitude", lambda: 1)
    ks.start()
    ks.next()

    assert ks.altitude is not None


def test_error_message_lists_registered_streams_for_unknown_name(conn):
    ks = KSPStreams(conn)
    ks.add_stream("altitude", lambda: 1)

    with pytest.raises(AttributeError) as exc_info:
        _ = ks.bogus
    assert "ut" in str(exc_info.value)
    assert "altitude" in str(exc_info.value)
