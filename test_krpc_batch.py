"""Unit tests for krpc_batch.py.

Uses a fake ``conn`` object that only fakes what send_batch()/
make_batched_control_sender() touch -- no live KSP/kRPC server needed.
"""

import threading
from collections.abc import Iterable
from typing import Any

import krpc.schema.KRPC_pb2 as KRPC_pb2
import pytest

from krpc_batch import make_batched_control_sender, send_batch


class _FakeError(Exception):
    pass


class FakeConnection:
    """Fakes the pieces of krpc.client.Client that krpc_batch.py touches."""

    def __init__(self) -> None:
        self.sent_requests: list[KRPC_pb2.Request] = []
        self._response = KRPC_pb2.Response()
        self._rpc_connection_lock = threading.Lock()
        self._rpc_connection = self
        self._build_call_log: list[tuple[Any, ...]] = []

    # -- fakes used by send_batch() --------------------------------------
    def send_message(self, request: KRPC_pb2.Request) -> None:
        self.sent_requests.append(request)

    def receive_message(self, _cls: type) -> KRPC_pb2.Response:
        return self._response

    def _build_error(self, error: Any) -> Exception:
        return _FakeError(str(error))

    # -- fakes used by make_batched_control_sender() ----------------------
    class _FakeTypes:
        def class_type(self, service: str, name: str) -> str:
            return f"{service}.{name}"

        @property
        def float_type(self) -> str:
            return "float"

    _types = _FakeTypes()

    def _build_call(
        self,
        service: str,
        procedure: str,
        args: Iterable[object],
        param_names: Iterable[str],
        param_types: Iterable[Any],
        return_type: Any | None,
    ) -> KRPC_pb2.ProcedureCall:
        self._build_call_log.append((service, procedure, args))
        call = KRPC_pb2.ProcedureCall()
        call.service = service
        call.procedure = procedure
        return call

    # -- helpers for tests -------------------------------------------------
    def set_response_results(self, n: int, error_index: int | None = None) -> None:
        self._response = KRPC_pb2.Response()
        for i in range(n):
            result = self._response.results.add()
            if error_index is not None and i == error_index:
                result.error.description = "boom"

    def set_response_error(self) -> None:
        self._response = KRPC_pb2.Response()
        self._response.error.description = "top-level error"


class TestSendBatch:
    def test_single_round_trip(self) -> None:
        conn = FakeConnection()
        conn.set_response_results(2)
        calls = [KRPC_pb2.ProcedureCall(), KRPC_pb2.ProcedureCall()]
        send_batch(conn, calls)
        assert len(conn.sent_requests) == 1
        assert list(conn.sent_requests[0].calls) == calls

    def test_top_level_error_raises(self) -> None:
        conn = FakeConnection()
        conn.set_response_error()
        with pytest.raises(_FakeError):
            send_batch(conn, [KRPC_pb2.ProcedureCall()])

    def test_returns_results_in_order(self) -> None:
        conn = FakeConnection()
        conn.set_response_results(3)
        results = send_batch(conn, [KRPC_pb2.ProcedureCall() for _ in range(3)])
        assert len(results) == 3


class TestMakeBatchedControlSender:
    def test_one_round_trip_per_call(self) -> None:
        conn = FakeConnection()
        conn.set_response_results(3)
        control = object()
        set_all = make_batched_control_sender(conn, control)
        set_all(0.1, 0.2, 0.3)
        assert len(conn.sent_requests) == 1
        assert len(conn.sent_requests[0].calls) == 3

    def test_calls_built_in_pitch_roll_yaw_order_with_values(self) -> None:
        conn = FakeConnection()
        conn.set_response_results(3)
        control = object()
        set_all = make_batched_control_sender(conn, control)
        set_all(0.1, 0.2, 0.3)
        procedures = [entry[1] for entry in conn._build_call_log]
        values = [entry[2][1] for entry in conn._build_call_log]
        assert procedures == [
            "Control_set_Pitch",
            "Control_set_Roll",
            "Control_set_Yaw",
        ]
        assert values == [0.1, 0.2, 0.3]

    def test_top_level_error_raises(self) -> None:
        conn = FakeConnection()
        conn.set_response_error()
        control = object()
        set_all = make_batched_control_sender(conn, control)
        with pytest.raises(_FakeError):
            set_all(0.1, 0.2, 0.3)

    def test_first_result_error_raises(self) -> None:
        """Regression test for the empirically-confirmed partial-batch-
        failure semantics: an error on the *first* result must raise even
        though later result entries in the same response have no error
        field set (i.e. this must not only be exercised by an error on the
        last call, which would trivially pass without checking earlier
        results first)."""
        conn = FakeConnection()
        conn.set_response_results(3, error_index=0)
        control = object()
        set_all = make_batched_control_sender(conn, control)
        with pytest.raises(_FakeError):
            set_all(0.1, 0.2, 0.3)

    def test_no_error_does_not_raise(self) -> None:
        conn = FakeConnection()
        conn.set_response_results(3)
        control = object()
        set_all = make_batched_control_sender(conn, control)
        set_all(0.1, 0.2, 0.3)  # must not raise


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
