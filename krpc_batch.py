"""Batch multiple kRPC procedure calls into a single request/response round
trip. See PLAN.md ("Fix high-latency vessel.control.pitch/roll/yaw writes")
for why this exists: kRPC's public API (Client._invoke) sends exactly one
ProcedureCall per Request, even though the wire protocol supports many;
issuing pitch/roll/yaw as three separate round trips triples the exposure
to (a) a per-tick server-side scheduling floor and (b) GIL contention with
any concurrent CPU-bound Python thread (e.g. gravity_turn.py's replanning).

Depends on krpc.client.Client private attributes not part of its public
API contract: `_build_call`, `_rpc_connection`, `_rpc_connection_lock`,
`_types`, `_build_error`. Verified against krpc==0.5.4 (pinned in
pyproject.toml -- re-verify this module against krpc's CHANGELOG/source
before upgrading that pin).
"""

from collections.abc import Callable
from typing import Any

import krpc.client
import krpc.schema.KRPC_pb2 as KRPC_pb2

ControlSender = Callable[[float, float, float], None]


def send_batch(conn: krpc.client.Client, calls: list[Any]) -> list[Any]:
    """Send several ProcedureCalls in one request; return their results in
    order. Raises (via conn._build_error, the same exception types a normal
    single-call _invoke() would raise) on a request-level error.

    IMPORTANT (empirically confirmed against a live kRPC 0.5.4 server): if
    any call in the batch errors server-side, the server stops executing
    the request right there -- calls *after* the failed one are never
    attempted, but still come back as blank/no-error result entries, not
    as further errors. Callers must not treat "this result has no error"
    as "this call was applied" without also confirming no *earlier* result
    in the same batch had an error. See callers below for the pattern.
    """
    request = KRPC_pb2.Request()
    request.calls.extend(calls)
    with conn._rpc_connection_lock:
        conn._rpc_connection.send_message(request)
        response = conn._rpc_connection.receive_message(KRPC_pb2.Response)
    if response.HasField("error"):
        raise conn._build_error(response.error)
    return list(response.results)


def make_batched_control_sender(
    conn: krpc.client.Client, control: Any
) -> ControlSender:
    """Returns a function(pitch, roll, yaw) that applies all three via one
    round trip instead of three (Control_set_Pitch/Roll/Yaw)."""
    control_type = conn._types.class_type("SpaceCenter", "Control")
    float_type = conn._types.float_type

    def build(procedure: str, value: float) -> Any:
        return conn._build_call(
            "SpaceCenter",
            procedure,
            [control, value],
            ["self", "value"],
            [control_type, float_type],
            None,
        )

    def set_all(pitch: float, roll: float, yaw: float) -> None:
        results = send_batch(
            conn,
            [
                build("Control_set_Pitch", pitch),
                build("Control_set_Roll", roll),
                build("Control_set_Yaw", yaw),
            ],
        )
        # Stop at the first error -- per the module docstring, anything
        # after it was never applied regardless of its own error field.
        for result in results:
            if result.HasField("error"):
                raise conn._build_error(result.error)

    return set_all
