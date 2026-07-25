"""Thread-safe hand-off of linear-tangent guidance parameters.

``GuidanceCommand`` is an immutable snapshot of the linear-tangent steering
law (see ``sim.CircularizationPlan``). ``GuidanceLink`` is a single-slot
mailbox used to publish the latest command from the (main) planning thread
to the (autopilot) control thread; publishing ``None`` means "no attitude
command is currently in effect". ``evaluate_target`` is the pure function
the control thread uses to turn a command plus the current universal time
into a target direction / target direction rate for
``autopilot.CustomAutopilot.update()``.
"""

import math
import threading
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from sim import OrbitalPlane

Vector = NDArray[np.float64]


@dataclass(frozen=True)
class GuidanceCommand:
    """Linear-tangent steering law parameters, valid from ``t0`` onward.

        theta(t) = ref_angle + atan(a_coeff + b_coeff * (t - t0))   for t >= t0
        theta(t) = ref_angle + atan(a_coeff)                        for t <  t0

    ``t0`` is an absolute universal time (``ut``).
    """

    ref_angle: float
    a_coeff: float
    b_coeff: float
    t0: float
    plane: OrbitalPlane


class GuidanceLink:
    """Thread-safe single-slot mailbox for the latest ``GuidanceCommand``.

    The producer publishes the newest command with ``set()``; the consumer (autopilot
    thread) always reads the latest one with ``get()``. This is not a queue -- older,
    superseded commands are simply discarded.

    ``None`` means "no attitude command is currently in effect" (e.g. before the first
    real command is published, or while attitude commanding has been deliberately
    suspended, such as bracketing a staging event).

    ``GuidanceCommand`` is frozen/immutable. The lock exists for clarity/defensiveness,
    not because correctness secretly depends on CPython's GIL.

    """

    def __init__(self, initial: GuidanceCommand | None = None) -> None:
        self._lock = threading.Lock()
        self._command: GuidanceCommand | None = initial

    def set(self, command: GuidanceCommand | None) -> None:
        with self._lock:
            self._command = command

    def get(self) -> GuidanceCommand | None:
        with self._lock:
            return self._command


def evaluate_target(cmd: GuidanceCommand, ut: float) -> tuple[Vector, Vector]:
    """Linear-tangent thrust direction and its rate of change at absolute
    time ``ut``, given guidance parameters ``cmd``.

    For ``ut < cmd.t0`` (not yet burning), returns the fixed t=0 attitude
    with zero rate (hold and wait for ignition). For ``ut >= cmd.t0``,
    tracks theta(t) and its analytic derivative.
    """
    t = max(0.0, ut - cmd.t0)
    tan_val = cmd.a_coeff + cmd.b_coeff * t
    theta = cmd.ref_angle + math.atan(tan_val)
    dir2d = np.array([math.cos(theta), math.sin(theta)])
    thrust_dir = cmd.plane.from_plane(dir2d)

    if ut < cmd.t0:
        thrust_dir_dot = np.zeros(3)
    else:
        # d/dt atan(a + b*t) = b / (1 + (a + b*t)^2)
        dtheta_dt = cmd.b_coeff / (1.0 + tan_val**2)
        thrust_dir_dot = cmd.plane.from_plane(
            np.array([-math.sin(theta) * dtheta_dt, math.cos(theta) * dtheta_dt])
        )
    return thrust_dir, thrust_dir_dot
