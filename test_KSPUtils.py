"""Unit tests for KSPUtils.KSPStreams.

Covers the __getattr__ error messages -- specifically that "registered but
no value yet" (start() was called, but next() hasn't run since) is
distinguished from "no such stream" (never registered at all).
"""

from typing import Any, cast

import krpc.client
import pytest

from KSPUtils import G0, KSPStreams, _engine_group_stats, build_segments


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

    def add_stream(self, func: Any, *args: Any, **kwargs: Any) -> FakeStream:
        value = self._next_value
        self._next_value += 1.0
        return FakeStream(value)


@pytest.fixture
def conn() -> FakeConn:
    return FakeConn()


def test_never_registered_raises_no_stream_named(conn: FakeConn) -> None:
    ks = KSPStreams(cast(krpc.client.Client, conn))
    with pytest.raises(AttributeError, match="has no stream named 'bogus'"):
        _ = ks.bogus


def test_registered_but_next_not_called_raises_no_value_yet(conn: FakeConn) -> None:
    ks = KSPStreams(cast(krpc.client.Client, conn))
    ks.add_stream("altitude", lambda: 1)
    ks.start()

    with pytest.raises(AttributeError, match="registered but has no value yet"):
        _ = ks.altitude


def test_registered_and_next_called_returns_value(conn: FakeConn) -> None:
    ks = KSPStreams(cast(krpc.client.Client, conn))
    ks.add_stream("altitude", lambda: 1)
    ks.start()
    ks.next()

    assert ks.altitude is not None


def test_error_message_lists_registered_streams_for_unknown_name(
    conn: FakeConn,
) -> None:
    ks = KSPStreams(cast(krpc.client.Client, conn))
    ks.add_stream("altitude", lambda: 1)

    with pytest.raises(AttributeError) as exc_info:
        _ = ks.bogus
    assert "ut" in str(exc_info.value)
    assert "altitude" in str(exc_info.value)


# ***************   Engine groups and segments   ***********

# Each fake has just enough of the kRPC API surface that
# _engine_group_stats()/build_segments() touch.


class FakeResources:
    def __init__(self, names: set[str]) -> None:
        self._names = names

    def has_resource(self, name: str) -> bool:
        return name in self._names


class FakePart:
    def __init__(
        self,
        title: str,
        resources: set[str],
        *,
        stage: int = 0,
        decouple_stage: int = -1,
        dry_mass: float = 0.0,
    ) -> None:
        self.title = title
        self.resources = FakeResources(resources)
        self.stage = stage
        self.decouple_stage = decouple_stage
        self.dry_mass = dry_mass


class FakePropellant:
    def __init__(self, name: str, ratio: float, available: float) -> None:
        self.name = name
        self.ratio = ratio
        self.total_resource_available = available


class FakeEngine:
    def __init__(
        self,
        part: FakePart,
        thrust: float,
        isp: float,
        propellants: list[FakePropellant],
    ) -> None:
        self.part = part
        self.max_vacuum_thrust = thrust
        self.max_thrust = thrust
        self.vacuum_specific_impulse = isp
        self.propellants = propellants
        self.active = False
        self.has_fuel = True


class FakeControl:
    def __init__(self, current_stage: int) -> None:
        self.current_stage = current_stage


class FakeVesselParts:
    def __init__(self, parts: list[FakePart], engines: list[FakeEngine]) -> None:
        self.all = parts
        self.engines = engines


class FakeVessel:
    def __init__(
        self,
        mass: float,
        current_stage: int,
        parts: list[FakePart],
        engines: list[FakeEngine],
    ) -> None:
        self.mass = mass
        self.control = FakeControl(current_stage)
        self.parts = FakeVesselParts(parts, engines)


def _swivel_part() -> FakePart:
    return FakePart('LV-T45 "Swivel" Liquid Fuel Engine', set())


def _lf_engine(part: FakePart, available_lf: float) -> FakeEngine:
    return FakeEngine(
        part,
        thrust=215000.0,
        isp=320.0,
        propellants=[
            FakePropellant("LiquidFuel", 0.9, available_lf),
            FakePropellant("Oxidizer", 1.1, available_lf * 11 / 9),
        ],
    )


def test_single_engine_group_duration_from_own_tank() -> None:
    # One liquid engine on its own tank: 540 LF + 660 Ox, flow 68.51 kg/s.
    group = _engine_group_stats([_lf_engine(_swivel_part(), 540.0)])
    expected = 540.0 / (215000.0 / (320.0 * G0) / (0.9 * 5.0 + 1.1 * 5.0) * 0.9)
    assert group is not None
    assert group.fuel_duration == pytest.approx(expected, rel=1e-9)


def test_shared_tank_group_not_double_counted() -> None:
    # Two engines drawing from the same external tank: the group flow is
    # twice the single-engine flow, so the shared tank drains twice as fast.
    # The representative engine reports the whole shared tank, which must
    # NOT be summed over engines (that would double the burn time).
    group = _engine_group_stats(
        [_lf_engine(_swivel_part(), 540.0), _lf_engine(_swivel_part(), 540.0)]
    )
    single = _engine_group_stats([_lf_engine(_swivel_part(), 540.0)])
    assert group is not None
    assert single is not None
    assert group.fuel_duration == pytest.approx(single.fuel_duration / 2, rel=1e-9)


def test_srb_cluster_sums_own_tanks() -> None:
    # Two solid boosters, each with its own internal 820-unit tank. Each
    # engine reports only its own tank, so the group must burn 2x820 units
    # with the combined flow -- i.e. the same per-engine burn time. Before
    # the fix this was half the real duration, leaving the other tank's
    # mass unconsumed in build_segments' accounting.
    def thumper() -> FakeEngine:
        return FakeEngine(
            FakePart('BACC "Thumper" Solid Fuel Booster', {"SolidFuel"}),
            thrust=300000.0,
            isp=210.0,
            propellants=[FakePropellant("SolidFuel", 1.0, 820.0)],
        )

    group = _engine_group_stats([thumper(), thumper()])
    expected = 820.0 * 7.5 / (300000.0 / (210.0 * G0))
    assert group is not None
    assert group.fuel_duration == pytest.approx(expected, rel=1e-9)


def test_build_segments_srb_cluster_mass_accounting() -> None:
    # Regression test for the reported bug: a 2-stage rocket with a pair of
    # solid boosters under a liquid sustainer, on the pad. The last
    # segment (Terrier stage) has true initial mass 4450 kg, but before the
    # fix build_segments reported 10600 kg -- the unburned booster fuel was
    # never subtracted because each booster's own tank was under-counted.
    vessel = FakeVessel(
        mass=28140.0,
        current_stage=5,
        parts=[
            FakePart("Boosters", {"SolidFuel"}, dry_mass=5390.0, decouple_stage=3),
            FakePart("Terrier stage", set(), dry_mass=790.0, decouple_stage=1),
            FakePart("Payload", set(), dry_mass=1420.0, decouple_stage=-1),
        ],
        engines=[
            FakeEngine(
                FakePart(
                    'LV-909 "Terrier" Liquid Fuel Engine',
                    set(),
                    stage=2,
                    decouple_stage=1,
                ),
                thrust=60000.0,
                isp=345.0,
                propellants=[
                    FakePropellant("LiquidFuel", 0.9, 180.0),
                    FakePropellant("Oxidizer", 1.1, 220.0),
                ],
            ),
            _lf_engine(
                FakePart("Swivel part", set(), stage=4, decouple_stage=3), 540.0
            ),
            FakeEngine(
                FakePart(
                    'BACC "Thumper" Solid Fuel Booster',
                    {"SolidFuel"},
                    stage=4,
                    decouple_stage=3,
                ),
                thrust=300000.0,
                isp=210.0,
                propellants=[FakePropellant("SolidFuel", 1.0, 820.0)],
            ),
            FakeEngine(
                FakePart(
                    'BACC "Thumper" Solid Fuel Booster',
                    {"SolidFuel"},
                    stage=4,
                    decouple_stage=3,
                ),
                thrust=300000.0,
                isp=210.0,
                propellants=[FakePropellant("SolidFuel", 1.0, 820.0)],
            ),
        ],
    )

    segments = build_segments(vessel)

    assert len(segments) == 3
    # Booster+sustainer burn lasts until the boosters run dry: 2x820 units
    # of SolidFuel, each tank drained by its own booster.
    assert segments[0].max_burn_time == pytest.approx(
        820.0 * 7.5 / (300000.0 / (210.0 * G0)), rel=1e-6
    )
    assert segments[0].initial_mass == pytest.approx(28140.0)
    # Sustainer continues until its tank is dry.
    assert segments[1].initial_mass == pytest.approx(
        28140.0 - 815000.0 / (2264.771693864642) * segments[0].max_burn_time,
        rel=1e-6,
    )
    # Terrier stage: full mass minus all propellant minus the dry boosters.
    assert segments[2].initial_mass == pytest.approx(4450.0, rel=1e-3)
    assert segments[2].max_burn_time == pytest.approx(
        (180.0 + 220.0) * 5.0 / (60000.0 / (345.0 * G0)), rel=1e-6
    )
    assert segments[0].last_segment_of_stage is False
    assert segments[1].last_segment_of_stage is True
    assert segments[2].last_segment_of_stage is True
