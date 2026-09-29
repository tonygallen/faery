import collections.abc
import dataclasses
import threading
import typing

import numpy
import pytest

import faery

WAIT_TIMEOUT = 3.0
SHORT_WAIT_TIMEOUT = 0.1
DIMENSIONS = (4, 3)
TIME_RANGE = (10 * faery.us, 40 * faery.us)
FREQUENCY_HZ = 1_000.0


def events(*rows: tuple[int, int, int, bool]) -> numpy.ndarray:
    result = numpy.zeros(len(rows), dtype=faery.EVENTS_DTYPE)
    for index, row in enumerate(rows):
        result[index] = row
    return result


def event_packets() -> list[numpy.ndarray]:
    return [
        events((10, 0, 0, True), (11, 1, 0, False)),
        events((20, 2, 1, True)),
        events((30, 3, 2, False), (31, 0, 2, True)),
    ]


def frames() -> list[faery.Frame]:
    return [
        faery.Frame(
            t=(index + 1) * faery.ms,
            pixels=numpy.full((3, 4, 4), index, dtype=numpy.uint8),
        )
        for index in range(3)
    ]


class _PacketSource:
    def __init__(self, packets: list[typing.Any]):
        self.packets = packets
        self.iteration_count = 0

    def __iter__(self) -> collections.abc.Iterator[typing.Any]:
        self.iteration_count += 1
        yield from self.packets

    def dimensions(self) -> tuple[int, int]:
        return DIMENSIONS

    def time_range(self) -> tuple[faery.Time, faery.Time]:
        return TIME_RANGE

    def frequency_hz(self) -> float:
        return FREQUENCY_HZ


class _EventsSource(_PacketSource, faery.EventsStream):
    pass


class _FiniteEventsSource(_PacketSource, faery.FiniteEventsStream):
    pass


class _RegularEventsSource(_PacketSource, faery.RegularEventsStream):
    pass


class _FiniteRegularEventsSource(_PacketSource, faery.FiniteRegularEventsStream):
    pass


class _FrameSource(_PacketSource, faery.FrameStream):
    pass


class _FiniteFrameSource(_PacketSource, faery.FiniteFrameStream):
    pass


class _RegularFrameSource(_PacketSource, faery.RegularFrameStream):
    pass


class _FiniteRegularFrameSource(_PacketSource, faery.FiniteRegularFrameStream):
    pass


class _TrackedEventsSource(faery.EventsStream):
    def __init__(self, packets: list[typing.Any]):
        self.packets = packets
        self.iteration_count = 0
        self.started = threading.Event()
        self.pulled = [threading.Event() for _ in packets]
        self.exhausted = threading.Event()
        self.closed = threading.Event()

    def __iter__(self) -> collections.abc.Iterator[typing.Any]:
        self.iteration_count += 1
        self.started.set()
        try:
            for index, packet in enumerate(self.packets):
                self.pulled[index].set()
                yield packet
            self.exhausted.set()
        finally:
            self.closed.set()

    def dimensions(self) -> tuple[int, int]:
        return DIMENSIONS


class _ReusedArrayEventsSource(faery.EventsStream):
    def __init__(self):
        self.packet = events((10, 0, 0, True))
        self.exhausted = threading.Event()

    def __iter__(self) -> collections.abc.Iterator[numpy.ndarray]:
        for value in range(3):
            self.packet["x"][0] = value
            yield self.packet
        self.exhausted.set()

    def dimensions(self) -> tuple[int, int]:
        return DIMENSIONS


class _ReusedPixelsFrameSource(faery.FrameStream):
    def __init__(self):
        self.frame = faery.Frame(
            t=0 * faery.us,
            pixels=numpy.zeros((3, 4, 4), dtype=numpy.uint8),
        )
        self.exhausted = threading.Event()

    def __iter__(self) -> collections.abc.Iterator[faery.Frame]:
        for value in range(3):
            self.frame.t = (value + 1) * faery.ms
            self.frame.pixels.fill(value)
            yield self.frame
        self.exhausted.set()

    def dimensions(self) -> tuple[int, int]:
        return DIMENSIONS


class _CopyFailingPacket:
    def __init__(self, error_type: type[BaseException]):
        self.error_type = error_type

    def copy(self):
        raise self.error_type("retained packet copy failed")


class _UpstreamFailure(Exception):
    pass


class _FailingEventsSource(faery.EventsStream):
    def __init__(self, packet: numpy.ndarray):
        self.packet = packet
        self.raised = threading.Event()

    def __iter__(self) -> collections.abc.Iterator[numpy.ndarray]:
        yield self.packet
        self.raised.set()
        raise _UpstreamFailure("upstream failed")

    def dimensions(self) -> tuple[int, int]:
        return DIMENSIONS


class _CloseableBlockingIterator:
    def __init__(self):
        self.entered = threading.Event()
        self.closed = threading.Event()
        self._release = threading.Event()

    def __iter__(self) -> "_CloseableBlockingIterator":
        return self

    def __next__(self) -> numpy.ndarray:
        self.entered.set()
        if not self._release.wait(WAIT_TIMEOUT):
            raise RuntimeError("test iterator was not closed")
        if self.closed.is_set():
            raise StopIteration
        raise AssertionError("blocking iterator released without being closed")

    def close(self) -> None:
        self.closed.set()
        self._release.set()


class _CloseableBlockingEventsSource(faery.EventsStream):
    def __init__(self):
        self.iterator = _CloseableBlockingIterator()
        self.iteration_count = 0

    def __iter__(self) -> _CloseableBlockingIterator:
        self.iteration_count += 1
        return self.iterator

    def dimensions(self) -> tuple[int, int]:
        return DIMENSIONS


class _InstrumentedBlockingCloseIterator:
    def __init__(self):
        self.next_entered = threading.Event()
        self.release_next = threading.Event()
        self.close_entered = threading.Event()
        self.release_close = threading.Event()
        self.close_calls = 0
        self.maximum_concurrent_closes = 0
        self._active_closes = 0
        self._lock = threading.Lock()

    def __iter__(self) -> "_InstrumentedBlockingCloseIterator":
        return self

    def __next__(self) -> numpy.ndarray:
        self.next_entered.set()
        if not self.release_next.wait(WAIT_TIMEOUT * 2):
            raise RuntimeError("test iterator's next() was not released")
        raise StopIteration

    def close(self) -> None:
        with self._lock:
            self.close_calls += 1
            self._active_closes += 1
            self.maximum_concurrent_closes = max(
                self.maximum_concurrent_closes,
                self._active_closes,
            )
        self.close_entered.set()
        try:
            # Deliberately has no timeout. Tests release it in cleanup.
            self.release_close.wait()
            self.release_next.set()
        finally:
            with self._lock:
                self._active_closes -= 1


class _InstrumentedBlockingCloseEventsSource(faery.EventsStream):
    def __init__(self):
        self.iterator = _InstrumentedBlockingCloseIterator()

    def __iter__(self) -> _InstrumentedBlockingCloseIterator:
        return self.iterator

    def dimensions(self) -> tuple[int, int]:
        return DIMENSIONS


@dataclasses.dataclass
class _Consumer:
    thread: threading.Thread
    packets: list[typing.Any]
    errors: list[BaseException]
    done: threading.Event


def _start_consumer(
    iterable: collections.abc.Iterable[typing.Any],
    barrier: typing.Optional[threading.Barrier] = None,
    first_packet: typing.Optional[threading.Event] = None,
) -> _Consumer:
    packets: list[typing.Any] = []
    errors: list[BaseException] = []
    done = threading.Event()

    def target() -> None:
        try:
            if barrier is not None:
                barrier.wait(timeout=WAIT_TIMEOUT)
            for packet in iterable:
                packets.append(packet)
                if len(packets) == 1 and first_packet is not None:
                    first_packet.set()
        except BaseException as error:
            errors.append(error)
        finally:
            done.set()

    thread = threading.Thread(target=target, daemon=False)
    thread.start()
    return _Consumer(thread=thread, packets=packets, errors=errors, done=done)


def _wait(event: threading.Event, message: str) -> None:
    assert event.wait(WAIT_TIMEOUT), message


def _join(consumer: _Consumer) -> None:
    consumer.thread.join(WAIT_TIMEOUT)
    assert not consumer.thread.is_alive(), "consumer thread did not stop"
    assert consumer.done.is_set()


def _cleanup(tee: faery.Tee, consumers: list[_Consumer]) -> None:
    tee.close()
    for consumer in consumers:
        consumer.thread.join(WAIT_TIMEOUT)


def _drain_concurrently(tee: faery.Tee) -> list[_Consumer]:
    barrier = threading.Barrier(len(tee) + 1)
    consumers = [_start_consumer(branch, barrier=barrier) for branch in tee]
    try:
        barrier.wait(timeout=WAIT_TIMEOUT)
        for consumer in consumers:
            _join(consumer)
            assert consumer.errors == []
    except BaseException:
        tee.close()
        for consumer in consumers:
            consumer.thread.join(WAIT_TIMEOUT)
        raise
    return consumers


def _drain_after_upstream_exhausts(
    tee: faery.Tee, exhausted: threading.Event
) -> list[_Consumer]:
    first_packets: list[typing.Any] = []
    first_errors: list[BaseException] = []
    first_received = threading.Event()
    release_first = threading.Event()
    first_done = threading.Event()

    def consume_first() -> None:
        iterator = iter(tee[0])
        try:
            first_packets.append(next(iterator))
            first_received.set()
            if not release_first.wait(WAIT_TIMEOUT):
                raise RuntimeError("first branch was not released")
            first_packets.extend(iterator)
        except BaseException as error:
            first_errors.append(error)
        finally:
            first_done.set()

    first_thread = threading.Thread(target=consume_first, daemon=False)
    first = _Consumer(
        thread=first_thread,
        packets=first_packets,
        errors=first_errors,
        done=first_done,
    )
    first_thread.start()
    second: typing.Optional[_Consumer] = None
    try:
        _wait(first_received, "first branch did not receive its initial packet")
        _wait(exhausted, "unbounded producer did not exhaust the upstream")
        second = _start_consumer(tee[1])
        release_first.set()
        _join(first)
        _join(second)
        assert first.errors == []
        assert second.errors == []
        return [first, second]
    except BaseException:
        release_first.set()
        tee.close()
        first.thread.join(WAIT_TIMEOUT)
        if second is not None:
            second.thread.join(WAIT_TIMEOUT)
        raise


def _assert_event_packets(
    actual: list[numpy.ndarray], expected: list[numpy.ndarray]
) -> None:
    assert len(actual) == len(expected)
    for actual_packet, expected_packet in zip(actual, expected):
        numpy.testing.assert_array_equal(actual_packet, expected_packet)


def test_tee_delivers_every_packet_once_to_more_than_two_branches():
    packets = event_packets()
    source = _TrackedEventsSource(packets)
    tee = source.tee(n=3, maxsize=1)
    consumers: list[_Consumer] = []
    try:
        consumers = _drain_concurrently(tee)
        assert source.iteration_count == 1
        for consumer in consumers:
            _assert_event_packets(consumer.packets, packets)
    finally:
        _cleanup(tee, consumers)


def test_tee_is_a_sequence_and_remains_unpackable():
    tee = _EventsSource(event_packets()).tee(n=3)
    try:
        assert isinstance(tee, faery.Tee)
        assert len(tee) == 3
        first, second, third = tee
        assert tee[0] is first
        assert tee[1:] == [second, third]
    finally:
        tee.close()


@pytest.mark.parametrize(
    ("source_type", "branch_type", "finite", "regular"),
    [
        (_EventsSource, faery.EventsStream, False, False),
        (_FiniteEventsSource, faery.FiniteEventsStream, True, False),
        (_RegularEventsSource, faery.RegularEventsStream, False, True),
        (
            _FiniteRegularEventsSource,
            faery.FiniteRegularEventsStream,
            True,
            True,
        ),
    ],
)
def test_event_tee_preserves_stream_category_and_metadata(
    source_type, branch_type, finite, regular
):
    source = source_type(event_packets())
    tee = source.tee()
    try:
        assert source.iteration_count == 0
        for branch in tee:
            assert isinstance(branch, branch_type)
            assert branch.dimensions() == DIMENSIONS
            if finite:
                assert branch.time_range() == TIME_RANGE
            if regular:
                assert branch.frequency_hz() == FREQUENCY_HZ
    finally:
        tee.close()


@pytest.mark.parametrize(
    ("source_type", "branch_type", "finite", "regular"),
    [
        (_FrameSource, faery.FrameStream, False, False),
        (_FiniteFrameSource, faery.FiniteFrameStream, True, False),
        (_RegularFrameSource, faery.RegularFrameStream, False, True),
        (
            _FiniteRegularFrameSource,
            faery.FiniteRegularFrameStream,
            True,
            True,
        ),
    ],
)
def test_frame_tee_preserves_stream_category_and_metadata(
    source_type, branch_type, finite, regular
):
    source = source_type(frames())
    tee = source.tee()
    try:
        assert source.iteration_count == 0
        for branch in tee:
            assert isinstance(branch, branch_type)
            assert branch.dimensions() == DIMENSIONS
            if finite:
                assert branch.time_range() == TIME_RANGE
            if regular:
                assert branch.frequency_hz() == FREQUENCY_HZ
    finally:
        tee.close()


def test_event_tee_copy_true_returns_independent_mutable_packets():
    packet = events((10, 1, 2, True))
    tee = _EventsSource([packet]).tee(copy=True, maxsize=1)
    consumers: list[_Consumer] = []
    try:
        consumers = _drain_concurrently(tee)
        first = consumers[0].packets[0]
        second = consumers[1].packets[0]
        assert first.flags.writeable
        assert second.flags.writeable
        assert not numpy.shares_memory(first, second)
        first["x"][0] = 3
        assert second["x"][0] == 1
        assert packet["x"][0] == 1
    finally:
        _cleanup(tee, consumers)


def test_event_tee_copy_false_returns_read_only_shared_packets():
    tee = _EventsSource([events((10, 1, 2, True))]).tee(copy=False, maxsize=1)
    consumers: list[_Consumer] = []
    try:
        consumers = _drain_concurrently(tee)
        first = consumers[0].packets[0]
        second = consumers[1].packets[0]
        assert not first.flags.writeable
        assert not second.flags.writeable
        assert numpy.shares_memory(first, second)
        with pytest.raises(ValueError, match="read-only"):
            first["x"][0] = 3
    finally:
        _cleanup(tee, consumers)


@pytest.mark.parametrize("copy", [True, False])
def test_event_tee_snapshots_a_reused_upstream_array_per_yield(copy):
    source = _ReusedArrayEventsSource()
    tee = source.tee(copy=copy, maxsize=None)
    consumers: list[_Consumer] = []
    try:
        consumers = _drain_after_upstream_exhausts(tee, source.exhausted)
        for consumer in consumers:
            assert [int(packet["x"][0]) for packet in consumer.packets] == [0, 1, 2]
    finally:
        _cleanup(tee, consumers)


@pytest.mark.parametrize("copy", [True, False])
@pytest.mark.parametrize("error_type", [RuntimeError, StopIteration])
def test_retained_packet_copy_failure_reaches_every_branch(copy, error_type):
    tee = _EventsSource([_CopyFailingPacket(error_type)]).tee(copy=copy, maxsize=1)
    barrier = threading.Barrier(3)
    branch_errors: list[list[BaseException]] = [[], []]

    def consume_one(branch_index: int) -> None:
        try:
            barrier.wait(timeout=WAIT_TIMEOUT)
            next(iter(tee[branch_index]))
        except BaseException as error:
            branch_errors[branch_index].append(error)

    threads = [
        threading.Thread(target=consume_one, args=(index,), daemon=False)
        for index in range(2)
    ]
    try:
        for thread in threads:
            thread.start()
        barrier.wait(timeout=WAIT_TIMEOUT)
        for thread in threads:
            thread.join(WAIT_TIMEOUT)
            assert not thread.is_alive(), "branch consumer did not stop"

        for errors in branch_errors:
            assert len(errors) == 1
            if error_type is StopIteration:
                assert type(errors[0]) is RuntimeError
                assert str(errors[0]) == (
                    "tee packet retention raised StopIteration: "
                    "retained packet copy failed"
                )
                assert type(errors[0].__cause__) is StopIteration
                assert str(errors[0].__cause__) == "retained packet copy failed"
            else:
                assert type(errors[0]) is error_type
                assert str(errors[0]) == "retained packet copy failed"
    finally:
        tee.close()
        for thread in threads:
            if thread.ident is not None:
                thread.join(WAIT_TIMEOUT)


@pytest.mark.parametrize("copy", [True, False])
def test_event_tee_supports_immutable_byte_packets(copy):
    payload = b"raw event-camera packet"
    tee = _EventsSource([payload]).tee(copy=copy, maxsize=1)
    consumers: list[_Consumer] = []
    try:
        consumers = _drain_concurrently(tee)
        assert [consumer.packets for consumer in consumers] == [
            [payload],
            [payload],
        ]
    finally:
        _cleanup(tee, consumers)


def test_frame_tee_copy_true_returns_independent_mutable_pixels():
    frame = frames()[0]
    tee = _FrameSource([frame]).tee(copy=True, maxsize=1)
    consumers: list[_Consumer] = []
    try:
        consumers = _drain_concurrently(tee)
        first = consumers[0].packets[0]
        second = consumers[1].packets[0]
        assert first.t == frame.t
        assert second.t == frame.t
        assert first is not second
        assert first.pixels.flags.writeable
        assert second.pixels.flags.writeable
        assert not numpy.shares_memory(first.pixels, second.pixels)
        first.pixels[0, 0, 0] = 255
        assert second.pixels[0, 0, 0] == 0
        assert frame.pixels[0, 0, 0] == 0
    finally:
        _cleanup(tee, consumers)


def test_frame_tee_copy_false_returns_read_only_shared_pixels():
    tee = _FrameSource([frames()[0]]).tee(copy=False, maxsize=1)
    consumers: list[_Consumer] = []
    try:
        consumers = _drain_concurrently(tee)
        first = consumers[0].packets[0]
        second = consumers[1].packets[0]
        assert first.t == second.t
        assert not first.pixels.flags.writeable
        assert not second.pixels.flags.writeable
        assert numpy.shares_memory(first.pixels, second.pixels)
        with pytest.raises(ValueError, match="read-only"):
            first.pixels[0, 0, 0] = 255
    finally:
        _cleanup(tee, consumers)


@pytest.mark.parametrize("copy", [True, False])
def test_frame_tee_snapshots_reused_upstream_pixels_per_yield(copy):
    source = _ReusedPixelsFrameSource()
    tee = source.tee(copy=copy, maxsize=None)
    consumers: list[_Consumer] = []
    try:
        consumers = _drain_after_upstream_exhausts(tee, source.exhausted)
        for consumer in consumers:
            assert [int(frame.pixels[0, 0, 0]) for frame in consumer.packets] == [
                0,
                1,
                2,
            ]
            assert [frame.t for frame in consumer.packets] == [
                1 * faery.ms,
                2 * faery.ms,
                3 * faery.ms,
            ]
    finally:
        _cleanup(tee, consumers)


@pytest.mark.parametrize("n", [-1, 0, 1])
def test_tee_rejects_invalid_branch_count(n):
    with pytest.raises(ValueError, match="n"):
        _EventsSource(event_packets()).tee(n=n)


@pytest.mark.parametrize("maxsize", [-1, 0])
def test_tee_rejects_invalid_buffer_size(maxsize):
    with pytest.raises(ValueError, match="maxsize"):
        _EventsSource(event_packets()).tee(maxsize=maxsize)


def test_tee_accepts_an_explicitly_unbounded_buffer():
    tee = _EventsSource(event_packets()).tee(maxsize=None)
    consumers: list[_Consumer] = []
    try:
        consumers = _drain_concurrently(tee)
        for consumer in consumers:
            _assert_event_packets(consumer.packets, event_packets())
    finally:
        _cleanup(tee, consumers)


@pytest.mark.parametrize(
    "overflow",
    [
        "drop_newest",
        ("block",),
        ("block", "drop_newest"),
        1,
    ],
)
def test_tee_rejects_invalid_overflow_configuration(overflow):
    with pytest.raises(ValueError, match="overflow"):
        _EventsSource(event_packets()).tee(overflow=overflow)


def test_drop_oldest_requires_a_bounded_buffer():
    with pytest.raises(ValueError, match="bounded maxsize"):
        _EventsSource(event_packets()).tee(
            maxsize=None,
            overflow="drop_oldest",
        )


def test_per_branch_drop_oldest_does_not_backpressure_a_lossless_branch():
    packets = event_packets()
    source = _TrackedEventsSource(packets)
    tee = source.tee(
        maxsize=1,
        overflow=("block", "drop_oldest"),
    )
    fast = _start_consumer(tee[0])
    slow: typing.Optional[_Consumer] = None
    try:
        _join(fast)
        _wait(source.exhausted, "lossy branch unexpectedly blocked the producer")

        slow = _start_consumer(tee[1])
        _join(slow)

        assert fast.errors == []
        assert slow.errors == []
        _assert_event_packets(fast.packets, packets)
        _assert_event_packets(slow.packets, [packets[-1]])
        assert tee.dropped_packets == (0, len(packets) - 1)
    finally:
        tee.close()
        fast.thread.join(WAIT_TIMEOUT)
        if slow is not None:
            slow.thread.join(WAIT_TIMEOUT)


def test_tee_starts_the_upstream_producer_lazily():
    source = _TrackedEventsSource([events((10, 0, 0, True))])
    tee = source.tee(maxsize=1)
    try:
        assert source.iteration_count == 0
        assert not source.started.is_set()
        iterator = iter(tee[0])
        assert source.iteration_count == 0
        assert not source.started.is_set()
        numpy.testing.assert_array_equal(next(iterator), source.packets[0])
        _wait(source.started, "upstream producer did not start on first consumption")
        assert source.iteration_count == 1
    finally:
        tee.close()


def test_branch_can_close_before_consumption_without_starting_upstream():
    source = _TrackedEventsSource([events((10, 0, 0, True))])
    tee = source.tee(maxsize=1)
    try:
        tee[0].close()
        assert list(tee[0]) == []
        assert source.iteration_count == 0
        assert not source.started.is_set()
    finally:
        tee.close()


def test_tee_rejects_concurrent_and_repeated_branch_iteration():
    source = _TrackedEventsSource([events((10, 0, 0, True))])
    tee = source.tee(maxsize=1)
    consumers: list[_Consumer] = []
    first_iterator = iter(tee[0])
    try:
        with pytest.raises(RuntimeError, match="only be iterated once"):
            iter(tee[0])

        barrier = threading.Barrier(3)
        consumers = [
            _start_consumer(first_iterator, barrier=barrier),
            _start_consumer(tee[1], barrier=barrier),
        ]
        barrier.wait(timeout=WAIT_TIMEOUT)
        for consumer in consumers:
            _join(consumer)
            assert consumer.errors == []

        with pytest.raises(RuntimeError, match="only be iterated once"):
            list(tee[0])
    finally:
        _cleanup(tee, consumers)


def test_tee_rejects_concurrent_next_calls_on_one_branch_iterator():
    source = _CloseableBlockingEventsSource()
    tee = source.tee(maxsize=1)
    iterator = iter(tee[0])
    first_errors: list[BaseException] = []
    first_done = threading.Event()

    def consume_first() -> None:
        try:
            next(iterator)
        except BaseException as error:
            first_errors.append(error)
        finally:
            first_done.set()

    first_thread = threading.Thread(target=consume_first, daemon=False)
    first_thread.start()
    try:
        _wait(source.iterator.entered, "first next() did not enter the upstream")
        with pytest.raises(RuntimeError, match="concurrently"):
            next(iterator)
    finally:
        tee.close()
        first_thread.join(WAIT_TIMEOUT)

    assert not first_thread.is_alive()
    assert first_done.is_set()
    assert len(first_errors) == 1
    assert isinstance(first_errors[0], StopIteration)


def test_slowest_active_branch_applies_bounded_backpressure():
    packets = event_packets()
    source = _TrackedEventsSource(packets)
    tee = source.tee(maxsize=1)
    fast_packets: list[numpy.ndarray] = []
    fast_errors: list[BaseException] = []
    fast_first = threading.Event()
    fast_attempting_second = threading.Event()
    fast_second = threading.Event()
    fast_done = threading.Event()

    def consume_fast() -> None:
        iterator = iter(tee[0])
        try:
            fast_packets.append(next(iterator))
            fast_first.set()
            fast_attempting_second.set()
            fast_packets.append(next(iterator))
            fast_second.set()
            fast_packets.extend(iterator)
        except BaseException as error:
            fast_errors.append(error)
        finally:
            fast_done.set()

    fast_thread = threading.Thread(target=consume_fast, daemon=False)
    fast_thread.start()
    slow: typing.Optional[_Consumer] = None
    try:
        _wait(fast_first, "fast branch did not receive the first packet")
        _wait(fast_attempting_second, "fast branch did not request packet two")
        assert not source.pulled[1].wait(SHORT_WAIT_TIMEOUT)
        assert not fast_second.wait(SHORT_WAIT_TIMEOUT)

        slow = _start_consumer(tee[1])
        _wait(source.pulled[1], "producer did not resume after slow branch consumed")
        _wait(fast_second, "fast branch did not resume after slow branch consumed")
        assert fast_done.wait(WAIT_TIMEOUT)
        fast_thread.join(WAIT_TIMEOUT)
        assert not fast_thread.is_alive()
        _join(slow)
        assert fast_errors == []
        assert slow.errors == []
        _assert_event_packets(fast_packets, packets)
        _assert_event_packets(slow.packets, packets)
    finally:
        tee.close()
        fast_thread.join(WAIT_TIMEOUT)
        if slow is not None:
            slow.thread.join(WAIT_TIMEOUT)


def test_closing_a_lagging_branch_releases_the_producer():
    packets = event_packets()
    source = _TrackedEventsSource(packets)
    tee = source.tee(maxsize=1)
    first_packet = threading.Event()
    fast = _start_consumer(tee[0], first_packet=first_packet)
    try:
        _wait(first_packet, "fast branch did not receive the first packet")
        assert not source.pulled[1].wait(SHORT_WAIT_TIMEOUT)
        tee[1].close()
        _wait(source.pulled[1], "producer did not resume after lagging branch closed")
        _join(fast)
        assert fast.errors == []
        _assert_event_packets(fast.packets, packets)
    finally:
        _cleanup(tee, [fast])


def test_all_branches_can_close_while_the_producer_waits_for_capacity():
    source = _TrackedEventsSource(event_packets())
    tee = source.tee(maxsize=1)
    first_packet = threading.Event()
    fast = _start_consumer(tee[0], first_packet=first_packet)
    try:
        _wait(first_packet, "fast branch did not receive the first packet")
        assert not source.pulled[1].wait(SHORT_WAIT_TIMEOUT)
        tee[0].close()
        tee[1].close()
        _join(fast)
        assert fast.errors == []
        _wait(source.closed, "closing all branches did not close the upstream")
    finally:
        _cleanup(tee, [fast])


def test_normal_end_of_stream_is_visible_while_the_buffer_is_full():
    packet = events((10, 0, 0, True))
    source = _TrackedEventsSource([packet])
    tee = source.tee(maxsize=1)
    first_packets: list[numpy.ndarray] = []
    first_errors: list[BaseException] = []
    first_received = threading.Event()
    first_waiting_for_end = threading.Event()
    first_done = threading.Event()

    def consume_first() -> None:
        iterator = iter(tee[0])
        try:
            first_packets.append(next(iterator))
            first_received.set()
            first_waiting_for_end.set()
            with pytest.raises(StopIteration):
                next(iterator)
        except BaseException as error:
            first_errors.append(error)
        finally:
            first_done.set()

    first_thread = threading.Thread(target=consume_first, daemon=False)
    first_thread.start()
    second: typing.Optional[_Consumer] = None
    try:
        _wait(first_received, "first branch did not receive the buffered packet")
        _wait(first_waiting_for_end, "first branch did not wait for end-of-stream")
        assert not source.exhausted.wait(SHORT_WAIT_TIMEOUT)
        assert not first_done.is_set()

        second = _start_consumer(tee[1])
        _wait(source.exhausted, "producer did not publish end-of-stream")
        _wait(first_done, "first branch did not observe end-of-stream")
        first_thread.join(WAIT_TIMEOUT)
        assert not first_thread.is_alive()
        _join(second)
        _assert_event_packets(first_packets, [packet])
        _assert_event_packets(second.packets, [packet])
        assert first_errors == []
        assert second.errors == []
    finally:
        tee.close()
        first_thread.join(WAIT_TIMEOUT)
        if second is not None:
            second.thread.join(WAIT_TIMEOUT)


def test_upstream_failure_follows_buffered_packets_on_every_active_branch():
    packet = events((10, 0, 0, True))
    source = _FailingEventsSource(packet)
    tee = source.tee(maxsize=1)
    first_packets: list[numpy.ndarray] = []
    first_errors: list[BaseException] = []
    first_received = threading.Event()
    first_waiting_for_failure = threading.Event()
    first_done = threading.Event()

    def consume_first() -> None:
        iterator = iter(tee[0])
        try:
            first_packets.append(next(iterator))
            first_received.set()
            first_waiting_for_failure.set()
            next(iterator)
        except BaseException as error:
            first_errors.append(error)
        finally:
            first_done.set()

    first_thread = threading.Thread(target=consume_first, daemon=False)
    first_thread.start()
    second: typing.Optional[_Consumer] = None
    try:
        _wait(first_received, "first branch did not receive the buffered packet")
        _wait(
            first_waiting_for_failure,
            "first branch did not wait for the upstream failure",
        )
        assert not source.raised.wait(SHORT_WAIT_TIMEOUT)
        assert not first_done.is_set()

        second = _start_consumer(tee[1])
        _wait(source.raised, "producer did not observe the upstream failure")
        _wait(first_done, "first branch did not receive the upstream failure")
        first_thread.join(WAIT_TIMEOUT)
        assert not first_thread.is_alive()
        _join(second)
        _assert_event_packets(first_packets, [packet])
        assert len(first_errors) == 1
        assert isinstance(first_errors[0], _UpstreamFailure)
        assert str(first_errors[0]) == "upstream failed"
        _assert_event_packets(second.packets, [packet])
        assert len(second.errors) == 1
        assert isinstance(second.errors[0], _UpstreamFailure)
        assert str(second.errors[0]) == "upstream failed"
    finally:
        tee.close()
        first_thread.join(WAIT_TIMEOUT)
        if second is not None:
            second.thread.join(WAIT_TIMEOUT)


class _DownstreamFailure(Exception):
    pass


def test_downstream_failure_does_not_deadlock_a_sibling_branch():
    packets = event_packets()
    tee = _EventsSource(packets).tee(maxsize=1)
    barrier = threading.Barrier(3)
    failed_packets: list[numpy.ndarray] = []
    failed_errors: list[BaseException] = []
    failed_done = threading.Event()

    def fail_downstream() -> None:
        try:
            barrier.wait(timeout=WAIT_TIMEOUT)
            for packet in tee[0]:
                failed_packets.append(packet)
                raise _DownstreamFailure("sink failed")
        except BaseException as error:
            failed_errors.append(error)
        finally:
            tee[0].close()
            failed_done.set()

    failed_thread = threading.Thread(target=fail_downstream, daemon=False)
    failed_thread.start()
    sibling = _start_consumer(tee[1], barrier=barrier)
    try:
        barrier.wait(timeout=WAIT_TIMEOUT)
        _wait(failed_done, "failing downstream did not exit")
        failed_thread.join(WAIT_TIMEOUT)
        assert not failed_thread.is_alive()
        _join(sibling)
        assert len(failed_packets) == 1
        assert len(failed_errors) == 1
        assert isinstance(failed_errors[0], _DownstreamFailure)
        assert sibling.errors == []
        _assert_event_packets(sibling.packets, packets)
    finally:
        tee.close()
        failed_thread.join(WAIT_TIMEOUT)
        sibling.thread.join(WAIT_TIMEOUT)


def test_context_manager_closes_a_blocking_source_with_an_unstarted_branch():
    source = _CloseableBlockingEventsSource()
    consumer: typing.Optional[_Consumer] = None
    with source.tee(maxsize=1) as tee:
        consumer = _start_consumer(tee[0])
        _wait(source.iterator.entered, "producer did not enter blocking upstream")
        # tee[1] is intentionally never iterated.

    assert consumer is not None
    _join(consumer)
    assert consumer.errors == []
    assert source.iteration_count == 1
    assert source.iterator.closed.is_set()


def test_concurrent_repeated_group_close_closes_upstream_at_most_once():
    source = _InstrumentedBlockingCloseEventsSource()
    tee = source.tee(maxsize=1)
    consumer = _start_consumer(tee[0])
    close_barrier = threading.Barrier(5)
    close_errors: list[BaseException] = []

    def close_group() -> None:
        try:
            close_barrier.wait(timeout=WAIT_TIMEOUT)
            tee.close()
        except BaseException as error:
            close_errors.append(error)

    close_threads = [
        threading.Thread(target=close_group, daemon=False) for _ in range(4)
    ]
    try:
        _wait(source.iterator.next_entered, "producer did not enter upstream next()")
        for thread in close_threads:
            thread.start()
        close_barrier.wait(timeout=WAIT_TIMEOUT)
        _wait(source.iterator.close_entered, "upstream close() was not called")
        source.iterator.release_close.set()

        for thread in close_threads:
            thread.join(WAIT_TIMEOUT)
            assert not thread.is_alive(), "group close thread did not stop"
        _join(consumer)
        tee.close()
        tee.close()

        assert close_errors == []
        assert source.iterator.close_calls == 1
        assert source.iterator.maximum_concurrent_closes == 1
    finally:
        source.iterator.release_close.set()
        source.iterator.release_next.set()
        tee.close()
        for thread in close_threads:
            if thread.ident is not None:
                thread.join(WAIT_TIMEOUT)
        consumer.thread.join(WAIT_TIMEOUT)


def test_context_exit_is_bounded_when_upstream_close_blocks():
    source = _InstrumentedBlockingCloseEventsSource()
    tee = source.tee(maxsize=1)
    consumer = _start_consumer(tee[0])
    exit_errors: list[BaseException] = []
    exited = threading.Event()

    def exit_context() -> None:
        try:
            with tee:
                pass
        except BaseException as error:
            exit_errors.append(error)
        finally:
            exited.set()

    exit_thread = threading.Thread(target=exit_context, daemon=False)
    try:
        _wait(source.iterator.next_entered, "producer did not enter upstream next()")
        exit_thread.start()
        _wait(source.iterator.close_entered, "upstream close() was not called")
        assert exited.wait(2.0), "context exit waited indefinitely for upstream close()"
        exit_thread.join(WAIT_TIMEOUT)
        assert not exit_thread.is_alive()
        _join(consumer)
        assert exit_errors == []
        assert source.iterator.close_calls == 1
    finally:
        source.iterator.release_close.set()
        source.iterator.release_next.set()
        tee.close()
        if exit_thread.ident is not None:
            exit_thread.join(WAIT_TIMEOUT)
        consumer.thread.join(WAIT_TIMEOUT)
