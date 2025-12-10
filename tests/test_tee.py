import typing

import numpy
import pytest

import faery


def test_tee_basic():
    """Test basic tee functionality with a simple event array."""
    # Create a simple event array
    events = numpy.array(
        [(0, 10, 20, True), (1, 11, 21, False), (2, 12, 22, True)],
        dtype=faery.EVENTS_DTYPE,
    )
    dimensions = (100, 100)

    # Create a stream from the array
    stream = faery.events_stream_from_array(events, dimensions)

    # Tee the stream
    stream1, stream2 = stream.tee()

    # Collect events from both streams
    events1 = []
    events2 = []

    for e in stream1:
        events1.append(e)

    for e in stream2:
        events2.append(e)

    # Verify both streams received the same data
    assert len(events1) == len(events2)
    assert len(events1) > 0

    # Concatenate and compare
    result1 = (
        numpy.concatenate(events1)
        if events1
        else numpy.array([], dtype=faery.EVENTS_DTYPE)
    )
    result2 = (
        numpy.concatenate(events2)
        if events2
        else numpy.array([], dtype=faery.EVENTS_DTYPE)
    )

    assert len(result1) == len(events)
    assert len(result2) == len(events)
    assert numpy.array_equal(result1, result2)


def test_tee_multiple_outputs():
    """Test tee with more than 2 outputs."""
    events = numpy.array(
        [(0, 10, 20, True), (1, 11, 21, False), (2, 12, 22, True)],
        dtype=faery.EVENTS_DTYPE,
    )
    dimensions = (100, 100)

    stream = faery.events_stream_from_array(events, dimensions)

    # Tee into 3 streams
    stream1, stream2, stream3 = stream.tee(n=3)

    # Collect from all streams
    results = []
    for s in [stream1, stream2, stream3]:
        stream_events = []
        for e in s:
            stream_events.append(e)
        results.append(
            numpy.concatenate(stream_events)
            if stream_events
            else numpy.array([], dtype=faery.EVENTS_DTYPE)
        )

    # All should have the same data
    for result in results:
        assert len(result) == len(events)
        assert numpy.array_equal(result["t"], events["t"])


def test_tee_with_filters():
    """Test that tee works with filters applied to each branch."""
    events = numpy.array(
        [
            (0, 10, 20, True),
            (100, 11, 21, False),
            (200, 12, 22, True),
            (300, 13, 23, False),
        ],
        dtype=faery.EVENTS_DTYPE,
    )
    dimensions = (100, 100)

    stream = faery.events_stream_from_array(events, dimensions)

    # Tee and apply different filters
    stream1, stream2 = stream.tee()

    # Filter stream1 to only ON events
    on_events = []
    for e in stream1.remove_off_events():
        on_events.append(e)

    # Filter stream2 to only OFF events
    off_events = []
    for e in stream2.remove_on_events():
        off_events.append(e)

    on_result = (
        numpy.concatenate(on_events)
        if on_events
        else numpy.array([], dtype=faery.EVENTS_DTYPE)
    )
    off_result = (
        numpy.concatenate(off_events)
        if off_events
        else numpy.array([], dtype=faery.EVENTS_DTYPE)
    )

    # Verify filtering worked correctly
    assert all(on_result["on"])
    assert all(~off_result["on"])

    # Together they should have all events
    assert len(on_result) + len(off_result) == len(events)


def test_tee_dimensions():
    """Test that teed streams preserve dimensions."""
    events = numpy.array(
        [(0, 10, 20, True)],
        dtype=faery.EVENTS_DTYPE,
    )
    dimensions = (640, 480)

    stream = faery.events_stream_from_array(events, dimensions)
    stream1, stream2 = stream.tee()

    assert stream1.dimensions() == dimensions
    assert stream2.dimensions() == dimensions


def test_tee_empty_stream():
    """Test tee with an empty event stream."""
    events = numpy.array([], dtype=faery.EVENTS_DTYPE)
    dimensions = (100, 100)

    stream = faery.events_stream_from_array(events, dimensions)
    stream1, stream2 = stream.tee()

    result1 = list(stream1)
    result2 = list(stream2)

    # Both should handle empty streams gracefully
    assert len(result1) <= 1  # May have one empty packet or no packets
    assert len(result2) <= 1


def test_tee_invalid_n():
    """Test that tee raises error for invalid n values."""
    events = numpy.array(
        [(0, 10, 20, True)],
        dtype=faery.EVENTS_DTYPE,
    )
    dimensions = (100, 100)

    stream = faery.events_stream_from_array(events, dimensions)

    with pytest.raises(ValueError):
        stream.tee(n=1)

    with pytest.raises(ValueError):
        stream.tee(n=0)
