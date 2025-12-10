"""Test to verify sequential consumption works correctly."""

import numpy

import faery


def test_tee_sequential_consumption():
    """
    Test that tee() works when one branch is fully consumed before the other.
    This is the use case from the issue: saving to file (consuming stream1 entirely)
    then rendering (consuming stream2 entirely).
    """
    events = numpy.array(
        [(i, 10, 20, True) for i in range(100)],
        dtype=faery.EVENTS_DTYPE,
    )
    dimensions = (100, 100)

    stream = faery.events_stream_from_array(events, dimensions)
    stream1, stream2 = stream.tee()

    # Fully consume stream1 first (simulates recording.to_file())
    count1 = sum(len(packet) for packet in stream1)

    # Then consume stream2 (simulates viewing.render().view())
    count2 = sum(len(packet) for packet in stream2)

    print(f"Stream1 consumed: {count1}")
    print(f"Stream2 consumed: {count2}")

    # Both should get all events
    assert count1 == 100
    assert count2 == 100


def test_tee_reverse_sequential_consumption():
    """Test that order doesn't matter - stream2 can be consumed before stream1."""
    events = numpy.array(
        [(i, 10, 20, True) for i in range(50)],
        dtype=faery.EVENTS_DTYPE,
    )
    dimensions = (100, 100)

    stream = faery.events_stream_from_array(events, dimensions)
    stream1, stream2 = stream.tee()

    # Consume stream2 first this time
    count2 = sum(len(packet) for packet in stream2)

    # Then consume stream1
    count1 = sum(len(packet) for packet in stream1)

    # Both should still get all events
    assert count1 == 50
    assert count2 == 50


def test_tee_partial_then_full_consumption():
    """Test consuming one stream partially, then fully consuming both."""
    events = numpy.array(
        [(i, 10, 20, True) for i in range(100)],
        dtype=faery.EVENTS_DTYPE,
    )
    dimensions = (100, 100)

    stream = faery.events_stream_from_array(events, dimensions)
    stream1, stream2 = stream.tee()

    # Partially consume stream1
    iter1 = iter(stream1)
    packet1 = next(iter1)  # Get first packet
    partial_count = len(packet1)

    # Fully consume stream2
    count2 = sum(len(packet) for packet in stream2)

    # Finish consuming stream1
    remaining1 = sum(len(packet) for packet in iter1)
    count1 = partial_count + remaining1

    # Both should have all events
    assert count1 == 100
    assert count2 == 100
