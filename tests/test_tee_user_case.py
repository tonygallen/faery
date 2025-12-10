"""
Test that demonstrates the exact use case from the issue:
simultaneous saving and viewing with sequential consumption.
"""

import numpy

import faery


def test_save_then_view_pattern():
    """
    Reproduces the exact pattern from the issue:
    1. Split stream with tee()
    2. Fully consume first branch (simulating to_file())
    3. Then fully consume second branch (simulating render().view())
    """
    # Create test events
    events = numpy.array(
        [(i * 1000, 10 + i % 10, 20 + i % 10, i % 2 == 0) for i in range(200)],
        dtype=faery.EVENTS_DTYPE,
    )
    dimensions = (100, 100)

    # Simulate: stream = faery.events_stream_from_camera(driver='NeuromorphicDrivers')
    stream = faery.events_stream_from_array(events, dimensions)

    # Split the stream into two branches (as in the issue)
    recording, viewing = stream.tee(2)

    # First branch: save events to a file (simulated by counting)
    print("Processing recording branch (simulating to_file())...")
    recording_count = sum(len(packet) for packet in recording)
    print(f"  Recorded {recording_count} events")

    # Second branch: render and view the events (simulated by counting with filter)
    print("Processing viewing branch (simulating render().view())...")
    viewing_count = sum(
        len(packet) for packet in viewing.regularize(frequency_hz=30).crop(5, 95, 5, 95)
    )
    print(f"  Viewed {viewing_count} events (after crop)")

    # Both branches should have processed events
    assert (
        recording_count == 200
    ), f"Expected 200 recorded events, got {recording_count}"
    assert (
        viewing_count == 200
    ), f"Expected 200 viewed events, got {viewing_count}"  # After crop

    print("✓ Sequential save-then-view pattern works correctly!")


if __name__ == "__main__":
    test_save_then_view_pattern()
