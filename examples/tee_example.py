#!/usr/bin/env python3
"""Example demonstrating the tee() operation for EventStreams."""

import numpy

import faery

# Create some sample events
events = numpy.array(
    [(i * 1000, 10 + i, 20 + i, i % 2 == 0) for i in range(100)],
    dtype=faery.EVENTS_DTYPE,
)
dimensions = (100, 100)

# Create an event stream from the array
stream = faery.events_stream_from_array(events, dimensions)

# Use tee to split the stream into two independent streams
print("Splitting stream using tee()...")
stream1, stream2 = stream.tee()

# Process stream1: count ON events
print("Processing stream1: counting ON events...")
on_count = 0
for packet in stream1.remove_off_events():
    on_count += len(packet)
print(f"Stream1 found {on_count} ON events")

# Process stream2: count OFF events
print("Processing stream2: counting OFF events...")
off_count = 0
for packet in stream2.remove_on_events():
    off_count += len(packet)
print(f"Stream2 found {off_count} OFF events")

print(f"\nTotal events: {on_count + off_count} (expected: {len(events)})")
print("✓ Tee operation successful!")

# Example with 3 outputs
print("\n" + "=" * 60)
print("Splitting stream into 3 outputs...")
stream = faery.events_stream_from_array(events, dimensions)
stream1, stream2, stream3 = stream.tee(n=3)

counts = []
for i, s in enumerate([stream1, stream2, stream3], 1):
    count = sum(len(packet) for packet in s)
    counts.append(count)
    print(f"Stream{i} processed {count} events")

print(
    f"\nAll streams processed the same number of events: {all(c == counts[0] for c in counts)}"
)
print("✓ Multiple output tee operation successful!")
