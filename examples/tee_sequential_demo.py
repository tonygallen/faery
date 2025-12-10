#!/usr/bin/env python3
"""
Visual demonstration that the sequential consumption fix works.

This example mimics the exact use case from the issue:
1. Open a stream
2. Tee it into recording and viewing branches
3. Fully consume recording (simulating to_file())
4. Then fully consume viewing (simulating render().view())
"""

import numpy

import faery

print("=" * 70)
print("DEMONSTRATION: Sequential consumption with tee()")
print("=" * 70)

# Create sample events
events = numpy.array(
    [(i * 100, 10 + i % 20, 20 + i % 20, i % 2 == 0) for i in range(500)],
    dtype=faery.EVENTS_DTYPE,
)
dimensions = (100, 100)

print("\n1. Creating event stream with 500 events...")
stream = faery.events_stream_from_array(events, dimensions)
print(f"   Stream dimensions: {stream.dimensions()}")

print("\n2. Splitting stream with tee()...")
recording, viewing = stream.tee()
print("   Created two branches: 'recording' and 'viewing'")

print("\n3. Fully consuming 'recording' branch (simulates to_file())...")
recording_events = []
packet_count = 0
for packet in recording:
    recording_events.extend(packet)
    packet_count += 1
print(f"   ✓ Recording branch consumed {len(recording_events)} events")
print(f"   ✓ Received in {packet_count} packet(s)")

print("\n4. Now consuming 'viewing' branch (simulates render().view())...")
print("   Applying filters: regularize(30hz) -> crop...")
viewing_events = []
packet_count = 0
for packet in viewing.regularize(frequency_hz=30).crop(10, 90, 10, 90):
    viewing_events.extend(packet)
    packet_count += 1
print(f"   ✓ Viewing branch consumed {len(viewing_events)} events")
print(f"   ✓ Received in {packet_count} packet(s)")

print("\n" + "=" * 70)
print("RESULT:")
print("=" * 70)
print(f"Recording branch: {len(recording_events)} events processed")
print(f"Viewing branch:   {len(viewing_events)} events processed")
print()

if len(recording_events) == 500 and len(viewing_events) == 500:
    print("✓ SUCCESS! Both branches processed all events correctly.")
    print("✓ Sequential consumption works as expected!")
else:
    print("✗ FAILED! Expected 500 events in each branch.")
    print("  This indicates the tee operation is not working correctly.")

print()
print("This demonstrates that you can now:")
print("  stream = faery.events_stream_from_camera()")
print("  recording, viewing = stream.tee()")
print("  recording.to_file('output.es')  # Fully consume recording first")
print("  viewing.render(...).view()       # Then fully consume viewing")
print()
