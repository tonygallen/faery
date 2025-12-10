#!/usr/bin/env python3
"""
Example demonstrating simultaneous saving and rendering with tee().

This example shows the exact use case described in the issue:
simultaneously saving EventStreams and rendering them.
"""

import numpy

import faery

# Create sample events representing a simple pattern
events = numpy.array(
    [(i * 100, (i % 50) + 10, (i % 50) + 10, i % 2 == 0) for i in range(1000)],
    dtype=faery.EVENTS_DTYPE,
)
dimensions = (100, 100)

# Create an event stream
print("Creating event stream...")
stream = faery.events_stream_from_array(events, dimensions)

# Tee the stream into two: one for recording, one for viewing
print("Splitting stream with tee() for simultaneous recording and viewing...")
recording, viewing = stream.tee()

# Note: In a real scenario, you would do:
# recording, viewing = faery.events_stream_from_camera().tee()

# Save the recording branch to a file
# In a real scenario: recording.to_file("output.aedat4")
print("\nRecording branch: Processing events...")
recorded_count = 0
for packet in recording:
    recorded_count += len(packet)
print(f"  Recorded {recorded_count} events")

# Process the viewing branch with rendering and other operations
print("\nViewing branch: Rendering and processing...")
viewing_count = 0
for packet in viewing.crop(10, 90, 10, 90):  # Apply a crop filter
    viewing_count += len(packet)
# In a real scenario: viewing.render(...).to_file("output.mp4")
print(f"  Processed {viewing_count} events after cropping")

print("\n" + "=" * 60)
print("✓ Successfully demonstrated tee() for simultaneous operations!")
print(f"  Recording: {recorded_count} events")
print(f"  Viewing:   {viewing_count} events (after crop)")
print("\nUse case: This allows you to save raw events while simultaneously")
print("visualizing or analyzing them in real-time!")
