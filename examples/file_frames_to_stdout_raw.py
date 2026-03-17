"""
Headless test for to_stdout_raw() using a pre-recorded event file.

Use this script on a headless Nvidia Jetson (or any machine without a
display or event camera) to confirm that the raw-RGB stdout pipeline is
working correctly before connecting real hardware.

The script reads from a bundled DVS recording, renders 30 fps frames, and
pipes the raw RGB bytes to stdout.  A second Python process reads those bytes
and checks that each frame has the expected dimensions.

Usage
-----
python examples/file_frames_to_stdout_raw.py

Expected output
---------------
  Frame 0 OK  (320 x 240 x 3 = 230400 bytes)
  Frame 1 OK  (320 x 240 x 3 = 230400 bytes)
  ...
  All N frames verified.
"""

import io
import sys
import unittest.mock

import numpy

import faery

# The bundled DVS recording shipped with the test suite.
EVENT_FILE = faery.dirname.parent / "tests" / "data" / "dvs.es"

WIDTH, HEIGHT = 320, 240
BYTES_PER_FRAME = WIDTH * HEIGHT * 3  # RGB only (no alpha)

buf = io.BytesIO()

with unittest.mock.patch.object(sys.stdout, "buffer", buf):
    (
        faery.events_stream_from_file(EVENT_FILE)
        .regularize(frequency_hz=30.0)
        .render(
            decay="exponential",
            tau="00:00:00.200000",
            colormap=faery.colormaps.managua.flipped(),
        )
        .to_stdout_raw()
    )

raw = buf.getvalue()
assert len(raw) % BYTES_PER_FRAME == 0, (
    f"Unexpected output length {len(raw)} — not a multiple of {BYTES_PER_FRAME}"
)

frame_count = len(raw) // BYTES_PER_FRAME
for i in range(frame_count):
    chunk = numpy.frombuffer(
        raw[i * BYTES_PER_FRAME : (i + 1) * BYTES_PER_FRAME], dtype=numpy.uint8
    ).reshape(HEIGHT, WIDTH, 3)
    sys.stderr.write(
        f"  Frame {i} OK  ({WIDTH} x {HEIGHT} x 3 = {BYTES_PER_FRAME} bytes)\n"
    )

sys.stderr.write(f"All {frame_count} frames verified.\n")
