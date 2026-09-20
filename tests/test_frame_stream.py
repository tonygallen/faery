import io
import struct
import sys
import unittest.mock

import faery

from . import assets

_PTS_HEADER_SIZE = 8  # bytes: little-endian uint64 nanoseconds
_PTS_FORMAT = "<Q"


def test_to_stdout_raw_output_size():
    """
    Verify that to_stdout_raw writes the correct number of bytes to stdout.

    Each frame is preceded by an 8-byte little-endian uint64 PTS (nanoseconds),
    followed by width * height * 3 bytes of RGB pixel data.  This test uses a
    pre-recorded event file so it works on headless machines (no camera, no
    display required).
    """
    width, height = 320, 240
    bytes_per_frame = _PTS_HEADER_SIZE + width * height * 3

    buf = io.BytesIO()

    with unittest.mock.patch.object(sys.stdout, "buffer", buf):
        (
            faery.events_stream_from_file(
                assets.dirname / "data" / "dvs.es",
            )
            .regularize(frequency_hz=30.0)
            .render(
                decay="exponential",
                tau="00:00:00.200000",
                colormap=faery.colormaps.managua.flipped(),
            )
            .to_stdout_raw()
        )

    raw = buf.getvalue()

    assert len(raw) > 0, "to_stdout_raw produced no output"
    assert len(raw) % bytes_per_frame == 0, (
        f"Output length {len(raw)} is not a multiple of {bytes_per_frame} "
        f"(PTS header={_PTS_HEADER_SIZE} + width={width} height={height} channels=3)"
    )


def test_to_stdout_raw_pts_and_content():
    """
    Verify that:
    - Each frame is preceded by the correct 8-byte little-endian nanosecond PTS.
    - PTS values are monotonically non-decreasing.
    - The RGB payload matches frame.pixels[:, :, :3] (alpha dropped, C order).
    """
    import numpy

    width, height = 320, 240
    rgb_bytes = width * height * 3
    stride = _PTS_HEADER_SIZE + rgb_bytes

    # Collect frames independently so we can compare PTS and pixel values.
    frames = list(
        faery.events_stream_from_file(
            assets.dirname / "data" / "dvs.es",
        )
        .regularize(frequency_hz=30.0)
        .render(
            decay="exponential",
            tau="00:00:00.200000",
            colormap=faery.colormaps.managua.flipped(),
        )
    )

    buf = io.BytesIO()
    with unittest.mock.patch.object(sys.stdout, "buffer", buf):
        (
            faery.events_stream_from_file(
                assets.dirname / "data" / "dvs.es",
            )
            .regularize(frequency_hz=30.0)
            .render(
                decay="exponential",
                tau="00:00:00.200000",
                colormap=faery.colormaps.managua.flipped(),
            )
            .to_stdout_raw()
        )

    raw = buf.getvalue()
    frame_count = len(raw) // stride
    assert frame_count == len(frames), (
        f"Expected {len(frames)} frames in raw output, got {frame_count}"
    )

    prev_pts = -1
    for i, frame in enumerate(frames):
        offset = i * stride

        # Verify PTS header
        (pts_ns,) = struct.unpack(_PTS_FORMAT, raw[offset : offset + _PTS_HEADER_SIZE])
        expected_pts_ns = frame.t.microseconds * 1000
        assert pts_ns == expected_pts_ns, (
            f"Frame {i}: PTS {pts_ns} ns != expected {expected_pts_ns} ns "
            f"(frame.t = {frame.t.microseconds} µs)"
        )
        assert pts_ns >= prev_pts, (
            f"Frame {i}: PTS {pts_ns} ns is less than previous PTS {prev_pts} ns"
        )
        prev_pts = pts_ns

        # Verify RGB payload
        chunk = numpy.frombuffer(
            raw[offset + _PTS_HEADER_SIZE : offset + stride], dtype=numpy.uint8
        ).reshape(height, width, 3)
        expected = frame.pixels[:, :, :3]
        assert numpy.array_equal(chunk, expected), (
            f"Frame {i}: raw bytes do not match frame.pixels[:, :, :3]"
        )
