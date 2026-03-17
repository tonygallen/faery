import io
import sys
import unittest.mock

import faery

from . import assets


def test_to_stdout_raw_output_size():
    """
    Verify that to_stdout_raw writes the correct number of raw RGB bytes to
    stdout.  This test uses a pre-recorded event file so it works on headless
    machines (no camera, no display required).

    The DVS test fixture has dimensions (width=320, height=240).  After
    regularise + render, each frame must produce exactly width * height * 3
    bytes (RGB, no alpha).  We capture sys.stdout.buffer via a BytesIO so the
    test produces no visible side-effects.
    """
    width, height = 320, 240
    bytes_per_frame = width * height * 3

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
        f"(width={width} height={height} channels=3)"
    )


def test_to_stdout_raw_frame_content():
    """
    Verify that the pixel values written by to_stdout_raw match what is
    present in frame.pixels[:, :, :3] — i.e. that the alpha channel is
    correctly dropped and byte order is row-major (C order).
    """
    import numpy

    width, height = 320, 240
    bytes_per_frame = width * height * 3

    # Collect frames independently so we can compare later.
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
    frame_count = len(raw) // bytes_per_frame
    assert frame_count == len(frames), (
        f"Expected {len(frames)} frames in raw output, got {frame_count}"
    )

    for i, frame in enumerate(frames):
        start = i * bytes_per_frame
        chunk = numpy.frombuffer(
            raw[start : start + bytes_per_frame], dtype=numpy.uint8
        ).reshape(height, width, 3)
        expected = frame.pixels[:, :, :3]
        assert numpy.array_equal(chunk, expected), (
            f"Frame {i}: raw bytes do not match frame.pixels[:, :, :3]"
        )
