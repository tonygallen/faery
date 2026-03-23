"""
Stream raw RGB frames from an event camera to stdout with per-frame PTS.

Wire format
-----------
Each frame is written as:

    [8 bytes: PTS in nanoseconds, unsigned 64-bit little-endian]
    [width * height * 3 bytes: RGB pixel data, row-major]

The PTS value comes directly from the event timestamp (faery uses microseconds
internally; the value is multiplied by 1000 to give nanoseconds, matching
GStreamer's GST_SECOND = 1,000,000,000 convention).

GStreamer appsrc usage (Python)
-------------------------------
Read the 8-byte PTS header before each frame and pass it to GstBuffer.pts so
that GStreamer can schedule frames correctly::

    import struct
    PTS_HEADER = 8
    FRAME_BYTES = width * height * 3

    while True:
        hdr = pipe.read(PTS_HEADER)
        if len(hdr) < PTS_HEADER:
            break
        pts_ns, = struct.unpack("<Q", hdr)
        data = pipe.read(FRAME_BYTES)
        if len(data) < FRAME_BYTES:
            break
        buf = Gst.Buffer.new_wrapped(data)
        buf.pts = pts_ns          # equivalent to gst_buffer_set_pts()
        appsrc.emit("push-buffer", buf)

Headless verification (no monitor required)
--------------------------------------------
Pipe this script into verify_raw_frames.py to confirm frames are flowing:

    python examples/camera_frames_to_stdout_raw.py | \\
        python examples/verify_raw_frames.py --width 640 --height 480

Replace 640x480 with your camera's actual sensor resolution:
  Inivation DVXplorer:  640x480
  Inivation DAVIS346:   346x260
  Prophesee EVK4:       1280x720
"""

import faery

(
    faery.events_stream_from_camera()
    .regularize(frequency_hz=30.0)
    .render(
        decay="exponential",
        tau="00:00:00.100000",
        colormap=faery.colormaps.starry_night,
    )
    .to_stdout_raw()
)
