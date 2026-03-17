"""
Stream raw RGB frames from an event camera to stdout.

On a headless machine (no monitor) you can verify the output is correct by
piping this script into verify_raw_frames.py:

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
