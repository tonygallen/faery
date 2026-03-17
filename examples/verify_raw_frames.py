"""
Headless verifier for raw RGB frame streams produced by to_stdout_raw().

Reads raw RGB frames from stdin and prints per-frame statistics to stderr,
so you can confirm on a headless machine (no monitor) that frames are
flowing correctly from the event camera pipeline.

Usage
-----
Run the camera pipeline in one shell and pipe its stdout into this script:

    python examples/camera_frames_to_stdout_raw.py | \\
        python examples/verify_raw_frames.py --width 640 --height 480

Stop after verifying a fixed number of frames (useful on a Jetson where
Ctrl-C to the producer may leave the pipe open):

    python examples/camera_frames_to_stdout_raw.py | \\
        python examples/verify_raw_frames.py --width 640 --height 480 --count 30

Common sensor resolutions
-------------------------
  Inivation DVXplorer / DAVIS346:  640x480  or  346x260
  Prophesee EVK4 / EVK3-HD:        1280x720
"""

import argparse
import sys

import numpy


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Verify raw RGB frames from to_stdout_raw() on a headless machine. "
            "Pipe the output of camera_frames_to_stdout_raw.py into this script."
        )
    )
    parser.add_argument(
        "--width",
        type=int,
        required=True,
        help="Frame width in pixels (must match the camera sensor resolution)",
    )
    parser.add_argument(
        "--height",
        type=int,
        required=True,
        help="Frame height in pixels (must match the camera sensor resolution)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        metavar="N",
        help="Stop after receiving N frames (default: run until stdin closes or Ctrl+C)",
    )
    args = parser.parse_args()

    width: int = args.width
    height: int = args.height
    bytes_per_frame: int = width * height * 3

    sys.stderr.write(
        f"Waiting for {width}x{height} RGB frames "
        f"({bytes_per_frame} bytes / frame) ...\n"
    )
    sys.stderr.flush()

    stdin = sys.stdin.buffer
    frame_index = 0

    try:
        while args.count is None or frame_index < args.count:
            raw = stdin.read(bytes_per_frame)

            if len(raw) == 0:
                sys.stderr.write("stdin closed — stream ended.\n")
                break

            if len(raw) < bytes_per_frame:
                sys.stderr.write(
                    f"Incomplete final frame: received {len(raw)} bytes, "
                    f"expected {bytes_per_frame}.\n"
                )
                break

            pixels = numpy.frombuffer(raw, dtype=numpy.uint8).reshape(
                height, width, 3
            )

            mean_r = float(pixels[:, :, 0].mean())
            mean_g = float(pixels[:, :, 1].mean())
            mean_b = float(pixels[:, :, 2].mean())
            non_black = int((pixels.sum(axis=2) > 0).sum())

            sys.stderr.write(
                f"Frame {frame_index:5d}  "
                f"mean RGB = ({mean_r:5.1f}, {mean_g:5.1f}, {mean_b:5.1f})  "
                f"non-black pixels = {non_black:7d} / {width * height}\n"
            )
            sys.stderr.flush()

            frame_index += 1

    except KeyboardInterrupt:
        pass

    sys.stderr.write(f"\nReceived {frame_index} frame(s). Done.\n")


if __name__ == "__main__":
    main()
