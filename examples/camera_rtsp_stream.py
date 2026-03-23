"""
Stream an event camera as a live RTSP H.264 feed using faery + GStreamer.

Architecture
------------
A background thread runs the faery pipeline (camera → regularize → render)
and pushes (pts_ns, rgb_bytes) tuples onto a thread-safe queue.  The
GStreamer appsrc element calls the need-data callback, which dequeues one
frame, stamps the Gst.Buffer with the correct PTS (nanoseconds), and pushes
it into the H.264 encoder.  GstRtspServer wraps everything in RTSP and serves
it over TCP at the configured address.

            faery thread                     GStreamer streaming thread
        ┌──────────────────┐               ┌──────────────────────────────┐
        │ events_stream_    │               │ appsrc → videoconvert →       │
        │ from_camera()     │               │ x264enc / nvv4l2h264enc →     │
        │ .regularize()     │  queue.Queue  │ rtph264pay → RTSPServer       │
        │ .render()         │──────────────▶│ (need-data callback sets PTS) │
        └──────────────────┘               └──────────────────────────────┘

Usage
-----
    python examples/camera_rtsp_stream.py [options]

Then connect any standard player:
    ffplay  rtsp://<jetson-ip>:8554/stream
    vlc     rtsp://<jetson-ip>:8554/stream
    gst-launch-1.0 rtspsrc location=rtsp://<jetson-ip>:8554/stream ! \\
        decodebin ! videoconvert ! autovideosink

Options
-------
  --host HOST        IP address for the RTSP server to listen on
                     (default: 0.0.0.0 — all interfaces)
  --port PORT        TCP port for the RTSP server (default: 8554)
  --mount MOUNT      RTSP mount point, e.g. /stream (default: /stream)
  --fps FPS          Render frame rate in Hz (default: 30)
  --tau TAU          Exponential decay time constant, hh:mm:ss.µµµµµµ
                     (default: 00:00:00.100000)
  --colormap NAME    Faery colormap name (default: starry_night)
  --width W          Frame width in pixels; auto-detected if omitted
  --height H         Frame height in pixels; auto-detected if omitted
  --hw-encoder       Use the Jetson hardware H.264 encoder (nvv4l2h264enc)
                     instead of the software encoder (x264enc)
  --driver DRIVER    Camera driver: Auto | EventCameraDrivers |
                     NeuromorphicDrivers (default: Auto)

Dependencies
------------
Install once on Ubuntu / Jetson:

    sudo apt install python3-gi gir1.2-gst-rtsp-server-1.0 \\
                     gstreamer1.0-rtsp gstreamer1.0-plugins-good \\
                     gstreamer1.0-plugins-ugly gstreamer1.0-plugins-bad \\
                     gstreamer1.0-plugins-base gstreamer1.0-tools

Common sensor resolutions
-------------------------
  Inivation DVXplorer:  640 × 480
  Inivation DAVIS346:   346 × 260
  Prophesee EVK4:       1280 × 720
"""

import argparse
import queue
import signal
import sys
import threading

import faery

try:
    import gi

    gi.require_version("Gst", "1.0")
    gi.require_version("GstApp", "1.0")
    gi.require_version("GstRtspServer", "1.0")
    from gi.repository import GLib, Gst, GstRtspServer
except (ImportError, ValueError) as _gst_err:
    sys.stderr.write(
        f"GStreamer Python bindings not available: {_gst_err}\n"
        "Install with:\n"
        "  sudo apt install python3-gi gir1.2-gst-rtsp-server-1.0 \\\n"
        "                   gstreamer1.0-rtsp gstreamer1.0-plugins-good \\\n"
        "                   gstreamer1.0-plugins-ugly gstreamer1.0-plugins-bad \\\n"
        "                   gstreamer1.0-plugins-base\n"
    )
    sys.exit(1)

# Sentinel pushed onto the queue to signal end-of-stream.
_SENTINEL = object()


class _FaeryRtspFactory(GstRtspServer.RTSPMediaFactory):
    """
    RTSP media factory that serves rendered event-camera frames as H.264.

    The factory holds a reference to the shared frame queue.  Each time
    GstRtspServer creates a new media (once per stream with shared=True),
    it configures the appsrc element and connects the need-data callback.
    """

    def __init__(
        self,
        frame_queue: queue.Queue,
        width: int,
        height: int,
        fps: int,
        use_hw_encoder: bool,
    ):
        super().__init__()
        self._queue = frame_queue
        self._width = width
        self._height = height
        self._fps = fps
        self._frame_duration_ns = int(1_000_000_000 / fps)

        encoder = (
            "nvv4l2h264enc maxperf-enable=1"
            if use_hw_encoder
            else "x264enc tune=zerolatency speed-preset=ultrafast"
        )
        caps = (
            f"video/x-raw,format=RGB,"
            f"width={width},height={height},"
            f"framerate={fps}/1"
        )
        pipeline = (
            f"( appsrc name=src is-live=true block=true format=time "
            f"caps={caps} ! "
            f"videoconvert ! "
            f"{encoder} ! "
            f"rtph264pay name=pay0 pt=96 )"
        )
        self.set_launch(pipeline)
        # Share the single pipeline among all connected clients so we only
        # iterate the camera once regardless of viewer count.
        self.set_shared(True)
        self.connect("media-configure", self._on_media_configure)

    # ------------------------------------------------------------------
    # Signal callbacks
    # ------------------------------------------------------------------

    def _on_media_configure(self, _factory, rtsp_media):
        """Called once when GstRtspServer builds the pipeline for a new media."""
        element = rtsp_media.get_element()
        appsrc = element.get_child_by_name("src")
        if appsrc is None:
            sys.stderr.write("ERROR: could not find appsrc element in pipeline\n")
            return
        appsrc.connect("need-data", self._on_need_data)

    def _on_need_data(self, src, _length):
        """
        Called by GStreamer (in its streaming thread) whenever appsrc needs
        more data.  We dequeue one frame and push it with the correct PTS.
        """
        try:
            item = self._queue.get(timeout=2.0)
        except queue.Empty:
            # No frame arrived within 2 s; signal end-of-stream.
            src.emit("end-of-stream")
            return

        if item is _SENTINEL:
            src.emit("end-of-stream")
            return

        pts_ns, rgb_bytes = item
        buf = Gst.Buffer.new_wrapped(rgb_bytes)
        buf.pts = pts_ns  # equivalent to gst_buffer_set_pts()
        buf.duration = self._frame_duration_ns
        src.emit("push-buffer", buf)


def _faery_producer(
    frame_queue: queue.Queue,
    fps: float,
    tau: str,
    colormap_name: str,
    driver: str,
    stop_event: threading.Event,
) -> None:
    """
    Runs the faery camera → regularize → render pipeline in a background
    thread and pushes (pts_ns, rgb_bytes) tuples onto frame_queue.

    Pushes _SENTINEL when the stream ends or stop_event is set.
    """
    colormap = getattr(faery.colormaps, colormap_name, None)
    if colormap is None:
        sys.stderr.write(
            f"Unknown colormap '{colormap_name}'. "
            "See python/faery/colormaps/ for available names.\n"
        )
        frame_queue.put(_SENTINEL)
        return

    try:
        rendered = (
            faery.events_stream_from_camera(driver=driver)
            .regularize(frequency_hz=float(fps))
            .render(
                decay="exponential",
                tau=tau,
                colormap=colormap,
            )
        )
        for frame in rendered:
            if stop_event.is_set():
                break
            # frame.t is in microseconds; GStreamer expects nanoseconds.
            pts_ns = frame.t.microseconds * 1000
            rgb_bytes = frame.pixels[:, :, :3].tobytes()
            # block=True on appsrc provides natural backpressure: put() will
            # block here if the GStreamer pipeline is running behind, which
            # prevents the queue from growing without bound.
            frame_queue.put((pts_ns, rgb_bytes))
    except Exception as exc:
        sys.stderr.write(f"Faery producer error: {exc}\n")
    finally:
        frame_queue.put(_SENTINEL)


def _detect_dimensions(driver: str) -> tuple:
    """
    Open the camera just long enough to read its sensor resolution, then
    return (width, height).  The camera connection is released immediately.
    """
    stream = faery.events_stream_from_camera(driver=driver)
    return stream.dimensions()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Stream an event camera as a live RTSP H.264 feed using "
            "faery + GStreamer RTSP server."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="IP address the RTSP server listens on (0.0.0.0 = all interfaces)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8554,
        help="TCP port for the RTSP server",
    )
    parser.add_argument(
        "--mount",
        default="/stream",
        help="RTSP mount point (must start with /)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Target render frame rate in Hz",
    )
    parser.add_argument(
        "--tau",
        default="00:00:00.100000",
        help="Exponential decay time constant (timecode hh:mm:ss.µµµµµµ)",
    )
    parser.add_argument(
        "--colormap",
        default="starry_night",
        help=(
            "Faery colormap name (e.g. starry_night, managua, bwr). "
            "Must be an attribute of faery.colormaps."
        ),
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help=(
            "Frame width in pixels. "
            "Auto-detected from the camera sensor if omitted."
        ),
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help=(
            "Frame height in pixels. "
            "Auto-detected from the camera sensor if omitted."
        ),
    )
    parser.add_argument(
        "--hw-encoder",
        action="store_true",
        help=(
            "Use the Jetson hardware H.264 encoder (nvv4l2h264enc). "
            "Omit to use the software encoder (x264enc)."
        ),
    )
    parser.add_argument(
        "--driver",
        choices=["Auto", "EventCameraDrivers", "NeuromorphicDrivers"],
        default="Auto",
        help="Camera driver to use",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # Resolve sensor dimensions                                            #
    # ------------------------------------------------------------------ #
    if args.width is None or args.height is None:
        sys.stderr.write("Auto-detecting camera sensor dimensions...\n")
        detected_w, detected_h = _detect_dimensions(args.driver)
        width = args.width if args.width is not None else detected_w
        height = args.height if args.height is not None else detected_h
        sys.stderr.write(f"  Detected: {width} x {height}\n\n")
    else:
        width = args.width
        height = args.height

    # ------------------------------------------------------------------ #
    # GStreamer init                                                        #
    # ------------------------------------------------------------------ #
    Gst.init(None)

    # ------------------------------------------------------------------ #
    # Shared frame queue (small — we want low latency, not buffering)      #
    # ------------------------------------------------------------------ #
    frame_queue: queue.Queue = queue.Queue(maxsize=4)
    stop_event = threading.Event()

    # ------------------------------------------------------------------ #
    # Faery producer thread                                                #
    # ------------------------------------------------------------------ #
    producer = threading.Thread(
        target=_faery_producer,
        args=(
            frame_queue,
            args.fps,
            args.tau,
            args.colormap,
            args.driver,
            stop_event,
        ),
        daemon=True,
        name="faery-producer",
    )
    producer.start()

    # ------------------------------------------------------------------ #
    # RTSP server                                                          #
    # ------------------------------------------------------------------ #
    server = GstRtspServer.RTSPServer()
    server.set_address(args.host)
    server.set_service(str(args.port))

    factory = _FaeryRtspFactory(
        frame_queue=frame_queue,
        width=width,
        height=height,
        fps=args.fps,
        use_hw_encoder=args.hw_encoder,
    )
    server.get_mount_points().add_factory(args.mount, factory)
    server.attach(None)

    rtsp_url = f"rtsp://{args.host}:{args.port}{args.mount}"
    sys.stderr.write(
        f"RTSP stream running at:\n"
        f"  {rtsp_url}\n\n"
        f"Connect with:\n"
        f"  ffplay  {rtsp_url}\n"
        f"  vlc     {rtsp_url}\n"
        f"  gst-launch-1.0 rtspsrc location={rtsp_url} ! "
        f"decodebin ! videoconvert ! autovideosink\n\n"
        f"Press Ctrl+C to stop.\n\n"
    )

    # ------------------------------------------------------------------ #
    # GLib main loop (drives the RTSP server)                             #
    # ------------------------------------------------------------------ #
    loop = GLib.MainLoop()

    def _shutdown(*_args):
        sys.stderr.write("\nShutting down...\n")
        stop_event.set()
        loop.quit()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # Keep Python's signal handler alive while blocked inside loop.run().
    GLib.timeout_add(200, lambda: True)

    try:
        loop.run()
    except KeyboardInterrupt:
        _shutdown()
    finally:
        stop_event.set()
        producer.join(timeout=3.0)


if __name__ == "__main__":
    main()
