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
        ┌──────────────────┐               ┌──────────────────────────────────────┐
        │ events_stream_    │               │ appsrc (block=false)                 │
        │ from_camera()     │               │  → leaky queue                       │
        │ .regularize()     │  queue.Queue  │  → videoconvert                      │
        │ .render()         │──────────────▶│  → video/x-raw,format=I420           │
        └──────────────────┘               │  → nvvidconv (hw) / passthrough (sw) │
                                           │  → nvv4l2h264enc / x264enc           │
                                           │  → h264parse                         │
                                           │  → rtph264pay → RTSPServer           │
                                           └──────────────────────────────────────┘

Pipeline details
----------------
The key fix over a naïve `appsrc → videoconvert → x264enc` pipeline is the
explicit `video/x-raw,format=I420` caps filter after videoconvert.  Without
it, GStreamer's caps negotiation leaves the format as RGB (4:4:4), which
causes ``x264 [error]: baseline profile doesn't support 4:4:4``.

Software encoder pipeline (default)::

    appsrc name=src is-live=true block=false format=time
        caps=video/x-raw,format=RGB,width=W,height=H,framerate=FPS/1 !
    queue max-size-buffers=10 leaky=downstream !
    videoconvert !
    video/x-raw,format=I420 !
    x264enc tune=zerolatency speed-preset=ultrafast !
    h264parse !
    video/x-h264,alignment=au,stream-format=byte-stream !
    rtph264pay name=pay0 pt=96

Hardware encoder pipeline (--hw-encoder, Jetson)::

    appsrc name=src is-live=true block=false format=time
        caps=video/x-raw,format=RGB,width=W,height=H,framerate=FPS/1 !
    queue max-size-buffers=10 max-size-time=0 max-size-bytes=0 leaky=downstream !
    videoconvert !
    video/x-raw,format=I420 !
    nvvidconv !
    video/x-raw(memory:NVMM),format=I420,framerate=FPS/1 !
    nvv4l2h264enc bitrate=BITRATE profile=0 preset-level=2 maxperf-enable=true !
    h264parse !
    video/x-h264,alignment=au,stream-format=byte-stream !
    rtph264pay name=pay0 pt=96

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
  --bitrate BPS      Target bitrate in bits/s for the hw encoder
                     (default: 20000000 = 20 Mbit/s)
  --driver DRIVER    Camera driver: Auto | EventCameraDrivers |
                     NeuromorphicDrivers (default: Auto)

Dependencies
------------
Install once on Ubuntu / Jetson:

    sudo apt install python3-gi gir1.2-gst-rtsp-server-1.0 \\
                     gstreamer1.0-rtsp gstreamer1.0-plugins-good \\
                     gstreamer1.0-plugins-ugly gstreamer1.0-plugins-bad \\
                     gstreamer1.0-plugins-base gstreamer1.0-tools

For the hardware encoder path (--hw-encoder), GStreamer Jetson packages
(libgstreamer-plugins-nvvidconv, gstreamer1.0-nvvidconv, etc.) must be
installed — these are pre-installed on Jetson JetPack images.

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


def _build_pipeline(
    width: int,
    height: int,
    fps: int,
    use_hw_encoder: bool,
    bitrate: int,
) -> str:
    """
    Return the GStreamer pipeline description string for the RTSP factory.

    The explicit ``video/x-raw,format=I420`` caps filter after videoconvert is
    required so that GStreamer negotiates I420 (4:2:0) before handing frames to
    the H.264 encoder.  Without it, the encoder receives RGB (4:4:4) and
    baseline H.264 rejects it with::

        x264 [error]: baseline profile doesn't support 4:4:4

    ``block=false`` on appsrc combined with a leaky downstream queue lets
    GStreamer drop frames under backpressure rather than stalling the faery
    producer thread.
    """
    src_caps = (
        f"video/x-raw,format=RGB,"
        f"width={width},height={height},"
        f"framerate={fps}/1"
    )

    if use_hw_encoder:
        # Jetson hardware path:
        #   videoconvert → I420 → nvvidconv (NVMM) → nvv4l2h264enc
        # nvvidconv moves data to NVMM memory so nvv4l2h264enc can access it
        # without a CPU round-trip.
        encoder_chain = (
            f"videoconvert ! "
            f"video/x-raw,format=I420 ! "
            f"nvvidconv ! "
            f"video/x-raw(memory:NVMM),format=I420,framerate={fps}/1 ! "
            f"nvv4l2h264enc "
            f"bitrate={bitrate} profile=0 preset-level=2 maxperf-enable=true ! "
            f"h264parse ! "
            f"video/x-h264,alignment=au,stream-format=byte-stream"
        )
    else:
        # Software path: videoconvert → I420 → x264enc
        # The explicit I420 caps filter prevents the baseline-profile 4:4:4 error.
        encoder_chain = (
            f"videoconvert ! "
            f"video/x-raw,format=I420 ! "
            f"x264enc tune=zerolatency speed-preset=ultrafast ! "
            f"h264parse ! "
            f"video/x-h264,alignment=au,stream-format=byte-stream"
        )

    return (
        f"( appsrc name=src is-live=true block=false format=time "
        f"caps={src_caps} ! "
        # max-size-time=0 and max-size-bytes=0 disable those limits so only
        # the buffer count (max-size-buffers) is active.  leaky=downstream
        # drops the oldest buffer when the queue is full, preventing the
        # faery producer from stalling if the encoder falls behind.
        f"queue max-size-buffers=10 max-size-time=0 max-size-bytes=0 "
        f"leaky=downstream ! "
        f"{encoder_chain} ! "
        f"rtph264pay name=pay0 pt=96 )"
    )


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
        bitrate: int,
    ):
        super().__init__()
        self._queue = frame_queue
        self._frame_duration_ns = int(1_000_000_000 / fps)

        pipeline = _build_pipeline(
            width=width,
            height=height,
            fps=fps,
            use_hw_encoder=use_hw_encoder,
            bitrate=bitrate,
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
            # appsrc uses block=false with a leaky queue, so put() must not
            # block here — use a short timeout and skip the frame on overflow
            # rather than stalling the producer indefinitely.
            try:
                frame_queue.put((pts_ns, rgb_bytes), timeout=0.5)
            except queue.Full:
                pass  # leaky queue in GStreamer will drop; skip here too
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
            "Use the Jetson hardware H.264 encoder (nvv4l2h264enc) via nvvidconv. "
            "Omit to use the software encoder (x264enc)."
        ),
    )
    parser.add_argument(
        "--bitrate",
        type=int,
        default=20_000_000,
        help=(
            "Target bitrate in bits/s for the hardware encoder (--hw-encoder). "
            "Ignored for the software encoder."
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
        try:
            detected_w, detected_h = _detect_dimensions(args.driver)
        except Exception as exc:
            sys.stderr.write(
                f"ERROR: Could not detect camera sensor dimensions: {exc}\n"
                "Please supply --width and --height explicitly, e.g.:\n"
                "  --width 640 --height 480   (Inivation DVXplorer)\n"
                "  --width 346 --height 260   (Inivation DAVIS346)\n"
                "  --width 1280 --height 720  (Prophesee EVK4)\n"
            )
            sys.exit(1)
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
    # Shared frame queue                                                   #
    # appsrc uses block=false with a leaky downstream queue inside        #
    # GStreamer, so the Python-side queue just needs to be large enough   #
    # to smooth over thread scheduling jitter.                            #
    # ------------------------------------------------------------------ #
    frame_queue: queue.Queue = queue.Queue(maxsize=10)
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
        bitrate=args.bitrate,
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
    # GLib's main loop runs in C and doesn't return to Python between
    # iterations, so SIGINT would never be delivered.  A short periodic
    # timeout forces the loop to yield to the Python interpreter regularly,
    # allowing the signal handler registered above to execute.
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
