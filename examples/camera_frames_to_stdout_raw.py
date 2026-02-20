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
