import faery

(
    faery.events_stream_from_udp(
        dimensions=(1280, 720),
        address=("127.0.0.1", 29999),
        format="t32_x16_y15_on1",
    )
    .regularize(frequency_hz=60)
    .render(
        tau="00:00:00.2",
        decay="exponential",
    )
    .view()
)
