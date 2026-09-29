import faery

(
    faery.events_stream_from_file(faery.dirname.parent / "tests" / "data" / "dvs.es")
    .regularize(frequency_hz=60.0)
    .accumulate(
        decay="exponential",
        decay_param=1_000_000.0,
        min_potential=0.0,
        max_potential=1.0,
        neutral_potential=0.5,
        event_contribution=0.15,
        ignore_polarity=False,
        synchronous_decay=False,
    )
    .to_file(faery.dirname.parent / "tests" / "data_generated" / "accumulator.mp4")
)
