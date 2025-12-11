import faery

(
    faery.events_stream_from_file(
        faery.dirname.parent / "tests" / "data" / "dvs.es",
    )
    .view_3d(
        time_window_us=200000,
        time_slices=40,
        colormap=faery.colormaps.managua,
    )
    .view()
)
