"""Record losslessly and view event stream with ``tee``."""

import argparse
import pathlib
import threading

import faery


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=pathlib.Path)
    parser.add_argument("output", type=pathlib.Path)
    arguments = parser.parse_args()

    worker_errors: list[BaseException] = []
    source = faery.events_stream_from_file(arguments.input)

    tee = source.tee(
        maxsize=100,
        # Recording backpressures; viewing discards stale packets if necessary.
        overflow=("block", "drop_oldest"),
    )
    with tee as (recording, viewing):

        def record() -> None:
            try:
                recording.to_file(arguments.output)
            except BaseException as error:
                worker_errors.append(error)
                # Wake the main-thread pipeline so it can join and re-raise.
                viewing.close()
            finally:
                recording.close()

        record_thread = threading.Thread(
            target=record,
            name="faery-recorder",
            daemon=False,
        )
        record_thread.start()
        try:
            (
                viewing.regularize(frequency_hz=30.0)
                .render(
                    decay="exponential",
                    tau="00:00:00.200000",
                    colormap=faery.colormaps.starry_night,
                )
                # The viewer stays on the main thread, as GUI platforms require.
                .view()
            )
        finally:
            # Closing both branches also handles a viewer closed before EOF.
            viewing.close()
            recording.close()
            record_thread.join()

    if worker_errors:
        raise worker_errors[0]

    print(f"viewer packets dropped: {tee.dropped_packets[1]}")


if __name__ == "__main__":
    main()
