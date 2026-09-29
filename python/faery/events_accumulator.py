import collections.abc
import types
import typing

import numpy

from . import color as color_module
from . import enums, events_stream, frame_stream, stream, timestamp

if typing.TYPE_CHECKING:
    from .types import accumulator  # type: ignore
else:
    from .extension import accumulator


def restrict(prefixes: set[typing.Literal["Regular", "FiniteRegular"]]):
    def decorator(method):
        method._prefixes = prefixes
        return method

    return decorator


FILTERS: dict[str, typing.Any] = {}


def typed_accumulator(
    prefixes: set[typing.Literal["Regular", "FiniteRegular"]],
):
    def decorator(accumulator_class):
        attributes = [
            name
            for name, item in accumulator_class.__dict__.items()
            if isinstance(item, types.FunctionType)
        ]
        for prefix in prefixes:

            class Generated(getattr(frame_stream, f"{prefix}FrameStream")):
                pass

            for attribute in attributes:
                method = getattr(accumulator_class, attribute)
                if not hasattr(method, "_prefixes") or prefix in method._prefixes:
                    setattr(Generated, attribute, method)
            Generated.__name__ = f"{prefix}{accumulator_class.__name__}"
            Generated.__qualname__ = Generated.__name__
            FILTERS[Generated.__name__] = Generated
        return None

    return decorator


@typed_accumulator({"Regular", "FiniteRegular"})
class Accumulate(frame_stream.FiniteRegularFrameStream):
    """Convert regular event packets into accumulated RGBA frames.

    Unlike :meth:`render`, which visualizes event recency (or independently
    tone-maps activity), this filter maintains one persistent potential per
    pixel. On-events add ``event_contribution``, off-events subtract it unless
    ``ignore_polarity`` is enabled, and potentials are clipped to the configured
    range before being normalized and mapped to RGBA frames.

    The parent must be regular because packet boundaries define frame times.
    Use ``events.regularize(frequency_hz=...)`` before this filter.

    Args:
        parent: A regular input event stream.
        decay: ``"none"``, ``"linear"``, ``"exponential"``, or ``"step"``.
        decay_param: Linear slope in potential per microsecond, or exponential
            time constant in microseconds. Ignored by none and step decay.
        min_potential: Minimum potential, mapped to the colormap's first color.
        max_potential: Maximum potential, mapped to the colormap's last color.
        neutral_potential: Initial potential and the linear/step reset target.
            As in dv-processing, exponential decay always approaches zero.
        event_contribution: Magnitude added or subtracted for each event.
        ignore_polarity: Treat off-events as positive contributions.
        synchronous_decay: Eagerly decay every pixel at each frame boundary.
            If false, a pixel is decayed lazily when it next receives an event.
        colormap: Maps normalized potentials to RGBA colors. Defaults to
            ``faery.colormaps.starry_night``.
    """

    def __init__(
        self,
        parent: stream.FiniteRegularStream[numpy.ndarray],
        decay: enums.AccumulatorDecay = "exponential",
        decay_param: float = 1_000_000.0,
        min_potential: float = 0.0,
        max_potential: float = 1.0,
        neutral_potential: float = 0.5,
        event_contribution: float = 0.15,
        ignore_polarity: bool = False,
        synchronous_decay: bool = False,
        colormap: color_module.Colormap = color_module.colormaps.starry_night,
    ):
        self.parent = parent
        self.decay: enums.AccumulatorDecay = enums.validate_accumulator_decay(decay)
        self.decay_param = decay_param
        self.min_potential = min_potential
        self.max_potential = max_potential
        self.neutral_potential = neutral_potential
        self.event_contribution = event_contribution
        self.ignore_polarity = ignore_polarity
        self.synchronous_decay = synchronous_decay
        self.colormap = colormap

    def dimensions(self) -> tuple[int, int]:
        return self.parent.dimensions()

    @restrict({"FiniteRegular"})
    def time_range(self) -> tuple[timestamp.Time, timestamp.Time]:
        return self.parent.time_range()

    def frequency_hz(self) -> float:
        return self.parent.frequency_hz()

    def __iter__(self) -> collections.abc.Iterator[frame_stream.Frame]:
        period_us = 1e6 / self.parent.frequency_hz()
        try:
            first_frame_t: typing.Optional[timestamp.Time] = self.parent.time_range()[0]
        except (AttributeError, NotImplementedError):
            first_frame_t = None
        frame_index = 0

        with accumulator.Accumulator(
            dimensions=self.dimensions(),
            decay=self.decay,
            decay_param=self.decay_param,
            min_potential=self.min_potential,
            max_potential=self.max_potential,
            neutral_potential=self.neutral_potential,
            event_contribution=self.event_contribution,
            ignore_polarity=self.ignore_polarity,
            synchronous_decay=self.synchronous_decay,
            colormap_rgba=self.colormap.rgba,
        ) as native_accumulator:
            for events in self.parent:
                if len(events) == 0:
                    frame_index += 1
                    if first_frame_t is None:
                        continue
                    frame_t = timestamp.Time(
                        microseconds=int(
                            round(
                                first_frame_t.to_microseconds()
                                + frame_index * period_us
                            )
                        )
                    )
                else:
                    if first_frame_t is None:
                        first_frame_t = timestamp.Time(
                            microseconds=int(
                                round(
                                    int(events["t"][0])
                                    + ((1 - frame_index) * period_us)
                                )
                            )
                        )
                        if frame_index > 0:
                            empty_events = numpy.array(
                                [], dtype=events_stream.EVENTS_DTYPE
                            )
                            for empty_frame_index in range(frame_index):
                                empty_frame_t = timestamp.Time(
                                    microseconds=int(
                                        round(
                                            first_frame_t.to_microseconds()
                                            + empty_frame_index * period_us
                                        )
                                    )
                                )
                                assert empty_frame_t >= timestamp.Time(microseconds=0)
                                yield frame_stream.Frame(
                                    t=empty_frame_t,
                                    pixels=native_accumulator.accumulate(
                                        events=empty_events,
                                        frame_t=empty_frame_t.to_microseconds(),
                                    ),
                                )
                    frame_index += 1
                    frame_t = timestamp.Time(
                        microseconds=int(
                            round(
                                first_frame_t.to_microseconds()
                                + frame_index * period_us
                            )
                        )
                    )
                yield frame_stream.Frame(
                    t=frame_t,
                    pixels=native_accumulator.accumulate(
                        events=events,
                        frame_t=frame_t.to_microseconds(),
                    ),
                )
