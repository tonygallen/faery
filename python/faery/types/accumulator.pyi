import types
import typing

import numpy
import numpy.typing

class Accumulator:
    def __init__(
        self,
        dimensions: tuple[int, int],
        decay: typing.Literal["none", "linear", "exponential", "step"] = "exponential",
        decay_param: float = 1_000_000.0,
        min_potential: float = 0.0,
        max_potential: float = 1.0,
        neutral_potential: float = 0.5,
        event_contribution: float = 0.15,
        ignore_polarity: bool = False,
        synchronous_decay: bool = False,
        colormap_rgba: typing.Optional[numpy.typing.NDArray[numpy.float64]] = None,
    ): ...
    def __enter__(self) -> Accumulator: ...
    def __exit__(
        self,
        exception_type: typing.Optional[typing.Type[BaseException]],
        value: typing.Optional[BaseException],
        traceback: typing.Optional[types.TracebackType],
    ) -> bool: ...
    def accept(self, events: numpy.ndarray) -> None: ...
    def generate_frame(
        self, frame_t: typing.Optional[int] = None
    ) -> numpy.typing.NDArray[numpy.uint8]: ...
    def accumulate(
        self, events: numpy.ndarray, frame_t: int
    ) -> numpy.typing.NDArray[numpy.uint8]: ...
    def clear(self) -> None: ...
