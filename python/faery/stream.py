import collections
import collections.abc
import threading
import typing

from . import enums, timestamp

ItemType = typing.TypeVar("ItemType")
BranchType = typing.TypeVar("BranchType")
TeeOverflowConfig = typing.Union[
    enums.TeeOverflow,
    collections.abc.Sequence[enums.TeeOverflow],
]


def _normalize_tee_overflow(
    overflow: TeeOverflowConfig,
    n: int,
) -> tuple[enums.TeeOverflow, ...]:
    if isinstance(overflow, str):
        policies = (overflow,) * n
    else:
        try:
            policies = tuple(overflow)
        except TypeError as error:
            raise ValueError(
                "overflow must be 'block', 'drop_oldest', or a sequence of "
                "one policy per branch"
            ) from error
        if len(policies) != n:
            raise ValueError(
                "an overflow policy sequence must have one item per branch"
            )

    for policy in policies:
        if policy not in ("block", "drop_oldest"):
            raise ValueError("overflow policies must be 'block' or 'drop_oldest'")
    return typing.cast(tuple[enums.TeeOverflow, ...], policies)


class _TeeState(typing.Generic[ItemType]):
    """Shared state for a group of tee branches."""

    def __init__(
        self,
        parent: collections.abc.Iterable[ItemType],
        n: int,
        maxsize: typing.Optional[int],
        retain_item: collections.abc.Callable[[ItemType], ItemType],
        overflow: TeeOverflowConfig = "block",
    ):
        if isinstance(n, bool) or not isinstance(n, int) or n < 2:
            raise ValueError("n must be an integer greater than or equal to 2")
        if maxsize is not None and (
            isinstance(maxsize, bool) or not isinstance(maxsize, int) or maxsize <= 0
        ):
            raise ValueError("maxsize must be a positive integer or None")
        overflow_policies = _normalize_tee_overflow(overflow=overflow, n=n)
        if maxsize is None and any(
            policy == "drop_oldest" for policy in overflow_policies
        ):
            raise ValueError("overflow='drop_oldest' requires a bounded maxsize")

        self._parent = parent
        self._maxsize = maxsize
        self._overflow_policies = overflow_policies
        self._retain_item = retain_item
        self._condition = threading.Condition()
        self._buffer: collections.deque[ItemType] = collections.deque()
        self._buffer_start = 0
        self._positions = [0 for _ in range(n)]
        self._dropped_packets = [0 for _ in range(n)]
        self._active = [True for _ in range(n)]
        self._producer: typing.Optional[threading.Thread] = None
        self._producer_started = False
        self._producer_finished = False
        self._upstream_iterator: typing.Optional[collections.abc.Iterator[ItemType]] = (
            None
        )
        self._upstream_close_claimed = False
        self._done = False
        self._cancel_requested = False
        self._failure: typing.Optional[BaseException] = None

    def start(self, branch_index: int) -> bool:
        """Start the producer for an active branch, at most once."""
        with self._condition:
            if not self._active[branch_index] or self._cancel_requested:
                return False
            if self._producer_started:
                return True
            self._producer_started = True
            self._producer = threading.Thread(
                target=self._run,
                name="faery-tee",
                daemon=True,
            )
            try:
                self._producer.start()
            except BaseException as error:
                self._producer = None
                self._failure = error
                self._done = True
                self._producer_finished = True
                self._condition.notify_all()
                raise
            return True

    def _run(self) -> None:
        upstream_iterator: typing.Optional[collections.abc.Iterator[ItemType]] = None
        try:
            with self._condition:
                if self._cancel_requested:
                    return
            upstream_iterator = iter(self._parent)
            with self._condition:
                self._upstream_iterator = upstream_iterator
                self._condition.notify_all()

            while True:
                with self._condition:
                    while (
                        not self._cancel_requested
                        and self._maxsize is not None
                        and len(self._buffer) >= self._maxsize
                        and self._blocking_branch_at_buffer_start_locked()
                    ):
                        self._condition.wait()
                    if self._cancel_requested:
                        break

                try:
                    item = next(upstream_iterator)
                except StopIteration:
                    with self._condition:
                        self._done = True
                        self._condition.notify_all()
                    break
                item = self._retain_item(item)

                with self._condition:
                    if self._cancel_requested:
                        break
                    while (
                        self._maxsize is not None
                        and len(self._buffer) >= self._maxsize
                    ):
                        self._drop_oldest_locked()
                        if len(self._buffer) >= self._maxsize:
                            # This is defensive: a blocking branch cannot move
                            # backwards while next(upstream) is in progress.
                            self._condition.wait()
                        if self._cancel_requested:
                            break
                    if self._cancel_requested:
                        break
                    self._buffer.append(item)
                    self._condition.notify_all()
        except BaseException as error:
            with self._condition:
                if not self._cancel_requested:
                    self._failure = error
                self._done = True
                self._condition.notify_all()
        finally:
            with self._condition:
                upstream_to_close = self._claim_upstream_close_locked()
            self._close_upstream(upstream_to_close)
            with self._condition:
                self._upstream_iterator = None
                self._done = True
                self._producer_finished = True
                self._condition.notify_all()

    @staticmethod
    def _close_upstream(
        upstream_iterator: typing.Optional[collections.abc.Iterator[ItemType]],
    ) -> bool:
        if upstream_iterator is None:
            return True
        close = getattr(upstream_iterator, "close", None)
        if close is not None:
            try:
                close()
            except BaseException:
                # Closing is a best-effort cancellation mechanism. In particular,
                # a Python generator raises ValueError if another thread is inside
                # next() when close() is called.
                return False
        return True

    def next(self, branch_index: int) -> ItemType:
        failure: typing.Optional[BaseException] = None
        with self._condition:
            while True:
                if not self._active[branch_index]:
                    raise StopIteration

                position = self._positions[branch_index]
                buffer_end = self._buffer_start + len(self._buffer)
                if position < buffer_end:
                    item = self._buffer[position - self._buffer_start]
                    self._positions[branch_index] = position + 1
                    self._reclaim_locked()
                    self._condition.notify_all()
                    return item

                if self._done:
                    failure = self._failure
                    break

                self._condition.wait()

        if failure is not None:
            if isinstance(failure, StopIteration):
                raise RuntimeError(
                    f"tee packet retention raised StopIteration: {failure}"
                ) from failure
            raise failure
        raise StopIteration

    def close_branch(self, branch_index: int) -> None:
        upstream_iterator: typing.Optional[collections.abc.Iterator[ItemType]] = None
        with self._condition:
            if not self._active[branch_index]:
                return
            self._active[branch_index] = False
            self._reclaim_locked()
            if not any(self._active):
                self._cancel_requested = True
                self._buffer.clear()
                upstream_iterator = self._claim_upstream_close_locked()
            self._condition.notify_all()
        self._close_upstream_async(upstream_iterator)

    def close(self) -> None:
        upstream_iterator: typing.Optional[collections.abc.Iterator[ItemType]] = None
        with self._condition:
            for index in range(len(self._active)):
                self._active[index] = False
            self._buffer.clear()
            self._cancel_requested = True
            upstream_iterator = self._claim_upstream_close_locked()
            self._condition.notify_all()

        self._close_upstream_async(upstream_iterator)
        producer = self._producer
        if producer is not None and producer is not threading.current_thread():
            # A native or otherwise blocking upstream iterator may not respond to
            # close(). Keep the producer daemonized and bound the cleanup wait.
            producer.join(timeout=1.0)

    def dropped_packets(self) -> tuple[int, ...]:
        """Return the number of packets dropped for each branch."""
        with self._condition:
            return tuple(self._dropped_packets)

    def _claim_upstream_close_locked(
        self,
    ) -> typing.Optional[collections.abc.Iterator[ItemType]]:
        if self._upstream_iterator is None or self._upstream_close_claimed:
            return None
        self._upstream_close_claimed = True
        return self._upstream_iterator

    def _close_upstream_async(
        self,
        upstream_iterator: typing.Optional[collections.abc.Iterator[ItemType]],
    ) -> None:
        if upstream_iterator is None:
            return
        closer = threading.Thread(
            target=self._close_upstream_after_cancel,
            args=(upstream_iterator,),
            name="faery-tee-close",
            daemon=True,
        )
        try:
            closer.start()
        except BaseException:
            # Let the producer claim closing responsibility if starting the
            # best-effort closer itself fails.
            with self._condition:
                if self._upstream_iterator is upstream_iterator:
                    self._upstream_close_claimed = False

    def _close_upstream_after_cancel(
        self,
        upstream_iterator: collections.abc.Iterator[ItemType],
    ) -> None:
        if self._close_upstream(upstream_iterator):
            return
        # Cross-thread generator.close() fails while next() is executing. If
        # that call eventually returns, retry once after the producer leaves
        # the iterator; this helper remains daemonized if next() never returns.
        with self._condition:
            while not self._producer_finished:
                self._condition.wait()
        self._close_upstream(upstream_iterator)

    def _reclaim_locked(self) -> None:
        active_positions = [
            position
            for position, active in zip(self._positions, self._active)
            if active
        ]
        if not active_positions:
            self._buffer.clear()
            return
        reclaim_count = min(active_positions) - self._buffer_start
        for _ in range(reclaim_count):
            self._buffer.popleft()
        self._buffer_start += reclaim_count

    def _drop_oldest_locked(self) -> None:
        """Advance lossy branches that prevent a full buffer from reclaiming."""
        for index, (position, active, overflow_policy) in enumerate(
            zip(self._positions, self._active, self._overflow_policies)
        ):
            if (
                active
                and overflow_policy == "drop_oldest"
                and position == self._buffer_start
            ):
                self._positions[index] = position + 1
                self._dropped_packets[index] += 1
        self._reclaim_locked()

    def _blocking_branch_at_buffer_start_locked(self) -> bool:
        return any(
            active
            and position == self._buffer_start
            and overflow_policy == "block"
            for position, active, overflow_policy in zip(
                self._positions,
                self._active,
                self._overflow_policies,
            )
        )


class _TeeBranchIterator(typing.Generic[ItemType]):
    def __init__(
        self,
        state: _TeeState[ItemType],
        branch_index: int,
        copy_item: collections.abc.Callable[[ItemType], ItemType],
    ):
        self._state = state
        self._branch_index = branch_index
        self._copy_item = copy_item
        self._started = False
        self._closed = False
        self._next_lock = threading.Lock()

    def __iter__(self) -> "_TeeBranchIterator[ItemType]":
        return self

    def __next__(self) -> ItemType:
        if not self._next_lock.acquire(blocking=False):
            raise RuntimeError("a tee branch cannot be consumed concurrently")
        try:
            if self._closed:
                raise StopIteration
            if not self._started:
                self._started = True
                try:
                    started = self._state.start(self._branch_index)
                except BaseException:
                    self.close()
                    raise
                if not started:
                    self.close()
                    raise StopIteration
            try:
                item = self._state.next(self._branch_index)
            except BaseException:
                self.close()
                raise
            try:
                return self._copy_item(item)
            except StopIteration as error:
                self.close()
                raise RuntimeError(
                    f"tee packet materialization raised StopIteration: {error}"
                ) from error
            except BaseException:
                self.close()
                raise
        finally:
            self._next_lock.release()

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._state.close_branch(self._branch_index)

    def __del__(self):
        try:
            self.close()
        except BaseException:
            pass


class _TeeBranch(typing.Generic[ItemType]):
    def __init__(
        self,
        state: _TeeState[ItemType],
        branch_index: int,
        copy_item: collections.abc.Callable[[ItemType], ItemType],
    ):
        self._state = state
        self._branch_index = branch_index
        self._copy_item = copy_item
        self._iterated = False
        self._iteration_lock = threading.Lock()

    def __iter__(self) -> collections.abc.Iterator[ItemType]:
        with self._iteration_lock:
            if self._iterated:
                raise RuntimeError("a tee branch can only be iterated once")
            self._iterated = True
        return _TeeBranchIterator(
            state=self._state,
            branch_index=self._branch_index,
            copy_item=self._copy_item,
        )

    def close(self) -> None:
        self._state.close_branch(self._branch_index)


class Tee(collections.abc.Sequence[BranchType], typing.Generic[BranchType]):
    """An unpackable collection of branches produced by ``stream.tee()``."""

    def __init__(
        self,
        branches: list[BranchType],
        state: _TeeState[typing.Any],
    ):
        self._branches = branches
        self._state = state

    def __len__(self) -> int:
        return len(self._branches)

    @typing.overload
    def __getitem__(self, index: int) -> BranchType: ...

    @typing.overload
    def __getitem__(self, index: slice) -> list[BranchType]: ...

    def __getitem__(
        self, index: typing.Union[int, slice]
    ) -> typing.Union[BranchType, list[BranchType]]:
        return self._branches[index]

    def __iter__(self) -> collections.abc.Iterator[BranchType]:
        return iter(self._branches)

    def close(self) -> None:
        """Close every branch and stop the producer when possible."""
        self._state.close()

    @property
    def dropped_packets(self) -> tuple[int, ...]:
        """Number of packets dropped for each branch, in branch order."""
        return self._state.dropped_packets()

    def __enter__(self) -> "Tee[BranchType]":
        return self

    def __exit__(self, exception_type, exception, traceback) -> None:
        self.close()


def _tee(
    parent: collections.abc.Iterable[ItemType],
    n: int,
    maxsize: typing.Optional[int],
    retain_item: collections.abc.Callable[[ItemType], ItemType],
    copy_item: collections.abc.Callable[[ItemType], ItemType],
    overflow: TeeOverflowConfig = "block",
) -> tuple[_TeeState[ItemType], list[_TeeBranch[ItemType]]]:
    """Create the generic state and cursors used by type-preserving tee adapters."""
    state = _TeeState(
        parent=parent,
        n=n,
        maxsize=maxsize,
        overflow=overflow,
        retain_item=retain_item,
    )
    return (
        state,
        [
            _TeeBranch(
                state=state,
                branch_index=branch_index,
                copy_item=copy_item,
            )
            for branch_index in range(n)
        ],
    )


class Stream(typing.Generic[ItemType]):
    """
    Generic base class for streams.

    This includes inputs (files, UDP sockets, cameras...) and filters.
    A stream generates packets of a given type (frame or events).
    """

    def __iter__(self) -> collections.abc.Iterator[ItemType]:
        raise NotImplementedError()

    def dimensions(self) -> tuple[int, int]:
        """
        Stream dimensions in pixels.

        Returns:
            tuple[int, int]: Width (left-right direction) and height (top-bottom direction) in pixels.
        """
        raise NotImplementedError()


class FiniteStream(Stream[ItemType]):
    """
    A stream whose source has a known begin and a known end.

    A file is a finite stream but a camera is not.
    """

    def time_range(self) -> tuple[timestamp.Time, timestamp.Time]:
        """
        Time of the stream's start and end.

        For instance, if the stream contains 3 events with timestamps `[2, 71, 828]`,
        the time range may be `(2 * faery.us, 829 * faery.us)`.
        It may also be wider, for instance `(0 * faery.us, 1000 * faery.us)`.

        Returns:
            tuple[int, int]: First and one-past-last times.
        """
        raise NotImplementedError()


class RegularStream(Stream[ItemType]):
    """
    A stream whose packets cover a fixed amount of time.

    Frame streams are always regular (under the assumption of constant frame rate)
    whereas event streams can be non-regular (variable packet duration).

    The packets of non-regular event streams may have a variable number of events and / or
    cover a variable amount of time.
    """

    def frequency_hz(self) -> float:
        """
        The stream's frequency (or packet rate), in Hertz.
        """
        raise NotImplementedError()


class FiniteRegularStream(FiniteStream[ItemType], RegularStream[ItemType]):
    """
    A stream that is finite and regular (see FiniteStream and RegularStream).
    """

    pass


class Filter(Stream[ItemType]):
    """
    A filter is a stream that consumes data from another stream.
    """

    def init(self, parent: Stream[ItemType]):
        """
        init plays the role of __init__ but can be called from
        an inherited class's __init__ without super().

        Filter-inherited classes are typically generated dynamically (see `events_filter.py`),
        which breaks super().__init__.
        """
        self.parent = parent

    def dimensions(self) -> tuple[int, int]:
        return self.parent.dimensions()


class FiniteFilter(FiniteStream[ItemType]):
    """
    A finite filter consumes data from a finite stream. It is thus a finite stream.
    """

    def init(self, parent: FiniteStream[ItemType]):
        """
        init plays the role of __init__ but can be called from
        an inherited class's __init__ without super().

        Filter-inherited classes are typically generated dynamically (see `events_filter.py`),
        which breaks super().__init__.
        """
        self.parent = parent

    def dimensions(self) -> tuple[int, int]:
        return self.parent.dimensions()

    def time_range(self) -> tuple[timestamp.Time, timestamp.Time]:
        return self.parent.time_range()


class RegularFilter(RegularStream[ItemType]):
    """
    A regular filter consumes data from a regular stream. It is thus a regular stream.
    """

    def init(self, parent: RegularStream[ItemType]):
        """
        init plays the role of __init__ but can be called from
        an inherited class's __init__ without super().

        Filter-inherited classes are typically generated dynamically (see `events_filter.py`),
        which breaks super().__init__.
        """
        self.parent = parent

    def dimensions(self) -> tuple[int, int]:
        return self.parent.dimensions()

    def frequency_hz(self) -> float:
        return self.parent.frequency_hz()


class FiniteRegularFilter(FiniteRegularStream[ItemType]):
    """
    A finite regular filter consumes data from a finite regular stream. It is thus a finite regular stream.
    """

    def init(self, parent: FiniteRegularStream[ItemType]):
        """
        init plays the role of __init__ but can be called from
        an inherited class's __init__ without super().

        Filter-inherited classes are typically generated dynamically (see `events_filter.py`),
        which breaks super().__init__.
        """
        self.parent = parent

    def dimensions(self) -> tuple[int, int]:
        return self.parent.dimensions()

    def time_range(self) -> tuple[timestamp.Time, timestamp.Time]:
        return self.parent.time_range()

    def frequency_hz(self) -> float:
        return self.parent.frequency_hz()
