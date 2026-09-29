import collections.abc
import dataclasses
import importlib.util
import logging
import threading
import typing

import numpy as np

import faery.events_stream as events_stream


@dataclasses.dataclass(frozen=True)
class NeuromorphicDeviceMetadata:
    """Immutable description of the neuromorphic-drivers device in use."""

    name: typing.Any
    connection: typing.Any
    address: str
    serial: typing.Optional[str]
    properties: typing.Any
    configuration_type: str

    @property
    def dimensions(self) -> tuple[int, int]:
        return (int(self.properties.width), int(self.properties.height))


@dataclasses.dataclass(frozen=True)
class NeuromorphicDeviceCapabilities:
    """Capabilities that callers can use to construct device controls."""

    supports_configuration_updates: bool
    configuration_updates_are_thread_safe: bool
    configuration_updates_are_queued: bool
    supports_biases: bool
    biases_bounds: typing.Optional[typing.Any]
    last_configuration_error: typing.Optional[str] = None


def _device_configuration_type(name: typing.Any) -> str:
    """Return neuromorphic-drivers' configuration discriminator for *name*."""

    return str(name.name).lower()


def _configuration_type(configuration: typing.Any) -> str:
    configuration_type = getattr(configuration, "type", None)
    if not callable(configuration_type):
        raise TypeError(
            "neuromorphic-drivers configurations must provide a type() method"
        )
    return str(configuration_type())


def has_event_camera_drivers():
    return importlib.util.find_spec("event_camera_drivers") is not None


def has_neuromorphic_drivers():
    return importlib.util.find_spec("neuromorphic_drivers") is not None


class EventCameraDriverStream(events_stream.EventsStream):
    def __init__(
        self,
        manufacturer: typing.Optional[typing.Literal["Prophesee", "Inivation"]] = None,
        buffer_size: int = 1024,
    ):
        """Create an events stream using the event-camera-drivers library

        Args:
            manufacturer (Optional[typing.Literal["Prophesee", "Inivation"]]): Camera manufacturer.
                Defaults to automatic detection which might take some time.
            buffer_size (int): The size of the buffer to use for the event stream, defaults to 1024

        Returns:
            An (infinite) event stream from the camera

        Usage:
            >>> stream = EventCameraStream() # Open camera (will fail if no camera is connected)
            >>> stream.map(...)              # Use the stream as any other (infinite) event stream
        """
        try:
            import event_camera_drivers as evd  # type: ignore

            self.camera = evd.InivationCamera(buffer_size=buffer_size)
        except ImportError as e:
            logging.error(
                "The event_camera_drivers library is not available, please install"
            )
            raise e
        except ValueError as e:
            logging.info("No camera found using libcaer")
            raise e

    def __iter__(self) -> collections.abc.Iterator[np.ndarray]:
        logging.info(f"Starting streaming from event_camera_drivers: {self.camera}")
        while self.camera.is_running():
            v = next(self.camera)
            yield v

    def dimensions(self):
        return self.camera.resolution()


class NeuromorphicCameraStream(events_stream.EventsStream):
    def __init__(
        self,
        configuration: typing.Optional[typing.Any] = None,
        serial: typing.Optional[str] = None,
        iterator_timeout: typing.Optional[float] = None,
        raw: bool = False,
        ring_configuration: typing.Optional[typing.Any] = None,
        address: typing.Optional[str] = None,
        iterator_maximum_raw_packets: int = 64,
        **kwargs: typing.Any,
    ):
        """Create an events stream using the neuromorphic-drivers library.

        Args:
            configuration: Device-specific configuration object.  Can be one of:
                - neuromorphic_drivers.prophesee_evk4.Configuration
                - neuromorphic_drivers.prophesee_evk3_hd.Configuration
                - neuromorphic_drivers.inivation_dvxplorer.Configuration
                - neuromorphic_drivers.inivation_davis346.Configuration
                - neuromorphic_drivers.lucid_triton.Configuration
                - None for auto-detection with defaults
            serial: Serial number to open a specific device.
            iterator_timeout: Timeout in seconds for non-blocking iteration.
                If None, iteration blocks until data is available.
            raw: If True, return raw bytes instead of parsed packets.
            ring_configuration: Software ring-buffer configuration. The type
                should match the selected camera.
            address: IPv4 address used to select an Ethernet camera such as the
                Lucid Triton.
            iterator_maximum_raw_packets: Maximum number of driver packets to
                merge into one yielded packet.

        Usage:
            >>> # Simple usage with defaults
            >>> stream = NeuromorphicCameraStream()

            >>> # With custom configuration for EVK4
            >>> import neuromorphic_drivers as nd
            >>> config = nd.prophesee_evk4.Configuration()
            >>> config.biases.diff_on = 0x55  # Adjust sensitivity
            >>> stream = NeuromorphicCameraStream(configuration=config)

            >>> # Update configuration while streaming (from another thread)
            >>> stream.update_configuration(new_config)
        """
        try:
            import neuromorphic_drivers as nd

            self.nd = nd
            self.device_list = nd.list_devices()
            if len(self.device_list) == 0:
                raise RuntimeError(
                    "No event camera found, did you plug it in and install the udev rules?"
                )

            if iterator_timeout is not None and iterator_timeout < 0.0:
                raise ValueError("iterator_timeout must be greater than or equal to 0")
            if iterator_maximum_raw_packets < 1:
                raise ValueError("iterator_maximum_raw_packets must be at least 1")

            requested_configuration_type = (
                None if configuration is None else _configuration_type(configuration)
            )
            candidates = list(self.device_list)
            if serial is not None:
                candidates = [
                    device for device in candidates if device.serial == serial
                ]
                if not candidates:
                    raise ValueError(
                        f"No neuromorphic camera has serial number {serial!r}"
                    )
            if address is not None:
                candidates = [
                    device for device in candidates if device.address == address
                ]
                if not candidates:
                    raise ValueError(
                        "No neuromorphic camera matches both the requested device "
                        f"selectors (address={address!r}, serial={serial!r})"
                    )
            if requested_configuration_type is not None:
                matching_configuration = [
                    device
                    for device in candidates
                    if _device_configuration_type(device.name)
                    == requested_configuration_type
                ]
                if not matching_configuration:
                    selected_types = sorted(
                        {
                            _device_configuration_type(device.name)
                            for device in candidates
                        }
                    )
                    raise ValueError(
                        "Configuration type "
                        f"{requested_configuration_type!r} does not match the selected "
                        f"camera type(s): {', '.join(selected_types)}"
                    )
                candidates = matching_configuration

            usable_candidates = [
                device for device in candidates if device.error is None
            ]
            if not usable_candidates:
                errors = "; ".join(str(device.error) for device in candidates)
                raise RuntimeError(
                    f"The selected event camera is unavailable: {errors}"
                )

            selected = usable_candidates[0]
            selected_configuration_type = _device_configuration_type(selected.name)
            properties = nd.name_to_properties(selected.name)
            self._selected_device = NeuromorphicDeviceMetadata(
                name=selected.name,
                connection=selected.connection,
                address=selected.address,
                serial=selected.serial,
                properties=properties,
                configuration_type=selected_configuration_type,
            )

            device_module = getattr(nd, selected_configuration_type, None)
            self._supports_biases = getattr(device_module, "Biases", None) is not None
            if ring_configuration is not None:
                expected_ring_type = getattr(
                    device_module,
                    "RingConfiguration",
                    None,
                )
                if expected_ring_type is None:
                    raise RuntimeError(
                        "neuromorphic-drivers does not expose RingConfiguration "
                        f"for {selected_configuration_type}"
                    )
                if not isinstance(ring_configuration, expected_ring_type):
                    raise TypeError(
                        f"{selected_configuration_type} requires ring_configuration "
                        f"of type {expected_ring_type.__module__}."
                        f"{expected_ring_type.__name__}; got "
                        f"{type(ring_configuration).__module__}."
                        f"{type(ring_configuration).__name__}"
                    )

            # Store configuration parameters for use in __iter__.  If a
            # configuration selected a camera other than the first listed
            # camera, resolve an identifier so nd.open() selects the same one.
            open_serial = serial
            open_address = address
            if (
                open_serial is None
                and open_address is None
                and selected is not self.device_list[0]
            ):
                if selected.serial is not None:
                    open_serial = selected.serial
                elif selected.connection.name == "ETHERNET":
                    open_address = selected.address

            self._configuration = configuration
            self._ring_configuration = ring_configuration
            self._serial = open_serial
            self._address = open_address
            self._iterator_timeout = iterator_timeout
            self._iterator_maximum_raw_packets = iterator_maximum_raw_packets
            self._raw = raw
            self._kwargs = kwargs or {}

            # Driver calls remain on the iterator's owner thread.  Other
            # threads only exchange immutable metadata or pending configs.
            self._state_lock = threading.RLock()
            self._device: typing.Optional[typing.Any] = None
            self._pending_configuration: typing.Optional[typing.Any] = None
            self._biases_bounds: typing.Optional[typing.Any] = None
            self._last_configuration_error: typing.Optional[str] = None

        except ImportError as e:
            logging.error(
                "The neuromorphic_drivers library is not available, please install"
            )
            raise e

    @property
    def configuration_updates_are_thread_safe(self) -> bool:
        """Whether controllers may submit updates from non-owner threads."""

        return True

    @property
    def configuration_updates_are_queued(self) -> bool:
        """Whether update_configuration defers all hardware I/O to iteration."""

        return True

    @property
    def selected_device(self) -> NeuromorphicDeviceMetadata:
        """The camera selected when this stream was constructed."""

        return self._selected_device

    @property
    def active_device(self) -> typing.Optional[NeuromorphicDeviceMetadata]:
        """An immutable device snapshot while iteration is active, else None."""

        with self._state_lock:
            return self._selected_device if self._device is not None else None

    @property
    def capabilities(self) -> NeuromorphicDeviceCapabilities:
        """Return a thread-safe snapshot of live configuration capabilities."""

        with self._state_lock:
            return NeuromorphicDeviceCapabilities(
                supports_configuration_updates=True,
                configuration_updates_are_thread_safe=(
                    self.configuration_updates_are_thread_safe
                ),
                configuration_updates_are_queued=(
                    self.configuration_updates_are_queued
                ),
                supports_biases=self._supports_biases,
                biases_bounds=self._biases_bounds,
                last_configuration_error=self._last_configuration_error,
            )

    @property
    def last_configuration_error(self) -> typing.Optional[str]:
        """Return the most recent device-side update error, if any."""

        with self._state_lock:
            return self._last_configuration_error

    def biases_bounds(self) -> typing.Any:
        """Return bias bounds cached from the active camera.

        The driver is queried when iteration opens the device, keeping all
        hardware calls on the acquisition thread.  Callers may safely read the
        immutable result from any thread after iteration has started.
        """

        with self._state_lock:
            if self._device is None:
                raise RuntimeError(
                    "Cannot read bias bounds: stream is not currently iterating"
                )
            if not self._supports_biases:
                raise RuntimeError("The active camera does not expose bias controls")
            if self._biases_bounds is None:
                raise RuntimeError("Bias bounds are unavailable for the active camera")
            return self._biases_bounds

    def __iter__(self) -> collections.abc.Iterator[np.ndarray]:
        logging.info(
            "Starting streaming from neuromorphic_drivers: %s",
            self._selected_device,
        )
        with self.nd.open(
            configuration=self._configuration,
            ring_configuration=self._ring_configuration,
            serial=self._serial,
            address=self._address,
            iterator_timeout=self._iterator_timeout,
            iterator_maximum_raw_packets=self._iterator_maximum_raw_packets,
            raw=self._raw,
            **self._kwargs,
        ) as device:
            active_name = device.name()
            if active_name != self._selected_device.name:
                raise RuntimeError(
                    "neuromorphic-drivers opened an unexpected camera: "
                    f"expected {self._selected_device.name!r}, got {active_name!r}"
                )

            biases_bounds = None
            if self._supports_biases:
                try:
                    biases_bounds = device.biases_bounds()
                except Exception as error:
                    logging.warning("Could not read camera bias bounds: %s", error)

            with self._state_lock:
                self._device = device
                self._biases_bounds = biases_bounds
            try:
                # A controller may have queued a value after constructing the
                # stream but before its iterator entered the driver context.
                self._apply_pending_configuration_safely(device)
                for _status, packet in device:
                    self._apply_pending_configuration_safely(device)

                    # Timed neuromorphic-drivers iterators emit packet=None
                    # when their timeout expires.  The iteration is a useful
                    # opportunity to apply controls but does not contain data.
                    if packet is None:
                        continue

                    if self._raw:
                        yield packet
                    else:
                        events = packet.polarity_events
                        if events is not None:
                            yield events.astype(events_stream.EVENTS_DTYPE)
            finally:
                with self._state_lock:
                    if self._device is device:
                        self._device = None

    def _apply_pending_configuration_safely(self, device: typing.Any) -> None:
        try:
            self._apply_pending_configuration(device)
        except Exception as error:
            # A rejected live value must not tear down acquisition or a stream
            # consuming it.  Bounds-aware callers should prevent most
            # rejections; retain the message for diagnostics and continue with
            # the last device state.
            with self._state_lock:
                self._last_configuration_error = str(error)
            logging.exception(
                "Live camera configuration was rejected; streaming continues"
            )

    def _apply_pending_configuration(self, device: typing.Any) -> None:
        with self._state_lock:
            configuration = self._pending_configuration
            self._pending_configuration = None

        if configuration is None:
            return

        device.update_configuration(configuration)
        with self._state_lock:
            self._last_configuration_error = None

    def update_configuration(self, configuration: typing.Any) -> None:
        """Update the device configuration while streaming.

        This method is thread-safe and queue-only: it does not call the
        hardware.  It may be called before iteration starts.  The newest
        queued configuration is applied by the device's owner thread when the
        camera opens or on its next driver iteration.  A finite driver iterator
        timeout therefore permits later updates even when no event packet
        arrives.

        Args:
            configuration: Device-specific configuration object matching the device type.
        """
        configuration_type = _configuration_type(configuration)
        if configuration_type != self._selected_device.configuration_type:
            raise ValueError(
                f"Configuration type {configuration_type!r} does not match selected "
                f"camera type {self._selected_device.configuration_type!r}"
            )
        with self._state_lock:
            self._pending_configuration = configuration

    def dimensions(self) -> tuple[int, int]:
        return self._selected_device.dimensions


def events_stream_from_camera(
    driver: typing.Optional[
        typing.Literal["EventCameraDrivers", "NeuromorphicDrivers", "Auto"]
    ] = None,
    manufacturer: typing.Optional[typing.Literal["Inivation", "Prophesee"]] = None,
    buffer_size: int = 1024,
    # New neuromorphic-drivers specific parameters
    nd_configuration: typing.Optional[typing.Any] = None,
    nd_serial: typing.Optional[str] = None,
    nd_iterator_timeout: typing.Optional[float] = None,
    nd_ring_configuration: typing.Optional[typing.Any] = None,
    nd_address: typing.Optional[str] = None,
    nd_iterator_maximum_raw_packets: int = 64,
    **nd_open_kwargs,
) -> events_stream.EventsStream:
    stream = None
    error = None
    if driver is None or driver == "EventCameraDrivers" or driver == "Auto":
        try:
            stream = EventCameraDriverStream(
                manufacturer=manufacturer, buffer_size=buffer_size
            )
        except Exception as e:
            error = e
    if driver is None or driver == "NeuromorphicDrivers" or driver == "Auto":
        try:
            stream = NeuromorphicCameraStream(
                configuration=nd_configuration,
                ring_configuration=nd_ring_configuration,
                serial=nd_serial,
                address=nd_address,
                iterator_timeout=nd_iterator_timeout,
                iterator_maximum_raw_packets=nd_iterator_maximum_raw_packets,
                **nd_open_kwargs,
            )
        except Exception as e:
            error = e

    if stream is None:
        raise ValueError("No Event Camera found:", error)

    return stream
