import collections.abc
import importlib.util
import logging
import typing

import numpy as np

import faery.events_stream as events_stream


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
        usb_configuration: typing.Optional[typing.Any] = None,
        serial: typing.Optional[str] = None,
        iterator_timeout: typing.Optional[float] = None,
        raw: bool = False,
        **kwargs: typing.Any,
    ):
        """Create an events stream using the neuromorphic-drivers library.

        Args:
            configuration: Device-specific configuration object.  Can be one of:
                - neuromorphic_drivers.prophesee_evk4.Configuration
                - neuromorphic_drivers.prophesee_evk3_hd.Configuration
                - neuromorphic_drivers.inivation_dvxplorer.Configuration
                - neuromorphic_drivers.inivation_davis346.Configuration
                - None for auto-detection with defaults
            usb_configuration: USB configuration for buffer sizes and DMA settings.
            serial: Serial number to open a specific device.
            iterator_timeout: Timeout in seconds for non-blocking iteration.
                If None, iteration blocks until data is available.
            raw: If True, return raw bytes instead of parsed packets.

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

            # Store configuration parameters for use in __iter__
            self._configuration = configuration
            self._usb_configuration = usb_configuration
            self._serial = serial
            self._iterator_timeout = iterator_timeout
            self._raw = raw
            self._kwargs = kwargs or {}

            # Reference to the active device (set during iteration)
            self._device: typing.Optional[typing.Any] = None
            # Pending configuration update (set via update_configuration)
            self._pending_configuration: typing.Optional[typing.Any] = None

        except ImportError as e:
            logging.error(
                "The neuromorphic_drivers library is not available, please install"
            )
            raise e

    def __iter__(self) -> collections.abc.Iterator[np.ndarray]:
        logging.info(
            f"Starting streaming from neuromorphic_drivers: {self.device_list[0]}"
        )
        with self.nd.open(
            configuration=self._configuration,
            usb_configuration=self._usb_configuration,
            serial=self._serial,
            iterator_timeout=self._iterator_timeout,
            raw=self._raw,
            **self._kwargs,
        ) as device:
            self._device = device
            try:
                for status, packet in device:
                    # Check for pending configuration update
                    if self._pending_configuration is not None:
                        device.update_configuration(self._pending_configuration)
                        self._pending_configuration = None

                    if self._raw:
                        # Handle raw bytes mode differently if needed
                        yield packet
                    else:
                        events = packet.polarity_events
                        if events is not None:
                            yield events.astype(events_stream.EVENTS_DTYPE)
            finally:
                self._device = None

    def update_configuration(self, configuration: typing.Any) -> None:
        """Update the device configuration while streaming.

        The configuration update will be applied on the next iteration.
        This method is thread-safe and can be called from another thread.

        Args:
            configuration: Device-specific configuration object matching the device type.

        Raises:
            RuntimeError: If the stream is not currently iterating.
        """
        if self._device is None:
            raise RuntimeError(
                "Cannot update configuration: stream is not currently iterating. "
                "Start iterating over the stream first."
            )
        self._pending_configuration = configuration


    def dimensions(self) -> tuple[int, int]:
        # If we have a specific serial, find that device
        target_device = self.device_list[0]
        if self._serial is not None:
            for dev in self.device_list:
                if dev.serial == self._serial:
                    target_device = dev
                    break

        # Use properties from device types when available
        if target_device.name == self.nd.generated.enums.Name.INIVATION_DVXPLORER:
            return (640, 480)
        elif target_device.name in (
            self.nd.generated.enums.Name.PROPHESEE_EVK4,
            self.nd.generated.enums.Name.PROPHESEE_EVK3_HD,
        ):
            return (1280, 720)
        elif target_device.name == self.nd.generated.enums.Name.INIVATION_DAVIS346:
            return (346, 260)
        else:
            raise ValueError("Unknown Camera", target_device.name)


def events_stream_from_camera(
    driver: typing.Optional[
        typing.Literal["EventCameraDrivers", "NeuromorphicDrivers", "Auto"]
    ] = None,
    manufacturer: typing.Optional[typing.Literal["Inivation", "Prophesee"]] = None,
    buffer_size: int = 1024,
    # New neuromorphic-drivers specific parameters
    nd_configuration: typing.Optional[typing.Any] = None,
    nd_usb_configuration: typing.Optional[typing.Any] = None,
    nd_serial: typing.Optional[str] = None,
    nd_iterator_timeout: typing.Optional[float] = None,
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
                usb_configuration=nd_usb_configuration,
                serial=nd_serial,
                iterator_timeout=nd_iterator_timeout,
                **nd_open_kwargs,
            )
        except Exception as e:
            error = e

    if stream is None:
        raise ValueError("No Event Camera found:", error)

    return stream
