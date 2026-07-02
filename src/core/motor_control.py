from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Literal


ConnectionInterface = Literal["usb", "rs232"]


@dataclass(slots=True)
class MotorStatus:
    """
    Current motor status.
    """

    connected: bool = False
    current_position: float = 0.0
    target_position: float = 0.0
    target_reached: bool = False
    is_moving: bool = False
    servo_on: bool = False
    minimum_position: float | None = None
    maximum_position: float | None = None
    controller_id: str = ""


class PIMotorController:
    """
    Wrapper around a PI Mercury C-863 motor controller.

    Supported communication modes
    -----------------------------
    rs232:
        Uses PIPython's low-level serial backend:

            PISerial -> GCSMessages -> GCSCommands

        This path does not require PI_GCS2_DLL_x64.dll.

    usb:
        Uses PIPython's GCSDevice backend.

        On Windows this usually requires PI_GCS2_DLL_x64.dll from the
        PI software / GCS DLL installation.

    bypass:
        Simulates the controller so the GUI can be tested without hardware.
    """

    def __init__(
        self,
        controller_serial_number: str = "0205500274",
        axis: str = "1",
        bypass: bool = False,
        interface: ConnectionInterface = "rs232",
        rs232_port: str | int | None = "COM5",
        baudrate: int = 38400,
        controller_name: str = "C-863",
    ) -> None:
        """
        Initialize the motor controller wrapper.

        Parameters
        ----------
        controller_serial_number:
            Used for USB connection.

        axis:
            Controller axis. For your C-863 setup this is usually "1".

        bypass:
            If True, no hardware communication is attempted.

        interface:
            "rs232" or "usb".

        rs232_port:
            Serial port for RS232 communication.
            Example on Windows: "COM5".
            Example on Linux: "/dev/ttyUSB0".

        baudrate:
            RS232 baud rate. Your controller responded at 38400.

        controller_name:
            PI controller name used by GCSDevice for USB.
        """
        self.controller_serial_number = str(controller_serial_number)
        self.axis = str(axis)
        self.bypass = bool(bypass)

        interface = str(interface).lower().strip()
        if interface not in {"usb", "rs232"}:
            raise ValueError("interface must be 'usb' or 'rs232'.")

        self.interface: str = interface
        self.rs232_port = rs232_port
        self.baudrate = int(baudrate)
        self.controller_name = str(controller_name)

        self.device: Any | None = None
        self.connected = False
        self.controller_id = ""

        self.current_position = 0.0
        self.target_position = 0.0
        self.minimum_position: float | None = None
        self.maximum_position: float | None = None

        self.speed = 500.0
        self.acceleration = 500.0
        self.deceleration = 500.0
        self.servo_on = False

        # RS232 backend resources.
        self.gateway_context: Any | None = None
        self.gateway: Any | None = None
        self.messages: Any | None = None
        self.device_context: Any | None = None

    def connect(self) -> MotorStatus:
        """
        Connect to the PI motor controller.
        """
        if self.connected:
            return self.status()

        if self.bypass:
            self.connected = True
            self.controller_id = "BYPASS PI motor controller"
            return self.status()

        try:
            if self.interface == "rs232":
                self._connect_rs232()
            elif self.interface == "usb":
                self._connect_usb()
            else:
                raise ValueError("interface must be 'usb' or 'rs232'.")

            self.connected = True
            self.controller_id = self._clean_controller_id(self.device.qIDN())
            self._read_travel_limits()

            return self.status()

        except Exception:
            self._cleanup_connection(switch_servo_off=False)
            raise

    def _connect_rs232(self) -> None:
        """
        Connect using PIPython's low-level serial backend.

        This path does not require PI_GCS2_DLL_x64.dll.
        """
        if self.rs232_port is None:
            raise ValueError("rs232_port must be set when interface='rs232'.")

        try:
            from pipython.pidevice.gcscommands import GCSCommands
            from pipython.pidevice.gcsmessages import GCSMessages
            from pipython.pidevice.interfaces.piserial import PISerial
        except ImportError as exc:
            raise RuntimeError(
                "PIPython and pyserial are required for RS232 communication. "
                "Install them with: pip install PIPython pyserial"
            ) from exc

        self.gateway_context = PISerial(
            port=self.rs232_port,
            baudrate=self.baudrate,
        )
        self.gateway = self.gateway_context.__enter__()

        self.messages = GCSMessages(self.gateway)

        self.device_context = GCSCommands(self.messages)
        self.device = self.device_context.__enter__()

    def _connect_usb(self) -> None:
        """
        Connect using PIPython's GCSDevice backend.

        On Windows this usually requires PI_GCS2_DLL_x64.dll.
        """
        try:
            from pipython import GCSDevice
        except ImportError as exc:
            raise RuntimeError(
                "PIPython is required for USB communication. "
                "Install it with: pip install PIPython"
            ) from exc

        self.device = GCSDevice(self.controller_name)
        self.device.ConnectUSB(serialnum=self.controller_serial_number)

    def initialize_axis(self) -> MotorStatus:
        """
        Initialize the axis position reference.

        This follows the sequence from the MATLAB code:

            SVO(axis, 0)
            RON(axis, 0)
            POS(axis, 0)
            RON(axis, 1)
            SVO(axis, previous_servo_state)
        """
        self._require_connected()

        if self.bypass:
            self.current_position = 0.0
            self.target_position = 0.0
            return self.status()

        servo_state = self.servo_is_on()

        self.device.SVO(self.axis, 0)
        self.device.RON(self.axis, 0)
        self.device.POS(self.axis, 0)
        self.device.RON(self.axis, 1)
        self.device.SVO(self.axis, 1 if servo_state else 0)

        return self.status()

    def set_motion_values(
        self,
        speed: float,
        acceleration: float,
        deceleration: float | None = None,
    ) -> MotorStatus:
        """
        Set velocity, acceleration, and deceleration.

        If deceleration is None, it follows acceleration.
        """
        self._require_connected()

        self.speed = float(speed)
        self.acceleration = float(acceleration)
        self.deceleration = float(acceleration if deceleration is None else deceleration)

        if not self.bypass:
            self.device.VEL(self.axis, self.speed)
            self.device.ACC(self.axis, self.acceleration)
            self.device.DEC(self.axis, self.deceleration)

        return self.status()

    def move_relative(self, distance: float) -> MotorStatus:
        """
        Move the motor relative to the current position.

        This does not automatically switch the servo on.
        Call set_servo(True) first if needed.
        """
        self._require_connected()

        distance = float(distance)

        if self.bypass:
            target = self.current_position + distance
            self._check_absolute_position_limit(target)
            self.target_position = target
            self.current_position = target
            return self.status()

        current = self.status().current_position
        target = current + distance
        self._check_absolute_position_limit(target)

        self.device.MVR(self.axis, distance)

        return self.status()

    def move_absolute(self, position: float) -> MotorStatus:
        """
        Move the motor to an absolute position.

        This does not automatically switch the servo on.
        Call set_servo(True) first if needed.
        """
        self._require_connected()

        position = float(position)
        self._check_absolute_position_limit(position)

        if self.bypass:
            self.target_position = position
            self.current_position = position
            return self.status()

        self.device.MOV(self.axis, position)

        return self.status()

    def define_home(self) -> MotorStatus:
        """
        Define the current position as home.
        """
        self._require_connected()

        if self.bypass:
            self.current_position = 0.0
            self.target_position = 0.0
            return self.status()

        self.device.DFH(self.axis)

        return self.status()

    def stop(self) -> MotorStatus:
        """
        Stop the motor and switch servo off.
        """
        self._require_connected()

        if self.bypass:
            self.target_position = self.current_position
            self.servo_on = False
            return self.status()

        self.device.STP()
        self.device.SVO(self.axis, 0)

        return self.status()

    def set_servo(self, enabled: bool) -> MotorStatus:
        """
        Set the servo state explicitly.
        """
        self._require_connected()

        enabled = bool(enabled)

        if self.bypass:
            self.servo_on = enabled
            return self.status()

        self.device.SVO(self.axis, 1 if enabled else 0)

        return self.status()

    def toggle_servo(self) -> MotorStatus:
        """
        Toggle the servo state.
        """
        self._require_connected()

        current = self.servo_is_on()
        return self.set_servo(not current)

    def servo_is_on(self) -> bool:
        """
        Return whether the servo is on.
        """
        if self.bypass:
            return bool(self.servo_on)

        self._require_connected()

        return bool(self._axis_value(self.device.qSVO(self.axis)))

    def wait_until_done(
        self,
        timeout_s: float = 30.0,
        poll_s: float = 0.1,
    ) -> MotorStatus:
        """
        Wait until the controller reports that the target has been reached.

        Raises TimeoutError if the target is not reached before timeout_s.
        """
        self._require_connected()

        if self.bypass:
            return self.status()

        deadline = time.monotonic() + float(timeout_s)
        last_status = self.status()

        while time.monotonic() < deadline:
            last_status = self.status()

            if last_status.target_reached and not last_status.is_moving:
                return last_status

            time.sleep(float(poll_s))

        raise TimeoutError(
            "Motor did not reach the target within "
            f"{timeout_s:.1f} seconds. Last status: {last_status}"
        )

    def status(self) -> MotorStatus:
        """
        Return current motor status.
        """
        if not self.connected:
            return MotorStatus(connected=False)

        if self.bypass:
            return MotorStatus(
                connected=True,
                current_position=float(self.current_position),
                target_position=float(self.target_position),
                target_reached=True,
                is_moving=False,
                servo_on=bool(self.servo_on),
                minimum_position=self.minimum_position,
                maximum_position=self.maximum_position,
                controller_id=self.controller_id,
            )

        current_position = float(self._axis_value(self.device.qPOS(self.axis)))
        target_position = float(self._axis_value(self.device.qMOV(self.axis)))
        target_reached = bool(self._axis_value(self.device.qONT(self.axis)))
        servo_on = bool(self._axis_value(self.device.qSVO(self.axis)))

        try:
            is_moving = bool(self._axis_value(self.device.IsMoving(self.axis)))
        except Exception:
            is_moving = bool(servo_on and not target_reached)

        self.current_position = current_position
        self.target_position = target_position
        self.servo_on = servo_on

        return MotorStatus(
            connected=True,
            current_position=current_position,
            target_position=target_position,
            target_reached=target_reached,
            is_moving=is_moving,
            servo_on=servo_on,
            minimum_position=self.minimum_position,
            maximum_position=self.maximum_position,
            controller_id=self.controller_id,
        )

    def close(self) -> None:
        """
        Close communication.

        The method tries to switch the servo off before closing.
        """
        self._cleanup_connection(switch_servo_off=True)

    def _cleanup_connection(self, switch_servo_off: bool = True) -> None:
        """
        Close USB or RS232 resources.
        """
        if self.bypass:
            self.connected = False
            return

        if switch_servo_off and self.device is not None:
            try:
                self.device.SVO(self.axis, 0)
            except Exception:
                pass

        if self.device_context is not None:
            try:
                self.device_context.__exit__(None, None, None)
            except Exception:
                pass
            self.device_context = None

        if self.gateway_context is not None:
            try:
                self.gateway_context.__exit__(None, None, None)
            except Exception:
                pass
            self.gateway_context = None

        if self.device is not None and self.interface == "usb":
            try:
                self.device.CloseConnection()
            except Exception:
                pass

        self.connected = False
        self.device = None
        self.gateway = None
        self.messages = None

    def _read_travel_limits(self) -> None:
        """
        Read software travel limits if available.
        """
        try:
            self.minimum_position = float(
                self._axis_value(self.device.qTMN(self.axis))
            )
        except Exception:
            self.minimum_position = None

        try:
            self.maximum_position = float(
                self._axis_value(self.device.qTMX(self.axis))
            )
        except Exception:
            self.maximum_position = None

    def _check_absolute_position_limit(self, position: float) -> None:
        """
        Raise if position is outside known commandable limits.
        """
        if self.minimum_position is not None and position < self.minimum_position:
            raise ValueError(
                f"Target position {position} is below minimum "
                f"{self.minimum_position}."
            )

        if self.maximum_position is not None and position > self.maximum_position:
            raise ValueError(
                f"Target position {position} is above maximum "
                f"{self.maximum_position}."
            )

    def _require_connected(self) -> None:
        """
        Raise if the motor is not connected.
        """
        if not self.connected:
            raise RuntimeError("Motor controller is not connected.")

    def _axis_value(self, value: Any) -> Any:
        """
        Extract an axis value from PI return objects.
        """
        if isinstance(value, dict):
            if self.axis in value:
                return value[self.axis]

            try:
                axis_int = int(self.axis)
                if axis_int in value:
                    return value[axis_int]
            except Exception:
                pass

            return next(iter(value.values()))

        if isinstance(value, (list, tuple)):
            return value[0]

        return value

    @staticmethod
    def _clean_controller_id(value: Any) -> str:
        """
        Clean controller identification string.
        """
        return str(value).replace("\n", "").replace("\r", "")

    def __enter__(self) -> PIMotorController:
        """
        Context-manager entry.
        """
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        """
        Context-manager exit.
        """
        self.close()
        return False