from __future__ import annotations

from dataclasses import dataclass


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
    Small wrapper around a PI motor controller.

    The real communication uses `pipython` when bypass is False.
    In bypass mode the GUI can be tested without hardware.
    """

    def __init__(
        self,
        controller_serial_number: str = "0205500274",
        axis: str = "1",
        bypass: bool = False,
    ) -> None:
        """
        Initialize the motor controller wrapper.
        """
        self.controller_serial_number = str(controller_serial_number)
        self.axis = str(axis)
        self.bypass = bool(bypass)

        self.device = None
        self.connected = False
        self.controller_id = ""

        self.current_position = 0.0
        self.target_position = 0.0
        self.minimum_position = None
        self.maximum_position = None
        self.speed = 500.0
        self.acceleration = 500.0
        self.deceleration = 500.0
        self.servo_on = False

    def connect(self) -> MotorStatus:
        """
        Connect to the PI motor controller.
        """
        if self.bypass:
            self.connected = True
            self.controller_id = "BYPASS PI motor controller"
            return self.status()

        try:
            from pipython import GCSDevice
        except ImportError as exc:
            raise RuntimeError(
                "The Python PI driver `pipython` is not installed. "
                "Install it or enable bypass mode for GUI testing."
            ) from exc

        self.device = GCSDevice()
        self.device.ConnectUSB(serialnum=self.controller_serial_number)

        self.connected = True
        self.controller_id = str(self.device.qIDN()).replace("\n", "").replace("\r", "")

        try:
            self.minimum_position = float(self._axis_value(self.device.qTMN(self.axis)))
        except Exception:
            self.minimum_position = None

        try:
            self.maximum_position = float(self._axis_value(self.device.qTMX(self.axis)))
        except Exception:
            self.maximum_position = None

        return self.status()

    def initialize_axis(self) -> MotorStatus:
        """
        Initialize the axis position reference.
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
        deceleration: float,
    ) -> MotorStatus:
        """
        Set velocity, acceleration, and deceleration.
        """
        self._require_connected()

        self.speed = float(speed)
        self.acceleration = float(acceleration)
        self.deceleration = float(deceleration)

        if not self.bypass:
            self.device.VEL(self.axis, self.speed)
            self.device.ACC(self.axis, self.acceleration)
            self.device.DEC(self.axis, self.deceleration)

        return self.status()

    def move_relative(self, distance: float) -> MotorStatus:
        """
        Move the motor relative to the current position.
        """
        self._require_connected()

        distance = float(distance)

        if self.bypass:
            self.target_position = self.current_position + distance
            self.current_position = self.target_position
            return self.status()

        self.device.MVR(self.axis, distance)

        return self.status()

    def move_absolute(self, position: float) -> MotorStatus:
        """
        Move the motor to an absolute position.
        """
        self._require_connected()

        position = float(position)

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

    def toggle_servo(self) -> MotorStatus:
        """
        Toggle the servo state.
        """
        self._require_connected()

        current = self.servo_is_on()
        new_value = not current

        if self.bypass:
            self.servo_on = new_value
            return self.status()

        self.device.SVO(self.axis, 1 if new_value else 0)

        return self.status()

    def servo_is_on(self) -> bool:
        """
        Return whether the servo is on.
        """
        if self.bypass:
            return bool(self.servo_on)

        self._require_connected()

        return bool(self._axis_value(self.device.qSVO(self.axis)))

    def close(self) -> None:
        """
        Close communication.
        """
        if self.bypass:
            self.connected = False
            return

        if self.device is not None:
            try:
                self.device.SVO(self.axis, 0)
            except Exception:
                pass

            try:
                self.device.CloseConnection()
            except Exception:
                pass

        self.connected = False
        self.device = None

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
            is_moving = not target_reached

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

    def _require_connected(self) -> None:
        """
        Raise if the motor is not connected.
        """
        if not self.connected:
            raise RuntimeError("Motor controller is not connected.")

    def _axis_value(self, value):
        """
        Extract an axis value from PI return objects.
        """
        if isinstance(value, dict):
            if self.axis in value:
                return value[self.axis]

            if int(self.axis) in value:
                return value[int(self.axis)]

            return next(iter(value.values()))

        if isinstance(value, (list, tuple)):
            return value[0]

        return value