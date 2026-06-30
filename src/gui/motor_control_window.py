from __future__ import annotations

from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from src.core.motor_control import PIMotorController
from src.io_utils.vna_reader import trigger_vna


class MotorControlWindow(QDialog):
    """
    Motor control GUI.

    The motor communication is handled by PIMotorController.
    This window only handles the GUI and button callbacks.
    """

    def __init__(self, parent=None) -> None:
        """
        Initialize the motor control window.
        """
        super().__init__(parent)

        self.setWindowTitle("Motor Control")
        self.resize(700, 430)

        self.motor: PIMotorController | None = None

        self.move_relative_value = 10000.0
        self.move_to1_value = 30000.0
        self.move_to2_value = 0.0
        self.speed_value = 500.0
        self.acceleration_value = 500.0
        self.deceleration_value = 500.0

        self.refresh_timer = QTimer(self)
        self.refresh_timer.setInterval(300)
        self.refresh_timer.timeout.connect(self.refresh_status)

        self._build_ui()
        self._apply_style()
        self._load_default_values()

    def _build_ui(self) -> None:
        """
        Build the motor control GUI.
        """
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(12)

        connection_grid = QGridLayout()
        connection_grid.setHorizontalSpacing(8)
        connection_grid.setVerticalSpacing(8)

        self.serial_edit = QLineEdit()
        self.axis_edit = QLineEdit()
        self.bypass_checkbox = QCheckBox("Bypass controller communication")

        self.connect_button = QPushButton("Connect")
        self.connect_button.clicked.connect(self.connect_motor)

        connection_grid.addWidget(QLabel("Controller serial"), 0, 0)
        connection_grid.addWidget(self.serial_edit, 0, 1)
        connection_grid.addWidget(QLabel("Axis"), 0, 2)
        connection_grid.addWidget(self.axis_edit, 0, 3)
        connection_grid.addWidget(self.connect_button, 0, 4)
        connection_grid.addWidget(self.bypass_checkbox, 1, 0, 1, 5)

        main_layout.addLayout(connection_grid)

        content_grid = QGridLayout()
        content_grid.setHorizontalSpacing(18)
        content_grid.setVerticalSpacing(8)

        self.axis_value_label = QLabel("")
        self.motor_status_label = QLabel("")
        self.current_position_label = QLabel("")
        self.target_position_label = QLabel("")
        self.target_reached_label = QLabel("")
        self.is_moving_label = QLabel("")
        self.controller_id_label = QLabel("")

        self.speed_edit = QLineEdit()
        self.acceleration_edit = QLineEdit()
        self.deceleration_edit = QLineEdit()

        self.relative_move_edit = QLineEdit()
        self.move_to1_edit = QLineEdit()
        self.move_to2_edit = QLineEdit()

        initialise_button = QPushButton("Initialise axis")
        toggle_servo_button = QPushButton("toggle")
        apply_motion_button = QPushButton("Apply speed/acc")
        refresh_button = QPushButton("Refresh this GUI")
        end_button = QPushButton("End Communication && Exit")

        relative_go_button = QPushButton("go...")
        relative_trigger_button = QPushButton("move + trigger VNA")
        move_to1_go_button = QPushButton("go...")
        move_to2_go_button = QPushButton("go...")
        define_home_button = QPushButton("define home")

        stop_button = QPushButton("STOP !!!")
        stop_button.setObjectName("StopButton")

        initialise_button.clicked.connect(self.initialise_axis)
        toggle_servo_button.clicked.connect(self.toggle_servo)
        apply_motion_button.clicked.connect(self.apply_motion_values)
        refresh_button.clicked.connect(self.refresh_status)
        end_button.clicked.connect(self.end_and_exit)

        relative_go_button.clicked.connect(self.do_relative_move)
        relative_trigger_button.clicked.connect(self.do_relative_move_and_trigger_vna)
        move_to1_go_button.clicked.connect(self.do_move_to1)
        move_to2_go_button.clicked.connect(self.do_move_to2)
        define_home_button.clicked.connect(self.define_home)
        stop_button.clicked.connect(self.stop_motor)

        row = 0
        content_grid.addWidget(QLabel("Axis"), row, 0)
        content_grid.addWidget(self.axis_value_label, row, 1)
        content_grid.addWidget(QLabel("Move Relative"), row, 3)
        content_grid.addWidget(self.relative_move_edit, row, 4)
        content_grid.addWidget(relative_go_button, row, 5)
        content_grid.addWidget(relative_trigger_button, row, 6)

        row += 1
        content_grid.addWidget(initialise_button, row, 0, 1, 2)
        content_grid.addWidget(QLabel("Move To"), row, 3)
        content_grid.addWidget(self.move_to1_edit, row, 4)
        content_grid.addWidget(move_to1_go_button, row, 5)

        row += 1
        content_grid.addWidget(QLabel("Motor Status"), row, 0)
        content_grid.addWidget(self.motor_status_label, row, 1)
        content_grid.addWidget(toggle_servo_button, row, 2)
        content_grid.addWidget(QLabel("Move To"), row, 3)
        content_grid.addWidget(self.move_to2_edit, row, 4)
        content_grid.addWidget(move_to2_go_button, row, 5)

        row += 1
        content_grid.addWidget(QLabel("Speed"), row, 0)
        content_grid.addWidget(self.speed_edit, row, 1)
        content_grid.addWidget(QLabel("Define Home"), row, 3)
        content_grid.addWidget(define_home_button, row, 4, 1, 2)

        row += 1
        content_grid.addWidget(QLabel("Acceleration"), row, 0)
        content_grid.addWidget(self.acceleration_edit, row, 1)
        content_grid.addWidget(QLabel("Current Position"), row, 3)
        content_grid.addWidget(self.current_position_label, row, 4)

        row += 1
        content_grid.addWidget(QLabel("Deceleration"), row, 0)
        content_grid.addWidget(self.deceleration_edit, row, 1)
        content_grid.addWidget(QLabel("Target Position"), row, 3)
        content_grid.addWidget(self.target_position_label, row, 4)

        row += 1
        content_grid.addWidget(apply_motion_button, row, 0, 1, 2)
        content_grid.addWidget(QLabel("Target Reached ?"), row, 3)
        content_grid.addWidget(self.target_reached_label, row, 4)

        row += 1
        content_grid.addWidget(refresh_button, row, 0, 1, 2)
        content_grid.addWidget(QLabel("Is moving ?"), row, 3)
        content_grid.addWidget(self.is_moving_label, row, 4)

        row += 1
        content_grid.addWidget(end_button, row, 0, 1, 2)

        row += 1
        content_grid.addWidget(stop_button, row, 0, 2, 7)

        main_layout.addLayout(content_grid)

        main_layout.addWidget(QLabel("Controller ID"))
        main_layout.addWidget(self.controller_id_label)

    def _apply_style(self) -> None:
        """
        Apply motor-control styling.
        """
        self.setStyleSheet(
            """
            QDialog {
                background-color: #f2f2f2;
            }

            QLabel {
                color: #111827;
                font-size: 9pt;
            }

            QLineEdit {
                background-color: #ffffff;
                border: 1px solid #b8b8b8;
                border-radius: 4px;
                padding: 4px;
            }

            QPushButton {
                background-color: #e5e7eb;
                color: #111827;
                border: 1px solid #9ca3af;
                border-radius: 4px;
                padding: 5px 9px;
            }

            QPushButton:hover {
                background-color: #dbeafe;
            }

            QPushButton#StopButton {
                color: red;
                font-size: 16pt;
                font-weight: bold;
                padding: 16px;
            }
            """
        )

    def _load_default_values(self) -> None:
        """
        Load default values into the GUI.
        """
        self.serial_edit.setText("0205500274")
        self.axis_edit.setText("1")

        self.relative_move_edit.setText(f"{self.move_relative_value:.0f}")
        self.move_to1_edit.setText(f"{self.move_to1_value:.0f}")
        self.move_to2_edit.setText(f"{self.move_to2_value:.0f}")

        self.speed_edit.setText(f"{self.speed_value:.0f}")
        self.acceleration_edit.setText(f"{self.acceleration_value:.0f}")
        self.deceleration_edit.setText(f"{self.deceleration_value:.0f}")

    def connect_motor(self) -> None:
        """
        Connect to the motor controller.
        """
        try:
            self.motor = PIMotorController(
                controller_serial_number=self.serial_edit.text().strip(),
                axis=self.axis_edit.text().strip(),
                bypass=self.bypass_checkbox.isChecked(),
            )
            self.motor.connect()
            self.apply_motion_values(refresh_after=False)
            self.refresh_status()
            self.refresh_timer.start()
        except Exception as exc:
            QMessageBox.warning(self, "Motor Control", str(exc))

    def _require_motor(self) -> PIMotorController:
        """
        Return the connected motor object.
        """
        if self.motor is None or not self.motor.connected:
            raise RuntimeError("Connect to the motor controller first.")

        return self.motor

    def update_move_values_from_fields(self) -> None:
        """
        Read motion values from GUI fields.
        """
        self.move_relative_value = float(self.relative_move_edit.text().strip())
        self.move_to1_value = float(self.move_to1_edit.text().strip())
        self.move_to2_value = float(self.move_to2_edit.text().strip())
        self.speed_value = float(self.speed_edit.text().strip())
        self.acceleration_value = float(self.acceleration_edit.text().strip())
        self.deceleration_value = float(self.deceleration_edit.text().strip())

    def apply_motion_values(self, refresh_after: bool = True) -> None:
        """
        Apply speed, acceleration, and deceleration to the controller.
        """
        try:
            motor = self._require_motor()
            self.update_move_values_from_fields()
            motor.set_motion_values(
                speed=self.speed_value,
                acceleration=self.acceleration_value,
                deceleration=self.deceleration_value,
            )
            if refresh_after:
                self.refresh_status()
        except Exception as exc:
            QMessageBox.warning(self, "Motor Control", str(exc))

    def initialise_axis(self) -> None:
        """
        Initialize motor axis.
        """
        try:
            self._require_motor().initialize_axis()
            self.refresh_status()
        except Exception as exc:
            QMessageBox.warning(self, "Motor Control", str(exc))

    def do_relative_move(self) -> None:
        """
        Execute a relative move.
        """
        try:
            self.update_move_values_from_fields()
            self._require_motor().move_relative(self.move_relative_value)
            self.refresh_status()
        except Exception as exc:
            QMessageBox.warning(self, "Motor Control", str(exc))

    def do_relative_move_and_trigger_vna(self) -> None:
        """
        Execute a relative motor move and send a trigger command to the VNA.
        """
        try:
            self.update_move_values_from_fields()
            self._require_motor().move_relative(self.move_relative_value)
            self.trigger_vna_from_parent_settings()
            self.refresh_status()
        except Exception as exc:
            QMessageBox.warning(self, "Motor + VNA Trigger", str(exc))

    def do_move_to1(self) -> None:
        """
        Move to absolute position 1.
        """
        try:
            self.update_move_values_from_fields()
            self._require_motor().move_absolute(self.move_to1_value)
            self.refresh_status()
        except Exception as exc:
            QMessageBox.warning(self, "Motor Control", str(exc))

    def do_move_to2(self) -> None:
        """
        Move to absolute position 2.
        """
        try:
            self.update_move_values_from_fields()
            self._require_motor().move_absolute(self.move_to2_value)
            self.refresh_status()
        except Exception as exc:
            QMessageBox.warning(self, "Motor Control", str(exc))

    def define_home(self) -> None:
        """
        Define current position as home.
        """
        try:
            self._require_motor().define_home()
            self.refresh_status()
        except Exception as exc:
            QMessageBox.warning(self, "Motor Control", str(exc))

    def stop_motor(self) -> None:
        """
        Stop motor immediately and switch servo off.
        """
        try:
            self._require_motor().stop()
            self.refresh_status()
        except Exception as exc:
            QMessageBox.warning(self, "Motor Control", str(exc))

    def toggle_servo(self) -> None:
        """
        Toggle motor servo state.
        """
        try:
            self._require_motor().toggle_servo()
            self.refresh_status()
        except Exception as exc:
            QMessageBox.warning(self, "Motor Control", str(exc))

    def trigger_vna_from_parent_settings(self) -> None:
        """
        Send a VNA trigger using the VNA settings stored in MainWindow.
        """
        parent = self.parent()

        ip_address = getattr(parent, "vna_ip_address", "128.11.11.11")
        port = getattr(parent, "vna_port", 1601)

        trigger_vna(
            ip_address=ip_address,
            port=port,
            trigger_command="*TRG\n",
        )

    def refresh_status(self) -> None:
        """
        Refresh motor status labels.
        """
        if self.motor is None or not self.motor.connected:
            return

        try:
            status = self.motor.status()
        except Exception:
            return

        self.axis_value_label.setText(self.axis_edit.text().strip())
        self.current_position_label.setText(f"{status.current_position:.1f}")
        self.target_position_label.setText(f"{status.target_position:.1f}")
        self.target_reached_label.setText("Yes" if status.target_reached else "No")
        self.is_moving_label.setText("yes" if status.is_moving else "no")
        self.motor_status_label.setText("ON" if status.servo_on else "OFF")
        self.controller_id_label.setText(status.controller_id)

        if status.target_reached:
            self.target_reached_label.setStyleSheet("color: black;")
        else:
            self.target_reached_label.setStyleSheet("color: blue;")

        if status.is_moving:
            self.is_moving_label.setStyleSheet("color: blue;")
        else:
            self.is_moving_label.setStyleSheet("color: black;")

    def end_and_exit(self) -> None:
        """
        End communication and close the window.
        """
        self.refresh_timer.stop()

        if self.motor is not None:
            self.motor.close()

        self.close()

    def closeEvent(self, event) -> None:
        """
        Keep communication cleanup safe when the window is closed.
        """
        self.refresh_timer.stop()

        if self.motor is not None:
            self.motor.close()

        event.accept()