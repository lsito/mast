from __future__ import annotations

from PySide6.QtWidgets import (
    QButtonGroup,
    QDialog,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
)

from src.data_models.meas_config import MeasurementConfig


class MeasurementConfigDialog(QDialog):
    """
    Dialog for editing measurement conditions.

    The target frequency field is read-only and is updated automatically from
    the editable measurement parameters.
    """

    def __init__(self, config: MeasurementConfig | None = None, parent=None) -> None:
        """
        Initialize the measurement configuration dialog.
        """
        super().__init__(parent)

        self.setWindowTitle("Measurement Conditions")
        self.setModal(True)
        self.resize(560, 420)

        self._config = config or MeasurementConfig()
        self._loaded_file = None

        self._build_ui()
        self._connect_live_updates()
        self.update_target_frequency()

    def _build_ui(self) -> None:
        """
        Build the dialog user interface.
        """
        main_layout = QVBoxLayout(self)

        grid = QGridLayout()
        grid.setHorizontalSpacing(20)
        grid.setVerticalSpacing(14)

        degree_celsius = "\u00B0C"

        grid.addWidget(QLabel("Design Op. Frequency [MHz]"), 0, 0)
        self.designed_frequency_edit = QLineEdit(
            str(self._config.designed_frequency_mhz)
        )
        grid.addWidget(self.designed_frequency_edit, 0, 1)

        grid.addWidget(QLabel(f"Design Op. Temperature [{degree_celsius}]"), 1, 0)
        self.designed_temp_edit = QLineEdit(
            str(self._config.designed_temperature_c)
        )
        grid.addWidget(self.designed_temp_edit, 1, 1)

        grid.addWidget(QLabel(f"RF Meas. Temperature [{degree_celsius}]"), 2, 0)
        self.rf_temp_edit = QLineEdit(
            str(self._config.rf_measurement_temperature_c)
        )
        grid.addWidget(self.rf_temp_edit, 2, 1)

        grid.addWidget(QLabel("Frequency Shift w/o VS w/ wire [MHz]"), 3, 0)
        self.freq_shift_edit = QLineEdit(
            str(self._config.frequency_shift_mhz)
        )
        grid.addWidget(self.freq_shift_edit, 3, 1)

        grid.addWidget(QLabel("Atmosphere in Structure"), 4, 0)

        atmosphere_layout = QHBoxLayout()

        self.nitrogen_radio = QRadioButton("Nitrogen")
        self.air_radio = QRadioButton("Air")

        self.atmosphere_group = QButtonGroup(self)
        self.atmosphere_group.addButton(self.nitrogen_radio)
        self.atmosphere_group.addButton(self.air_radio)

        atmosphere_layout.addWidget(self.nitrogen_radio)
        atmosphere_layout.addWidget(self.air_radio)
        atmosphere_layout.addStretch()

        if self._config.atmosphere.lower() == "air":
            self.air_radio.setChecked(True)
        else:
            self.nitrogen_radio.setChecked(True)

        grid.addLayout(atmosphere_layout, 4, 1)

        grid.addWidget(QLabel("Relative Humidity [%]"), 5, 0)
        self.humidity_edit = QLineEdit(
            str(self._config.relative_humidity_percent)
        )
        grid.addWidget(self.humidity_edit, 5, 1)

        grid.addWidget(QLabel("Target Frequency to Tune [MHz]"), 6, 0)
        self.target_frequency_edit = QLineEdit(
            f"{self._config.target_frequency_mhz:.6f}"
        )
        self.target_frequency_edit.setReadOnly(True)
        grid.addWidget(self.target_frequency_edit, 6, 1)

        main_layout.addLayout(grid)
        main_layout.addStretch()

        load_layout = QHBoxLayout()
        load_layout.addStretch()

        self.load_file_button = QPushButton("Load File...")
        self.load_file_button.clicked.connect(self.load_from_file)

        load_layout.addWidget(self.load_file_button)
        load_layout.addStretch()

        main_layout.addLayout(load_layout)

        button_layout = QHBoxLayout()

        self.ok_button = QPushButton("OK")
        self.apply_button = QPushButton("Apply")
        self.cancel_button = QPushButton("Cancel")

        self.ok_button.clicked.connect(self.on_ok)
        self.apply_button.clicked.connect(self.on_apply)
        self.cancel_button.clicked.connect(self.reject)

        button_layout.addWidget(self.ok_button)
        button_layout.addStretch()
        button_layout.addWidget(self.apply_button)
        button_layout.addStretch()
        button_layout.addWidget(self.cancel_button)

        main_layout.addLayout(button_layout)

    def _connect_live_updates(self) -> None:
        """
        Connect editable fields to the automatic target-frequency update.
        """
        self.designed_frequency_edit.textChanged.connect(self.update_target_frequency)
        self.designed_temp_edit.textChanged.connect(self.update_target_frequency)
        self.rf_temp_edit.textChanged.connect(self.update_target_frequency)
        self.freq_shift_edit.textChanged.connect(self.update_target_frequency)
        self.humidity_edit.textChanged.connect(self.update_target_frequency)

        self.air_radio.toggled.connect(self.update_target_frequency)
        self.nitrogen_radio.toggled.connect(self.update_target_frequency)

    def _read_float(self, edit: QLineEdit, field_name: str) -> float:
        """
        Read a float from a line edit.
        """
        text = edit.text().strip()

        try:
            return float(text)
        except ValueError as exc:
            raise ValueError(f"Invalid value for '{field_name}': {text!r}") from exc

    def _config_from_fields(self) -> MeasurementConfig:
        """
        Build a measurement configuration from the current dialog fields.
        """
        atmosphere = "Air" if self.air_radio.isChecked() else "Nitrogen"

        return MeasurementConfig(
            designed_frequency_mhz=self._read_float(
                self.designed_frequency_edit,
                "Designed Operating frequency",
            ),
            designed_temperature_c=self._read_float(
                self.designed_temp_edit,
                "Designed Operating temperature",
            ),
            rf_measurement_temperature_c=self._read_float(
                self.rf_temp_edit,
                "RF measurement temperature",
            ),
            frequency_shift_mhz=self._read_float(
                self.freq_shift_edit,
                "Frequency shift",
            ),
            atmosphere=atmosphere,
            relative_humidity_percent=self._read_float(
                self.humidity_edit,
                "Relative Humidity",
            ),
        )

    def update_target_frequency(self, *_args) -> None:
        """
        Update the read-only target-frequency field.
        """
        try:
            config = self._config_from_fields()
            self.target_frequency_edit.setText(
                f"{config.target_frequency_mhz:.6f}"
            )
        except ValueError:
            self.target_frequency_edit.setText("")

    def get_config(self) -> MeasurementConfig:
        """
        Return the current measurement configuration.
        """
        return self._config_from_fields()

    def on_apply(self) -> None:
        """
        Apply the current dialog values without closing the dialog.
        """
        try:
            self._config = self.get_config()
            self.update_target_frequency()
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid input", str(exc))
            return

    def on_ok(self) -> None:
        """
        Apply the current dialog values and close the dialog.
        """
        try:
            self._config = self.get_config()
            self.update_target_frequency()
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid input", str(exc))
            return

        self.accept()

    def load_from_file(self) -> None:
        """
        Select a measurement configuration file.

        The actual parser can be connected here later.
        """
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Load measurement config",
            "",
            "Config Files (*.json *.yaml *.yml *.toml *.txt);;All Files (*)",
        )

        if not filename:
            return

        self._loaded_file = filename

        QMessageBox.information(
            self,
            "Load file",
            f"Selected file:\n{filename}\n\nYou can connect this to your read_cfg() later.",
        )