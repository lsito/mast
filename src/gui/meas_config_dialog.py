from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QButtonGroup,
    QFileDialog,
    QMessageBox,
)
from PySide6.QtCore import Qt

from src.data_models.meas_config import MeasurementConfig

class MeasurementConfigDialog(QDialog):
    def __init__(self, config: MeasurementConfig | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Measurement condition table")
        self.setModal(True)
        self.resize(560, 420)

        self._config = config or MeasurementConfig()
        self._loaded_file = None

        main_layout = QVBoxLayout(self)

        grid = QGridLayout()
        grid.setHorizontalSpacing(20)
        grid.setVerticalSpacing(14)

        # Row 0
        grid.addWidget(QLabel("Designed Operating frequency (MHz)"), 0, 0)
        self.designed_frequency_edit = QLineEdit(str(self._config.designed_frequency_mhz))
        grid.addWidget(self.designed_frequency_edit, 0, 1)

        # Row 1
        grid.addWidget(QLabel("Designed Operating temperature (Deg C)"), 1, 0)
        self.designed_temp_edit = QLineEdit(str(self._config.designed_temperature_c))
        grid.addWidget(self.designed_temp_edit, 1, 1)

        # Row 2
        grid.addWidget(QLabel("RF measurement temperature (Deg C)"), 2, 0)
        self.rf_temp_edit = QLineEdit(str(self._config.rf_measurement_temperature_c))
        grid.addWidget(self.rf_temp_edit, 2, 1)

        # Row 3
        grid.addWidget(QLabel("Frequency shift w/o --> w/ wire (MHz)"), 3, 0)
        self.freq_shift_edit = QLineEdit(str(self._config.frequency_shift_mhz))
        grid.addWidget(self.freq_shift_edit, 3, 1)

        # Row 4 atmosphere
        grid.addWidget(QLabel("Atmosphere inside structure"), 4, 0)
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

        # Row 5
        grid.addWidget(QLabel("Relative Humidity (%)"), 5, 0)
        self.humidity_edit = QLineEdit(str(self._config.relative_humidity_percent))
        grid.addWidget(self.humidity_edit, 5, 1)

        # Row 6
        grid.addWidget(QLabel("Target frequency to tune (MHz)"), 6, 0)
        self.target_frequency_edit = QLineEdit(str(self._config.target_frequency_mhz))
        grid.addWidget(self.target_frequency_edit, 6, 1)

        main_layout.addLayout(grid)
        main_layout.addStretch()

        # Load file button row
        load_layout = QHBoxLayout()
        load_layout.addStretch()
        self.load_file_button = QPushButton("load file...")
        self.load_file_button.clicked.connect(self.load_from_file)
        load_layout.addWidget(self.load_file_button)
        load_layout.addStretch()
        main_layout.addLayout(load_layout)

        # Bottom buttons
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

    def _read_float(self, edit: QLineEdit, field_name: str) -> float:
        text = edit.text().strip()
        try:
            return float(text)
        except ValueError as exc:
            raise ValueError(f"Invalid value for '{field_name}': {text!r}") from exc

    def get_config(self) -> MeasurementConfig:
        atmosphere = "Air" if self.air_radio.isChecked() else "Nitrogen"

        return MeasurementConfig(
            designed_frequency_mhz=self._read_float(
                self.designed_frequency_edit, "Designed Operating frequency"
            ),
            designed_temperature_c=self._read_float(
                self.designed_temp_edit, "Designed Operating temperature"
            ),
            rf_measurement_temperature_c=self._read_float(
                self.rf_temp_edit, "RF measurement temperature"
            ),
            frequency_shift_mhz=self._read_float(
                self.freq_shift_edit, "Frequency shift"
            ),
            atmosphere=atmosphere,
            relative_humidity_percent=self._read_float(
                self.humidity_edit, "Relative Humidity"
            ),
            target_frequency_mhz=self._read_float(
                self.target_frequency_edit, "Target frequency to tune"
            ),
        )

    def on_apply(self):
        try:
            self._config = self.get_config()
        except ValueError as e:
            QMessageBox.warning(self, "Invalid input", str(e))
            return

    def on_ok(self):
        try:
            self._config = self.get_config()
        except ValueError as e:
            QMessageBox.warning(self, "Invalid input", str(e))
            return
        self.accept()

    def load_from_file(self):
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