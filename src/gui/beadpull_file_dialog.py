from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from src.data_models.bead_config import BeadpullConfig


class BeadpullFileDialog(QDialog):
    """
    Dialog for adding one bead-pull file.

    The dialog selects the bead-pull CSV file and allows per-file control of
    frequency, temperature, and bead-pull analysis options.
    """

    def __init__(
        self,
        default_options: BeadpullConfig | None = None,
        parent=None,
    ) -> None:
        """
        Initialize the bead-pull file dialog.
        """
        super().__init__(parent)

        self.setWindowTitle("Add bead-pull file")
        self.setModal(True)
        self.resize(520, 430)

        self.filename: str | None = None
        self.options = deepcopy(default_options) if default_options is not None else BeadpullConfig()

        self.f0_MHz_override: float | None = None
        self.temperature_degC_override: float | None = None
        self.invert_rf_structure_parameters = False
        self.invert_measurement_direction = False

        self._build_ui()
        self._load_options_to_fields()

    def _build_ui(self) -> None:
        """
        Build the dialog user interface.
        """
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(18, 18, 18, 18)
        main_layout.setSpacing(12)

        top_row = QHBoxLayout()
        top_row.addStretch()

        self.ok_button = QPushButton("OK")
        self.preview_button = QPushButton("Preview")
        self.cancel_button = QPushButton("Cancel")

        self.ok_button.clicked.connect(self.on_ok)
        self.preview_button.clicked.connect(self.on_preview)
        self.cancel_button.clicked.connect(self.reject)

        top_row.addWidget(self.ok_button)
        top_row.addWidget(self.preview_button)
        top_row.addWidget(self.cancel_button)

        main_layout.addLayout(top_row)
        main_layout.addStretch()

        file_row = QHBoxLayout()

        self.filename_edit = QLineEdit()
        self.filename_edit.setReadOnly(True)

        self.browse_button = QPushButton("Browse...")
        self.browse_button.clicked.connect(self.browse_file)

        file_row.addWidget(self.filename_edit)
        file_row.addWidget(self.browse_button)

        main_layout.addLayout(file_row)

        vna_row = QHBoxLayout()
        vna_row.addStretch()

        self.read_vna_button = QPushButton("Read from VNA and save csv...")
        self.read_vna_button.clicked.connect(self.read_from_vna_placeholder)

        vna_row.addWidget(self.read_vna_button)
        main_layout.addLayout(vna_row)

        option_row_1 = QHBoxLayout()

        self.use_output_checkbox = QCheckBox("use output S-parameters")
        self.invert_rf_checkbox = QCheckBox("invert RF structure parameters")

        option_row_1.addWidget(self.use_output_checkbox)
        option_row_1.addWidget(self.invert_rf_checkbox)

        main_layout.addLayout(option_row_1)

        option_row_2 = QHBoxLayout()

        self.invert_measurement_checkbox = QCheckBox("invert measurement direction")

        option_row_2.addStretch()
        option_row_2.addWidget(self.invert_measurement_checkbox)

        main_layout.addLayout(option_row_2)

        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(10)

        self.atmosphere_edit = QLineEdit()
        self.wire_detuning_edit = QLineEdit()
        self.temperature_edit = QLineEdit()
        self.frequency_edit = QLineEdit()

        self.n_zero_edit = QLineEdit()
        self.threshold_fraction_edit = QLineEdit()
        self.smooth_size_edit = QLineEdit()
        self.phase_tolerance_deg_edit = QLineEdit()
        self.remove_peaks_edit = QLineEdit()

        grid.addWidget(QLabel("Atmosphere"), 0, 0)
        grid.addWidget(self.atmosphere_edit, 0, 1)

        grid.addWidget(QLabel("Wire detuning"), 1, 0)
        grid.addWidget(self.wire_detuning_edit, 1, 1)

        grid.addWidget(QLabel("Temperature [°C]"), 2, 0)
        grid.addWidget(self.temperature_edit, 2, 1)

        grid.addWidget(QLabel("Frequency [MHz]"), 3, 0)
        grid.addWidget(self.frequency_edit, 3, 1)

        grid.addWidget(QLabel("n_zero"), 4, 0)
        grid.addWidget(self.n_zero_edit, 4, 1)

        grid.addWidget(QLabel("Threshold fraction"), 5, 0)
        grid.addWidget(self.threshold_fraction_edit, 5, 1)

        grid.addWidget(QLabel("Smooth size"), 6, 0)
        grid.addWidget(self.smooth_size_edit, 6, 1)

        grid.addWidget(QLabel("Phase tolerance [deg]"), 7, 0)
        grid.addWidget(self.phase_tolerance_deg_edit, 7, 1)

        grid.addWidget(QLabel("Remove peaks"), 8, 0)
        grid.addWidget(self.remove_peaks_edit, 8, 1)

        main_layout.addLayout(grid)

    def _load_options_to_fields(self) -> None:
        """
        Load bead-pull options into the dialog fields.
        """
        self.use_output_checkbox.setChecked(
            bool(getattr(self.options, "use_S_output_for_BP", False))
        )

        self.n_zero_edit.setText(str(getattr(self.options, "n_zero", 30)))
        self.threshold_fraction_edit.setText(
            str(getattr(self.options, "threshold_fraction", 0.15))
        )
        self.smooth_size_edit.setText(str(getattr(self.options, "smooth_size", 5)))

        phase_tolerance = float(getattr(self.options, "phase_tolerance", np.deg2rad(5.0)))
        self.phase_tolerance_deg_edit.setText(f"{np.rad2deg(phase_tolerance):.6g}")

        remove_peaks = getattr(self.options, "remove_peaks", [])
        self.remove_peaks_edit.setText(",".join(str(v) for v in remove_peaks))

    def browse_file(self) -> None:
        """
        Select a bead-pull CSV file.
        """
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Select bead-pull CSV",
            "",
            "CSV files (*.csv);;All files (*)",
        )

        if not filename:
            return

        self.filename = filename
        self.filename_edit.setText(filename)
        self._try_fill_frequency_and_temperature_from_filename(filename)

    def _try_fill_frequency_and_temperature_from_filename(self, filename: str) -> None:
        """
        Fill frequency and temperature fields from a filename like BP_11989.24_19.1deg.csv.
        """
        name = Path(filename).name

        try:
            stem = Path(name).stem

            if not stem.startswith("BP_"):
                return

            parts = stem.removeprefix("BP_").split("_")

            if len(parts) < 2:
                return

            f0_MHz = float(parts[0])
            temperature_text = parts[1].replace("deg", "")
            temperature_degC = float(temperature_text)

            self.frequency_edit.setText(f"{f0_MHz:.6f}")
            self.temperature_edit.setText(f"{temperature_degC:.6f}")

        except ValueError:
            return

    def _read_optional_float(self, edit: QLineEdit, field_name: str) -> float | None:
        """
        Read an optional float from a line edit.
        """
        text = edit.text().strip()

        if not text:
            return None

        try:
            return float(text)
        except ValueError as exc:
            raise ValueError(f"Invalid value for {field_name}: {text!r}") from exc

    def _read_int(self, edit: QLineEdit, field_name: str) -> int:
        """
        Read an integer from a line edit.
        """
        text = edit.text().strip()

        try:
            return int(text)
        except ValueError as exc:
            raise ValueError(f"Invalid value for {field_name}: {text!r}") from exc

    def _read_float(self, edit: QLineEdit, field_name: str) -> float:
        """
        Read a float from a line edit.
        """
        text = edit.text().strip()

        try:
            return float(text)
        except ValueError as exc:
            raise ValueError(f"Invalid value for {field_name}: {text!r}") from exc

    def _read_remove_peaks(self) -> list[int]:
        """
        Read comma-separated peak indices.
        """
        text = self.remove_peaks_edit.text().strip()

        if not text:
            return []

        values = []

        for item in text.split(","):
            item = item.strip()

            if not item:
                continue

            values.append(int(item))

        return values

    def _update_options_from_fields(self) -> None:
        """
        Update bead-pull options from the dialog fields.
        """
        if hasattr(self.options, "use_S_output_for_BP"):
            self.options.use_S_output_for_BP = self.use_output_checkbox.isChecked()

        if hasattr(self.options, "n_zero"):
            self.options.n_zero = self._read_int(self.n_zero_edit, "n_zero")

        if hasattr(self.options, "threshold_fraction"):
            self.options.threshold_fraction = self._read_float(
                self.threshold_fraction_edit,
                "threshold fraction",
            )

        if hasattr(self.options, "smooth_size"):
            self.options.smooth_size = self._read_int(
                self.smooth_size_edit,
                "smooth size",
            )

        if hasattr(self.options, "phase_tolerance"):
            self.options.phase_tolerance = np.deg2rad(
                self._read_float(
                    self.phase_tolerance_deg_edit,
                    "phase tolerance",
                )
            )

        if hasattr(self.options, "remove_peaks"):
            self.options.remove_peaks = self._read_remove_peaks()

    def on_ok(self) -> None:
        """
        Validate and accept the dialog.
        """
        try:
            if self.filename is None:
                raise ValueError("Select a bead-pull CSV file first.")

            self._update_options_from_fields()

            self.f0_MHz_override = self._read_optional_float(
                self.frequency_edit,
                "Frequency [MHz]",
            )
            self.temperature_degC_override = self._read_optional_float(
                self.temperature_edit,
                "Temperature [°C]",
            )
            self.invert_rf_structure_parameters = self.invert_rf_checkbox.isChecked()
            self.invert_measurement_direction = self.invert_measurement_checkbox.isChecked()

        except ValueError as exc:
            QMessageBox.warning(self, "Invalid bead-pull input", str(exc))
            return

        self.accept()

    def on_preview(self) -> None:
        """
        Placeholder for preview behavior.
        """
        QMessageBox.information(
            self,
            "Preview",
            "Preview can be connected to a quick bead-pull plot later.",
        )

    def read_from_vna_placeholder(self) -> None:
        """
        Placeholder for VNA readout.
        """
        QMessageBox.information(
            self,
            "Read from VNA",
            "VNA readout can be connected here later.",
        )