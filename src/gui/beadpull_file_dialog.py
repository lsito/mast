from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from src.data_models.bead_config import BeadpullConfig
from src.io_utils.vna_to_csv import read_vna_to_csv


class BeadpullFileDialog(QDialog):
    """
    Dialog for adding one or more bead-pull files.

    The dialog selects bead-pull CSV files and allows shared control of
    frequency override, temperature override, measurement direction, RF
    inversion, and bead-pull analysis options.

    The VNA button reads the current VNA traces and saves them as a CSV file,
    then selects that CSV as the bead-pull file.
    """

    def __init__(
        self,
        default_options: BeadpullConfig | None = None,
        default_rf_inverse: bool = False,
        parent=None,
    ) -> None:
        """
        Initialize the bead-pull file dialog.
        """
        super().__init__(parent)

        self.setWindowTitle("Add bead-pull files")
        self.setModal(True)
        self.resize(620, 570)

        self.filenames: list[str] = []
        self.filename: str | None = None
        self.options = deepcopy(default_options) if default_options is not None else BeadpullConfig()

        self.f0_MHz_override: float | None = None
        self.temperature_degC_override: float | None = None
        self.invert_rf_structure_parameters = bool(default_rf_inverse)
        self.invert_measurement_direction = False

        self._build_ui()
        self._apply_style()
        self._load_options_to_fields()

    def _build_ui(self) -> None:
        """
        Build the dialog user interface.
        """
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(18, 18, 18, 18)
        main_layout.setSpacing(14)

        main_layout.addLayout(self._build_file_selection_section())
        main_layout.addWidget(self._build_beadpull_options_section())
        main_layout.addWidget(self._build_override_parameters_section())
        main_layout.addStretch()
        main_layout.addLayout(self._build_bottom_button_row())

    def _apply_style(self) -> None:
        """
        Apply section-card styling for the dialog.
        """
        self.setStyleSheet(
            self.styleSheet()
            + """
            QFrame#BeadpullDialogSection {
                background-color: #ffffff;
                border: 1px solid #e5e7eb;
                border-radius: 10px;
            }

            QLabel#BeadpullDialogSectionTitle {
                font-weight: 700;
                font-size: 11pt;
                color: #374151;
                padding: 0px;
                margin: 0px;
            }
            """
        )

    def _make_section_frame(self, title: str) -> tuple[QFrame, QVBoxLayout]:
        """
        Create a white section frame with the title fully inside the rectangle.
        """
        frame = QFrame()
        frame.setObjectName("BeadpullDialogSection")

        layout = QVBoxLayout(frame)
        layout.setContentsMargins(16, 14, 16, 16)
        layout.setSpacing(12)

        title_label = QLabel(title)
        title_label.setObjectName("BeadpullDialogSectionTitle")
        title_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        layout.addWidget(title_label)

        return frame, layout

    def _build_file_selection_section(self) -> QVBoxLayout:
        """
        Build the file-selection section.
        """
        section_layout = QVBoxLayout()
        section_layout.setSpacing(10)

        file_row = QHBoxLayout()

        self.filename_edit = QLineEdit()
        self.filename_edit.setReadOnly(True)
        self.filename_edit.setPlaceholderText("Select one or more bead-pull CSV files")

        self.browse_button = QPushButton("Browse...")
        self.browse_button.clicked.connect(self.browse_files)

        file_row.addWidget(self.filename_edit)
        file_row.addWidget(self.browse_button)

        vna_row = QHBoxLayout()
        vna_row.addStretch()

        self.read_vna_button = QPushButton("Read from VNA and save csv...")
        self.read_vna_button.clicked.connect(self.read_from_vna_and_save_csv)

        vna_row.addWidget(self.read_vna_button)
        vna_row.addStretch()

        section_layout.addLayout(file_row)
        section_layout.addLayout(vna_row)

        return section_layout

    def _build_beadpull_options_section(self) -> QFrame:
        """
        Build the bead-pull analysis options section.
        """
        frame, layout = self._make_section_frame("Beadpull Options")

        checkbox_column = QVBoxLayout()
        checkbox_column.setSpacing(8)
        checkbox_column.setAlignment(Qt.AlignHCenter)

        self.use_output_checkbox = QCheckBox("use output S-parameters")
        self.invert_rf_checkbox = QCheckBox("invert RF structure parameters")
        self.invert_measurement_checkbox = QCheckBox("invert measurement direction")

        self.use_output_checkbox.setMinimumWidth(260)
        self.invert_rf_checkbox.setMinimumWidth(260)
        self.invert_measurement_checkbox.setMinimumWidth(260)

        checkbox_column.addWidget(self.use_output_checkbox, alignment=Qt.AlignHCenter)
        checkbox_column.addWidget(self.invert_rf_checkbox, alignment=Qt.AlignHCenter)
        checkbox_column.addWidget(self.invert_measurement_checkbox, alignment=Qt.AlignHCenter)

        options_grid = QGridLayout()
        options_grid.setHorizontalSpacing(14)
        options_grid.setVerticalSpacing(10)

        self.n_zero_edit = QLineEdit()
        self.threshold_fraction_edit = QLineEdit()
        self.smooth_size_edit = QLineEdit()
        self.phase_tolerance_deg_edit = QLineEdit()
        self.remove_peaks_edit = QLineEdit()

        options_grid.addWidget(QLabel("n_zero"), 0, 0)
        options_grid.addWidget(self.n_zero_edit, 0, 1)

        options_grid.addWidget(QLabel("Threshold fraction"), 1, 0)
        options_grid.addWidget(self.threshold_fraction_edit, 1, 1)

        options_grid.addWidget(QLabel("Smooth size"), 2, 0)
        options_grid.addWidget(self.smooth_size_edit, 2, 1)

        options_grid.addWidget(QLabel("Phase tolerance [deg]"), 3, 0)
        options_grid.addWidget(self.phase_tolerance_deg_edit, 3, 1)

        options_grid.addWidget(QLabel("Remove peaks"), 4, 0)
        options_grid.addWidget(self.remove_peaks_edit, 4, 1)

        options_grid.setColumnStretch(0, 0)
        options_grid.setColumnStretch(1, 1)

        layout.addLayout(checkbox_column)
        layout.addSpacing(4)
        layout.addLayout(options_grid)

        return frame

    def _build_override_parameters_section(self) -> QFrame:
        """
        Build the optional override-parameter section.
        """
        frame, layout = self._make_section_frame("Override Parameters")

        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(10)

        self.atmosphere_edit = QLineEdit()
        self.wire_detuning_edit = QLineEdit()
        self.temperature_edit = QLineEdit()
        self.frequency_edit = QLineEdit()

        self.temperature_edit.setPlaceholderText("from filename")
        self.frequency_edit.setPlaceholderText("from filename")

        grid.addWidget(QLabel("Atmosphere"), 0, 0)
        grid.addWidget(self.atmosphere_edit, 0, 1)

        grid.addWidget(QLabel("Wire detuning"), 1, 0)
        grid.addWidget(self.wire_detuning_edit, 1, 1)

        grid.addWidget(QLabel("Temperature [°C]"), 2, 0)
        grid.addWidget(self.temperature_edit, 2, 1)

        grid.addWidget(QLabel("Frequency [MHz]"), 3, 0)
        grid.addWidget(self.frequency_edit, 3, 1)

        grid.setColumnStretch(0, 0)
        grid.setColumnStretch(1, 1)

        layout.addLayout(grid)

        return frame

    def _build_bottom_button_row(self) -> QHBoxLayout:
        """
        Build the centered bottom button row.
        """
        button_row = QHBoxLayout()
        button_row.addStretch()

        self.ok_button = QPushButton("OK")
        self.preview_button = QPushButton("Preview")
        self.cancel_button = QPushButton("Cancel")

        self.ok_button.setMinimumWidth(92)
        self.preview_button.setMinimumWidth(92)
        self.cancel_button.setMinimumWidth(92)

        self.ok_button.clicked.connect(self.on_ok)
        self.preview_button.clicked.connect(self.on_preview)
        self.cancel_button.clicked.connect(self.reject)

        button_row.addWidget(self.ok_button)
        button_row.addSpacing(10)
        button_row.addWidget(self.preview_button)
        button_row.addSpacing(10)
        button_row.addWidget(self.cancel_button)
        button_row.addStretch()

        return button_row

    def _load_options_to_fields(self) -> None:
        """
        Load bead-pull options into the dialog fields.
        """
        self.use_output_checkbox.setChecked(
            bool(getattr(self.options, "use_S_output_for_BP", False))
        )

        self.invert_rf_checkbox.setChecked(bool(self.invert_rf_structure_parameters))
        self.invert_measurement_checkbox.setChecked(bool(self.invert_measurement_direction))

        self.n_zero_edit.setText(str(getattr(self.options, "n_zero", 30)))
        self.threshold_fraction_edit.setText(
            str(getattr(self.options, "threshold_fraction", 0.15))
        )
        self.smooth_size_edit.setText(str(getattr(self.options, "smooth_size", 5)))

        phase_tolerance = float(getattr(self.options, "phase_tolerance", np.deg2rad(5.0)))
        self.phase_tolerance_deg_edit.setText(f"{np.rad2deg(phase_tolerance):.6g}")

        remove_peaks = getattr(self.options, "remove_peaks", [])
        self.remove_peaks_edit.setText(",".join(str(v) for v in remove_peaks))

    def browse_files(self) -> None:
        """
        Select one or more bead-pull CSV files.
        """
        filenames, _ = QFileDialog.getOpenFileNames(
            self,
            "Select bead-pull CSV files",
            "",
            "CSV files (*.csv);;All files (*)",
        )

        if not filenames:
            return

        self.filenames = list(filenames)
        self.filename = self.filenames[0]
        self._update_filename_display()
        self._try_fill_frequency_and_temperature_from_selection()

    def browse_file(self) -> None:
        """
        Backward-compatible alias for selecting bead-pull files.
        """
        self.browse_files()

    def read_from_vna_and_save_csv(self) -> None:
        """
        Read the VNA, save a CSV file, and select it as the bead-pull file.
        """
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save VNA bead-pull CSV",
            "",
            "CSV files (*.csv);;All files (*)",
        )

        if not filename:
            return

        if Path(filename).suffix.lower() != ".csv":
            filename = f"{filename}.csv"

        parent = self.parent()
        ip_address = getattr(parent, "vna_ip_address", "128.11.11.11")
        port = getattr(parent, "vna_port", 5025)

        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.read_vna_button.setEnabled(False)

        try:
            goodread = read_vna_to_csv(
                filename=filename,
                ip_address=ip_address,
                port=port,
            )
        except Exception as exc:
            QApplication.restoreOverrideCursor()
            self.read_vna_button.setEnabled(True)
            QMessageBox.warning(
                self,
                "Read from VNA",
                f"VNA read failed.\n\n{exc}",
            )
            return

        QApplication.restoreOverrideCursor()
        self.read_vna_button.setEnabled(True)

        if not goodread:
            QMessageBox.warning(
                self,
                "Read from VNA",
                "The VNA read did not complete successfully.",
            )
            return

        self.filenames = [filename]
        self.filename = filename
        self._update_filename_display()
        self._try_fill_frequency_and_temperature_from_selection()

        QMessageBox.information(
            self,
            "Read from VNA",
            f"VNA data saved and selected:\n\n{filename}",
        )

    def _update_filename_display(self) -> None:
        """
        Show the selected filename or a compact multiple-file summary.
        """
        if len(self.filenames) == 0:
            self.filename_edit.clear()
            return

        if len(self.filenames) == 1:
            self.filename_edit.setText(self.filenames[0])
            return

        first_name = Path(self.filenames[0]).name
        self.filename_edit.setText(f"{len(self.filenames)} files selected, first: {first_name}")
        self.filename_edit.setToolTip("\n".join(self.filenames))

    def _try_fill_frequency_and_temperature_from_selection(self) -> None:
        """
        Fill frequency and temperature only when exactly one file is selected.
        """
        if len(self.filenames) != 1:
            self.frequency_edit.clear()
            self.temperature_edit.clear()
            return

        self._try_fill_frequency_and_temperature_from_filename(self.filenames[0])

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

            try:
                values.append(int(item))
            except ValueError as exc:
                raise ValueError(f"Invalid peak index in Remove peaks: {item!r}") from exc

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
            if len(self.filenames) == 0:
                raise ValueError("Select at least one bead-pull CSV file first.")

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