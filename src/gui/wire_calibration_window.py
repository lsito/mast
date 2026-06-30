from __future__ import annotations

from pathlib import Path

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.core.wire_calibration import calculate_wire_calibration


class WireCalibrationWindow(QDialog):
    """
    Wire calibration GUI.

    The calculation is performed by src.core.wire_calibration.
    This GUI only handles file selection, user inputs, plotting.
    """

    def __init__(self, parent=None) -> None:
        """
        Initialize the wire calibration window.
        """
        super().__init__(parent)

        self.setWindowTitle("Wire Calibration Tool")
        self.resize(520, 720)

        self.last_path = ""
        self.result = None

        self._build_ui()
        self._apply_style()

    def _build_ui(self) -> None:
        """
        Build the user interface.
        """
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(14, 14, 14, 14)
        main_layout.setSpacing(10)

        main_layout.addLayout(self._build_file_section())
        main_layout.addLayout(self._build_frequency_section())
        main_layout.addWidget(self._build_plot_section(), stretch=1)
        main_layout.addLayout(self._build_bottom_buttons())

    def _build_file_section(self) -> QGridLayout:
        """
        Build the S4P file selector fields.
        """
        layout = QGridLayout()
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(6)

        self.with_wire_edit = QLineEdit()
        self.with_wire_edit.setPlaceholderText("S4P file with wire")

        self.without_wire_edit = QLineEdit()
        self.without_wire_edit.setPlaceholderText("S4P file without wire")

        with_wire_button = QPushButton("Browse...")
        without_wire_button = QPushButton("Browse...")

        with_wire_button.clicked.connect(self.browse_with_wire)
        without_wire_button.clicked.connect(self.browse_without_wire)

        layout.addWidget(QLabel("S4P file with wire"), 0, 0, 1, 2)
        layout.addWidget(self.with_wire_edit, 1, 0)
        layout.addWidget(with_wire_button, 1, 1)

        layout.addWidget(QLabel("S4P file without wire"), 2, 0, 1, 2)
        layout.addWidget(self.without_wire_edit, 3, 0)
        layout.addWidget(without_wire_button, 3, 1)

        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 0)

        return layout

    def _build_frequency_section(self) -> QGridLayout:
        """
        Build frequency and calculation controls.
        """
        layout = QGridLayout()
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(6)

        self.center_mhz_edit = QLineEdit()
        self.center_mhz_edit.setPlaceholderText("MHz")

        self.span_mhz_edit = QLineEdit()
        self.span_mhz_edit.setPlaceholderText("MHz")

        self.s_parameter_combo = QComboBox()
        self.s_parameter_combo.addItems(["S21", "S12", "S11", "S22"])

        self.format_combo = QComboBox()
        self.format_combo.addItems(["phase", "mag"])

        calculate_button = QPushButton("Calculate")
        calculate_button.clicked.connect(self.calculate)

        layout.addWidget(QLabel("Frequency of interest"), 0, 0, 1, 8)

        layout.addWidget(QLabel("Center [MHz]"), 1, 0)
        layout.addWidget(self.center_mhz_edit, 1, 1)

        layout.addWidget(QLabel("Span [MHz]"), 1, 2)
        layout.addWidget(self.span_mhz_edit, 1, 3)

        layout.addWidget(QLabel("S-param"), 1, 4)
        layout.addWidget(self.s_parameter_combo, 1, 5)

        layout.addWidget(QLabel("Format"), 1, 6)
        layout.addWidget(self.format_combo, 1, 7)

        layout.addWidget(calculate_button, 2, 0, 1, 8)

        return layout

    def _build_plot_section(self) -> QWidget:
        """
        Build the Matplotlib plot area.
        """
        widget = QWidget()

        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.figure = Figure(facecolor=(0.95, 0.95, 0.95))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        self.ax_top = self.figure.add_axes([0.13, 0.58, 0.80, 0.30])
        self.ax_bottom = self.figure.add_axes([0.13, 0.12, 0.80, 0.34])

        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        return widget

    def _build_bottom_buttons(self) -> QHBoxLayout:
        """
        Build the bottom command buttons.
        """
        layout = QHBoxLayout()

        close_button = QPushButton("Close")

        close_button.clicked.connect(self.close)

        layout.addStretch()
        layout.addWidget(close_button)

        return layout

    def _apply_style(self) -> None:
        """
        Apply simple MATLAB-like styling.
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

            QLineEdit,
            QComboBox {
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
            """
        )

    def browse_with_wire(self) -> None:
        """
        Select the S4P file measured with the wire.
        """
        filename = self._browse_file("S4P file with wire")

        if filename:
            self.with_wire_edit.setText(filename)

    def browse_without_wire(self) -> None:
        """
        Select the S4P file measured without the wire.
        """
        filename = self._browse_file("S4P file without wire")

        if filename:
            self.without_wire_edit.setText(filename)

    def _browse_file(self, title: str) -> str:
        """
        Open a file browser for S4P files.
        """
        filename, _ = QFileDialog.getOpenFileName(
            self,
            title,
            self.last_path,
            "S4P files (*.s4p);;All files (*)",
        )

        if filename:
            self.last_path = str(Path(filename).parent)

        return filename

    def calculate(self) -> None:
        """
        Run the wire calibration calculation.
        """
        try:
            self.result = calculate_wire_calibration(
                file_without_wire=self.without_wire_edit.text().strip(),
                file_with_wire=self.with_wire_edit.text().strip(),
                center_mhz=float(self.center_mhz_edit.text().strip()),
                span_mhz=float(self.span_mhz_edit.text().strip()),
                sflag=self.s_parameter_combo.currentText(),
                formatflag=self.format_combo.currentText(),
            )
        except Exception as exc:
            QMessageBox.warning(self, "Wire Calibration", str(exc))
            return

        self._plot_result()

    def _plot_result(self) -> None:
        """
        Plot phase/magnitude comparison and wire-removal frequency shift.
        """
        if self.result is None:
            return

        result = self.result

        self.ax_top.clear()
        self.ax_bottom.clear()
        self.figure.legends.clear()

        formatflag = result["formatflag"].lower()

        if formatflag == "phase":
            y_label_top = f'Phase {result["sflag"]} [deg]'
            y_without_wire = result["py"][:, 0]
            y_with_wire = result["pyy_aligned"][:, 1]
        else:
            y_label_top = f'Abs {result["sflag"]}'
            y_without_wire = abs(result["ss"][:, 0])
            y_with_wire = abs(result["ps"][:, 1])

        line_no, = self.ax_top.plot(
            result["px"][:, 0],
            y_without_wire,
            "b-",
            linewidth=1.0,
            label="without wire",
        )

        line_wire, = self.ax_top.plot(
            result["pxx"][:, 1],
            y_with_wire,
            "r.",
            markersize=4.0,
            label="with wire",
        )

        self.ax_bottom.plot(
            result["pf"][:, 0] / 1e9,
            result["df_removewire"] / 1e6,
            "b-",
            linewidth=1.0,
        )

        x_min = (result["fop"] - result["fsp"] / 2.0) / 1e9
        x_max = (result["fop"] + result["fsp"] / 2.0) / 1e9

        self.ax_top.set_xlim(x_min, x_max)
        self.ax_bottom.set_xlim(x_min, x_max)

        self.ax_top.set_ylabel(y_label_top)
        self.ax_bottom.set_ylabel("df remove wire [MHz]")
        self.ax_bottom.set_xlabel("f [GHz]")

        self.figure.legend(
            handles=[line_no, line_wire],
            labels=["without wire", "with wire"],
            loc="upper center",
            bbox_to_anchor=(0.5, 0.985),
            ncol=2,
            frameon=True,
            fontsize=8,
        )

        for axis in [self.ax_top, self.ax_bottom]:
            axis.grid(True, linewidth=0.4, alpha=0.5)
            axis.tick_params(direction="in", top=True, right=True, labelsize=8)

            for spine in axis.spines.values():
                spine.set_linewidth(0.8)
                spine.set_color("black")

        self.canvas.draw_idle()