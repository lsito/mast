from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from src.core.tuning_simulator import TuningSimulationResult, TuningSimulatorModel


@dataclass(slots=True)
class SliderControl:
    """
    Widgets belonging to one tuning slider column.
    """

    cell_number: int
    slider: QSlider
    value_spinbox: QDoubleSpinBox
    min_spinbox: QDoubleSpinBox
    max_spinbox: QDoubleSpinBox


class TuningSimulatorWindow(QMainWindow):
    """
    MATLAB-like tuning simulator GUI.

    The computation is handled by `TuningSimulatorModel`.
    This class only owns the Qt controls and Matplotlib plotting.
    """

    slider_scale = 10.0

    def __init__(
        self,
        bdata,
        limits_mU: float = 100.0,
        dS11_start=None,
        sliders_for_these_cells=None,
        parent=None,
    ) -> None:
        """
        Initialize the tuning simulator window.
        """
        super().__init__(parent)

        self.bdata = bdata
        self.model = TuningSimulatorModel(
            bdata=bdata,
            limits_mU=limits_mU,
            dS11_start=dS11_start,
            sliders_for_these_cells=sliders_for_these_cells,
        )

        self.slider_controls: list[SliderControl] = []

        self.setWindowTitle("Tuning Simulations")
        self.resize(1260, 900)

        self._build_ui()
        self._apply_style()
        self._sync_controls_from_model()
        self.refresh_plot()

    def _build_ui(self) -> None:
        """
        Build the full simulator window.
        """
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(6)

        self.title_label = QLabel(self._title_text())
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setObjectName("TuningSimulatorTitle")

        main_layout.addWidget(self.title_label)
        main_layout.addWidget(self._build_plot_area(), stretch=1)
        main_layout.addWidget(self._build_slider_area())

    def _build_plot_area(self) -> QWidget:
        """
        Build the plot area with five axes.
        """
        widget = QWidget()

        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self.figure = Figure(facecolor=(0.95, 0.95, 0.95))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.toolbar = NavigationToolbar(self.canvas, self)

        self.ax_phiadv = self.figure.add_axes([0.04, 0.55, 0.45, 0.38])
        self.ax_ebp = self.figure.add_axes([0.04, 0.10, 0.45, 0.38])
        self.ax_wbn_abs = self.figure.add_axes([0.54, 0.72, 0.43, 0.21])
        self.ax_wbn_re = self.figure.add_axes([0.54, 0.41, 0.43, 0.21])
        self.ax_wbn_im = self.figure.add_axes([0.54, 0.10, 0.43, 0.21])

        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        return widget

    def _build_slider_area(self) -> QWidget:
        """
        Build the lower slider area.
        """
        wrapper = QWidget()
        wrapper.setObjectName("SliderAreaWrapper")

        layout = QVBoxLayout(wrapper)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        top_row = QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)
        top_row.setSpacing(6)

        self.global_limit_spinbox = QDoubleSpinBox()
        self.global_limit_spinbox.setRange(1.0, 1000.0)
        self.global_limit_spinbox.setDecimals(1)
        self.global_limit_spinbox.setSingleStep(10.0)
        self.global_limit_spinbox.setValue(float(self.model.limit_mU))
        self.global_limit_spinbox.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)

        apply_limit_button = QPushButton("Apply limit to all")
        zero_button = QPushButton("Zero all")
        current_button = QPushButton("Use current ds11")
        output_button = QPushButton("Use output correction")
        close_button = QPushButton("Close")

        apply_limit_button.clicked.connect(self.apply_global_limit_to_all)
        zero_button.clicked.connect(self.zero_all)
        current_button.clicked.connect(self.use_current_ds11)
        output_button.clicked.connect(self.use_output_correction)
        close_button.clicked.connect(self.close)

        top_row.addWidget(QLabel("Global limit [mU]"))
        top_row.addWidget(self.global_limit_spinbox)
        top_row.addWidget(apply_limit_button)
        top_row.addSpacing(10)
        top_row.addWidget(zero_button)
        top_row.addWidget(current_button)
        top_row.addWidget(output_button)
        top_row.addStretch()
        top_row.addWidget(close_button)

        layout.addLayout(top_row)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setMinimumHeight(270)

        slider_container = QWidget()
        self.slider_row_layout = QHBoxLayout(slider_container)
        self.slider_row_layout.setContentsMargins(4, 4, 4, 4)
        self.slider_row_layout.setSpacing(5)

        self._build_slider_columns()

        self.slider_row_layout.addStretch()
        scroll_area.setWidget(slider_container)

        layout.addWidget(scroll_area)

        return wrapper

    def _build_slider_columns(self) -> None:
        """
        Build one MATLAB-like vertical slider column per selected cell.
        """
        for cell_number in self.model.slider_cell_numbers:
            frame = QFrame()
            frame.setObjectName("SliderColumn")
            frame.setFixedWidth(38)

            layout = QVBoxLayout(frame)
            layout.setContentsMargins(2, 2, 2, 2)
            layout.setSpacing(2)

            cell_label = QLabel(f"{cell_number:d}")
            cell_label.setAlignment(Qt.AlignCenter)
            cell_label.setObjectName("CellNumberLabel")

            value_spinbox = self._make_small_spinbox(
                minimum=-1e4,
                maximum=1e4,
                value=0.0,
            )
            max_spinbox = self._make_small_spinbox(
                minimum=-1e4,
                maximum=1e4,
                value=float(self.model.limit_mU),
            )
            min_spinbox = self._make_small_spinbox(
                minimum=-1e4,
                maximum=1e4,
                value=-float(self.model.limit_mU),
            )

            slider = QSlider(Qt.Vertical)
            slider.setMinimum(self._to_slider_units(-float(self.model.limit_mU)))
            slider.setMaximum(self._to_slider_units(float(self.model.limit_mU)))
            slider.setValue(0)
            slider.setMinimumHeight(180)
            slider.setMaximumHeight(200)

            control = SliderControl(
                cell_number=int(cell_number),
                slider=slider,
                value_spinbox=value_spinbox,
                min_spinbox=min_spinbox,
                max_spinbox=max_spinbox,
            )

            value_spinbox.valueChanged.connect(
                lambda value, item=control: self._value_spinbox_changed(item, value)
            )
            min_spinbox.valueChanged.connect(
                lambda value, item=control: self._limit_spinbox_changed(item)
            )
            max_spinbox.valueChanged.connect(
                lambda value, item=control: self._limit_spinbox_changed(item)
            )
            slider.valueChanged.connect(
                lambda value, item=control: self._slider_changed(item, value)
            )

            layout.addWidget(cell_label)
            layout.addWidget(value_spinbox)
            layout.addWidget(max_spinbox)
            layout.addWidget(slider, alignment=Qt.AlignHCenter)
            layout.addWidget(min_spinbox)

            self.slider_controls.append(control)
            self.slider_row_layout.addWidget(frame)

    def _make_small_spinbox(
        self,
        minimum: float,
        maximum: float,
        value: float,
    ) -> QDoubleSpinBox:
        """
        Create a compact spinbox similar to MATLAB edit boxes.
        """
        spinbox = QDoubleSpinBox()
        spinbox.setRange(minimum, maximum)
        spinbox.setDecimals(1)
        spinbox.setSingleStep(0.1)
        spinbox.setValue(value)
        spinbox.setAlignment(Qt.AlignRight)
        spinbox.setFixedWidth(34)
        spinbox.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)

        return spinbox

    def _apply_style(self) -> None:
        """
        Apply MATLAB-like styling.
        """
        self.setStyleSheet(
            """
            QMainWindow {
                background-color: #f2f2f2;
            }

            QLabel#TuningSimulatorTitle {
                background-color: #f2f2f2;
                color: #111827;
                font-size: 8pt;
                padding: 2px;
            }

            QWidget#SliderAreaWrapper {
                background-color: #f2f2f2;
            }

            QFrame#SliderColumn {
                background-color: #f2f2f2;
                border: none;
            }

            QLabel#CellNumberLabel {
                color: #111827;
                font-size: 8pt;
            }

            QDoubleSpinBox {
                background-color: #ffffff;
                border: 1px solid #b8b8b8;
                padding: 1px;
                font-size: 8pt;
            }

            QPushButton {
                background-color: #e5e7eb;
                color: #111827;
                border: 1px solid #9ca3af;
                border-radius: 3px;
                padding: 4px 8px;
                font-size: 8pt;
            }

            QPushButton:hover {
                background-color: #dbeafe;
            }

            QScrollArea {
                background-color: #f2f2f2;
                border: none;
            }
            """
        )

    def _title_text(self) -> str:
        """
        Return the window title label.
        """
        if getattr(self.bdata, "filename", None) is None:
            return "BP_file N: record"

        return f'BP_file: "{Path(self.bdata.filename).name}"'

    def _to_slider_units(self, value_mU: float) -> int:
        """
        Convert mU to integer slider units.
        """
        return int(round(float(value_mU) * self.slider_scale))

    def _from_slider_units(self, value: int) -> float:
        """
        Convert integer slider units to mU.
        """
        return float(value) / self.slider_scale

    def _slider_changed(self, control: SliderControl, value: int) -> None:
        """
        Handle a vertical slider movement.
        """
        value_mU = self._from_slider_units(value)

        control.value_spinbox.blockSignals(True)
        control.value_spinbox.setValue(value_mU)
        control.value_spinbox.blockSignals(False)

        self.model.set_cell_value(control.cell_number, value_mU)
        self.refresh_plot()

    def _value_spinbox_changed(self, control: SliderControl, value: float) -> None:
        """
        Handle direct editing of the value box.
        """
        min_value = control.min_spinbox.value()
        max_value = control.max_spinbox.value()

        clipped_value = float(np.clip(value, min_value, max_value))

        if clipped_value != value:
            control.value_spinbox.blockSignals(True)
            control.value_spinbox.setValue(clipped_value)
            control.value_spinbox.blockSignals(False)

        control.slider.blockSignals(True)
        control.slider.setValue(self._to_slider_units(clipped_value))
        control.slider.blockSignals(False)

        self.model.set_cell_value(control.cell_number, clipped_value)
        self.refresh_plot()

    def _limit_spinbox_changed(self, control: SliderControl) -> None:
        """
        Handle direct editing of the min/max slider limits.
        """
        min_value = control.min_spinbox.value()
        max_value = control.max_spinbox.value()

        if min_value >= max_value:
            return

        current_value = control.value_spinbox.value()
        clipped_value = float(np.clip(current_value, min_value, max_value))

        control.slider.blockSignals(True)
        control.value_spinbox.blockSignals(True)

        control.slider.setRange(
            self._to_slider_units(min_value),
            self._to_slider_units(max_value),
        )
        control.value_spinbox.setRange(min_value, max_value)
        control.value_spinbox.setValue(clipped_value)
        control.slider.setValue(self._to_slider_units(clipped_value))

        control.slider.blockSignals(False)
        control.value_spinbox.blockSignals(False)

        self.model.set_cell_value(control.cell_number, clipped_value)
        self.refresh_plot()

    def apply_global_limit_to_all(self) -> None:
        """
        Apply the global +/- limit to every slider column.
        """
        limit = float(self.global_limit_spinbox.value())

        for control in self.slider_controls:
            control.min_spinbox.blockSignals(True)
            control.max_spinbox.blockSignals(True)

            control.min_spinbox.setValue(-limit)
            control.max_spinbox.setValue(limit)

            control.min_spinbox.blockSignals(False)
            control.max_spinbox.blockSignals(False)

            self._limit_spinbox_changed(control)

        self.refresh_plot()

    def zero_all(self) -> None:
        """
        Reset all dS11 values to zero.
        """
        self.model.zero_all()
        self._sync_controls_from_model()
        self.refresh_plot()

    def use_current_ds11(self) -> None:
        """
        Initialize dS11 from the currently computed record ds11.
        """
        try:
            self.model.use_record_ds11()
        except Exception as exc:
            QMessageBox.warning(self, "Tuning Simulator", str(exc))
            return

        self._sync_controls_from_model()
        self.refresh_plot()

    def use_output_correction(self) -> None:
        """
        Initialize dS11 using ds11_0 and ds11_1 on the last two cells.
        """
        try:
            self.model.use_output_correction()
        except Exception as exc:
            QMessageBox.warning(self, "Tuning Simulator", str(exc))
            return

        self._sync_controls_from_model()
        self.refresh_plot()

    def _sync_controls_from_model(self) -> None:
        """
        Copy model values into the visible slider controls.
        """
        for control in self.slider_controls:
            value_mU = float(self.model.dS11_mU[control.cell_number - 1])

            min_value = control.min_spinbox.value()
            max_value = control.max_spinbox.value()
            value_mU = float(np.clip(value_mU, min_value, max_value))

            control.slider.blockSignals(True)
            control.value_spinbox.blockSignals(True)

            control.value_spinbox.setRange(min_value, max_value)
            control.value_spinbox.setValue(value_mU)
            control.slider.setRange(
                self._to_slider_units(min_value),
                self._to_slider_units(max_value),
            )
            control.slider.setValue(self._to_slider_units(value_mU))

            control.slider.blockSignals(False)
            control.value_spinbox.blockSignals(False)

            self.model.set_cell_value(control.cell_number, value_mU)

    def refresh_plot(self) -> None:
        """
        Recompute and redraw the simulator plots.
        """
        try:
            result = self.model.simulate()
        except Exception as exc:
            QMessageBox.warning(self, "Tuning Simulator", str(exc))
            return

        self._plot_result(result)

    def _plot_result(self, result: TuningSimulationResult) -> None:
        """
        Draw all five MATLAB-like simulator plots.
        """
        self._clear_axes()

        phiadv_original = result.phiadv_original
        phiadv_c = result.phiadv_c

        ebp_original = result.ebp_original
        ebp_c = result.ebp_c

        wbn_original = result.wbn_original
        wbn_c = result.wbn_c

        x_phi_original = np.arange(1, len(phiadv_original) + 1)
        x_phi_c = np.arange(1, len(phiadv_c) + 1)

        self.ax_phiadv.plot(
            x_phi_original,
            phiadv_original,
            "bx-",
            linewidth=0.8,
            markersize=4.5,
        )
        self.ax_phiadv.plot(
            x_phi_c,
            phiadv_c + result.ddphi,
            "rx-",
            linewidth=0.8,
            markersize=4.5,
        )

        if len(phiadv_original) > 1:
            self.ax_phiadv.axhline(
                np.mean(phiadv_original[1:]),
                color="blue",
                linestyle=":",
                linewidth=0.8,
            )

        if len(phiadv_c) > 1:
            self.ax_phiadv.axhline(
                np.mean(phiadv_c[1:]) + result.ddphi,
                color="red",
                linestyle=":",
                linewidth=0.8,
            )

        x_ebp = np.arange(1, len(ebp_original) + 1)

        self.ax_ebp.plot(
            x_ebp,
            np.abs(ebp_original),
            "bx-",
            linewidth=0.8,
            markersize=4.5,
        )
        self.ax_ebp.plot(
            x_ebp,
            np.abs(ebp_c),
            "rx-",
            linewidth=0.8,
            markersize=4.5,
        )

        x_wbn = np.arange(1, len(wbn_original) + 1)

        self.ax_wbn_abs.plot(
            x_wbn,
            np.abs(wbn_original) * 1e3,
            "bo-",
            linewidth=0.8,
            markersize=4.5,
            markerfacecolor="none",
        )
        self.ax_wbn_abs.plot(
            x_wbn,
            np.abs(wbn_c) * 1e3,
            "ro-",
            linewidth=0.8,
            markersize=4.5,
            markerfacecolor="none",
        )

        self.ax_wbn_re.plot(
            x_wbn,
            np.real(wbn_original) * 1e3,
            "bo-",
            linewidth=0.8,
            markersize=4.5,
            markerfacecolor="none",
        )
        self.ax_wbn_re.plot(
            x_wbn,
            np.real(wbn_c) * 1e3,
            "ro-",
            linewidth=0.8,
            markersize=4.5,
            markerfacecolor="none",
        )

        self.ax_wbn_im.plot(
            x_wbn,
            np.imag(wbn_original) * 1e3,
            "bo-",
            linewidth=0.8,
            markersize=4.5,
            markerfacecolor="none",
        )
        self.ax_wbn_im.plot(
            x_wbn,
            np.imag(wbn_c) * 1e3,
            "ro-",
            linewidth=0.8,
            markersize=4.5,
            markerfacecolor="none",
        )

        self.ax_phiadv.set_ylabel("d_phi  [°]")
        self.ax_ebp.set_ylabel("Abs(E_peaks)  [a.u.]")
        self.ax_wbn_abs.set_ylabel("Abs(wbn)  [mU]")
        self.ax_wbn_re.set_ylabel("Re(wbn)  [mU]")
        self.ax_wbn_im.set_ylabel("Im(wbn)  [mU]")

        self.ax_phiadv.set_xlim(0, self.model.number_of_cells + 2)
        self.ax_ebp.set_xlim(0, self.model.number_of_cells + 2)
        self.ax_wbn_abs.set_xlim(0, self.model.number_of_cells + 2)
        self.ax_wbn_re.set_xlim(0, self.model.number_of_cells + 2)
        self.ax_wbn_im.set_xlim(0, self.model.number_of_cells + 2)

        for axis in [
            self.ax_phiadv,
            self.ax_ebp,
            self.ax_wbn_abs,
            self.ax_wbn_re,
            self.ax_wbn_im,
        ]:
            self._style_axis(axis)

        self.canvas.draw_idle()

    def _clear_axes(self) -> None:
        """
        Clear all axes.
        """
        for axis in [
            self.ax_phiadv,
            self.ax_ebp,
            self.ax_wbn_abs,
            self.ax_wbn_re,
            self.ax_wbn_im,
        ]:
            axis.clear()

    def _style_axis(self, axis) -> None:
        """
        Apply MATLAB-like axis styling.
        """
        axis.grid(True, linewidth=0.4, alpha=0.5)
        axis.tick_params(direction="in", top=True, right=True, labelsize=8)

        for spine in axis.spines.values():
            spine.set_linewidth(0.8)
            spine.set_color("black")