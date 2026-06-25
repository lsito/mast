from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class MatlabStylePlotCanvas(FigureCanvas):
    """
    Matplotlib canvas styled to resemble the original MATLAB bead-pull GUI.

    The canvas displays one selected plot type and overlays all loaded
    bead-pull records.
    """

    def __init__(self, parent=None) -> None:
        """
        Initialize the plot canvas.
        """
        self.figure = Figure(tight_layout=True)
        self.axes = self.figure.add_subplot(111)
        super().__init__(self.figure)
        self.setParent(parent)
        self.legend_visible = False

    def clear_plot(self) -> None:
        """
        Clear the canvas.
        """
        self.axes.clear()
        self.axes.grid(True, alpha=0.35)
        self.draw_idle()

    def set_legend_visible(self, visible: bool) -> None:
        """
        Set whether the legend should be shown.
        """
        self.legend_visible = visible

    def plot_records(self, records: Iterable, plot_key: str) -> None:
        """
        Plot all records using the selected plot type.
        """
        records = list(records)

        self.axes.clear()
        self.axes.grid(True, alpha=0.35)

        for record in records:
            self._plot_single_record(record, plot_key)

        self._finish_plot(plot_key)
        self.draw_idle()

    def _plot_single_record(self, bdata, plot_key: str) -> None:
        """
        Plot one bead-pull record.
        """
        label = self._record_label(bdata)

        if plot_key == "df_to_tune":
            self._plot_cell_array(bdata.df2tune, scale=1e-6, label=label)

        elif plot_key == "phase_advance":
            self._plot_cell_array(bdata.phiadv, scale=1.0, label=label)

        elif plot_key == "s11_beadpull":
            if bdata.a is not None:
                x = np.arange(len(bdata.a))
                self.axes.plot(
                    x,
                    np.abs(bdata.a),
                    marker="o",
                    linestyle="None",
                    markerfacecolor="none",
                    label=label,
                )

        elif plot_key == "ds11_bp":
            self._plot_cell_array(bdata.ds11, scale=1e3, label=label)

        elif plot_key == "abs_ds11_bp":
            if bdata.ds11 is not None:
                self._plot_cell_array(np.abs(bdata.ds11), scale=1e3, label=label)

        elif plot_key == "abs_ds11_bp_z":
            if bdata.ds11global is not None:
                self._plot_cell_array(np.abs(bdata.ds11global), scale=1e3, label=label)

        elif plot_key == "mag_e":
            if bdata.Ebp is not None:
                self._plot_cell_array(np.abs(bdata.Ebp), scale=1.0, label=label)

        elif plot_key == "mag_peaks_e":
            if bdata.dref is not None:
                self._plot_cell_array(np.sqrt(np.abs(bdata.dref)), scale=1.0, label=label)

        elif plot_key == "zero_line":
            self._plot_zero_line(bdata, label)

        elif plot_key == "pm_abs_ds11":
            if bdata.ds11 is not None:
                y = np.abs(bdata.ds11) * 1e3
                x = np.arange(1, len(y) + 1)
                self.axes.plot(
                    x,
                    y,
                    marker="o",
                    linestyle="None",
                    markerfacecolor="none",
                    label=f"{label} +",
                )
                self.axes.plot(
                    x,
                    -y,
                    marker="o",
                    linestyle="None",
                    markerfacecolor="none",
                    label=f"{label} -",
                )

        elif plot_key == "phi_vs_freq":
            if bdata.gamma is not None:
                self._plot_cell_array(np.angle(bdata.gamma), scale=1.0, label=label)

        elif plot_key == "local_s11":
            if bdata.s11local is not None:
                x = np.arange(1, len(bdata.s11local) + 1)
                self.axes.plot(
                    x,
                    np.real(bdata.s11local) * 1e3,
                    marker="o",
                    linestyle="None",
                    markerfacecolor="none",
                    label=f"{label} real",
                )
                self.axes.plot(
                    x,
                    np.imag(bdata.s11local) * 1e3,
                    marker="o",
                    linestyle="None",
                    markerfacecolor="none",
                    label=f"{label} imag",
                )

        elif plot_key == "local_s11_cell":
            if bdata.s11local is not None:
                self._plot_cell_array(np.abs(bdata.s11local), scale=1e3, label=label)

        elif plot_key == "wbn":
            if bdata.B is not None:
                self._plot_cell_array(np.abs(bdata.B), scale=1.0, label=label, start_at_zero=True)

        elif plot_key == "wfn":
            if bdata.A is not None:
                self._plot_cell_array(np.abs(bdata.A), scale=1.0, label=label, start_at_zero=True)

        elif plot_key == "arg_ds11_bp_z":
            if bdata.ds11global is not None:
                self._plot_cell_array(np.angle(bdata.ds11global), scale=1.0, label=label)

        elif plot_key == "abs_arg_ds11_bp_z":
            if bdata.ds11global is not None:
                self._plot_cell_array(np.abs(np.angle(bdata.ds11global)), scale=1.0, label=label)

    def _plot_cell_array(
        self,
        values,
        scale: float,
        label: str,
        start_at_zero: bool = False,
    ) -> None:
        """
        Plot an array against cell number.
        """
        if values is None:
            return

        y = np.asarray(values) * scale

        if start_at_zero:
            x = np.arange(len(y))
        else:
            x = np.arange(1, len(y) + 1)

        self.axes.plot(
            x,
            y,
            marker="o",
            linestyle="None",
            markerfacecolor="none",
            label=label,
        )

    def _plot_zero_line(self, bdata, label: str) -> None:
        """
        Plot zero-line diagnostic data.
        """
        if bdata.aorg is not None:
            x = np.arange(len(bdata.aorg))
            self.axes.plot(x, np.real(bdata.aorg), label=f"{label} real(aorg)")

        if bdata.a_zero is not None:
            x = np.arange(len(bdata.a_zero))
            self.axes.plot(x, np.real(bdata.a_zero), label=f"{label} real(a_zero)")

    def _finish_plot(self, plot_key: str) -> None:
        """
        Apply labels, title, and legend after plotting.
        """
        titles = {
            "df_to_tune": ("df to tune", "Cell", "df to tune [MHz]"),
            "phase_advance": ("phase advance", "Cell interval", "Phase advance [deg]"),
            "s11_beadpull": ("S11 bead-pull", "Sample", "|S11|"),
            "ds11_bp": ("dS11 BP", "Cell", "dS11 [mU]"),
            "abs_ds11_bp": ("|dS11| BP", "Cell", "|dS11| [mU]"),
            "abs_ds11_bp_z": ("|dS11| BP (z)", "Cell", "|dS11 global| [mU]"),
            "mag_e": ("Mag(E)", "Cell", "|E|"),
            "mag_peaks_e": ("Mag(peaks(E))", "Cell", "sqrt(|dref|)"),
            "zero_line": ("0-Line", "Sample", "Real part"),
            "pm_abs_ds11": ("+/-|dS11|", "Cell", "+/-|dS11| [mU]"),
            "phi_vs_freq": ("phi v.s. freq", "Cell", "arg(gamma) [rad]"),
            "local_s11": ("local S11", "Cell", "local S11 [mU]"),
            "local_s11_cell": ("local S11(cell)", "Cell", "|local S11| [mU]"),
            "wbn": ("wbn", "Cell boundary", "|wbn|"),
            "wfn": ("wfn", "Cell boundary", "|wfn|"),
            "arg_ds11_bp_z": ("arg(dS11) BP (z)", "Cell", "arg(dS11 global) [rad]"),
            "abs_arg_ds11_bp_z": ("|arg(dS11) BP(z)|", "Cell", "|arg(dS11 global)| [rad]"),
        }

        title, xlabel, ylabel = titles.get(plot_key, ("Plot", "x", "y"))

        self.axes.set_title(title)
        self.axes.set_xlabel(xlabel)
        self.axes.set_ylabel(ylabel)

        if self.legend_visible:
            self.axes.legend()

    def _record_label(self, bdata) -> str:
        """
        Return a compact label for a bead-pull record.
        """
        if bdata.filename is None:
            return "record"

        return Path(bdata.filename).stem