from __future__ import annotations

from PySide6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.widgets import Cursor

from src.gui.beadpull_plots import plot_records


class MatlabStylePlotCanvas(QWidget):
    """
    Interactive Matplotlib plot widget.

    The axes and colorbar axes have fixed positions so the plotting area does
    not shift when switching between plot types.
    """

    def __init__(self, parent=None) -> None:
        """
        Initialize the plot widget.
        """
        super().__init__(parent)

        self.figure = Figure(facecolor="white")
        self.axes = self.figure.add_axes([0.10, 0.12, 0.74, 0.78])
        self.colorbar_axes = self.figure.add_axes([0.88, 0.12, 0.03, 0.78])
        self.colorbar_axes.set_visible(False)

        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.coordinate_label = QLabel("x: -, y: -")
        self.legend_visible = False

        self.cursor = Cursor(
            self.axes,
            useblit=True,
            horizOn=True,
            vertOn=True,
            linewidth=0.8,
            color="0.35",
        )

        self._build_ui()
        self._connect_events()
        self.clear_plot()

    def _build_ui(self) -> None:
        """
        Build the internal widget layout.
        """
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.coordinate_label.setObjectName("CoordinateLabel")

        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        layout.addWidget(self.coordinate_label)

    def _connect_events(self) -> None:
        """
        Connect Matplotlib events.
        """
        self.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)

    def _on_mouse_move(self, event) -> None:
        """
        Update coordinate readout when the mouse moves over the axes.
        """
        if event.inaxes != self.axes or event.xdata is None or event.ydata is None:
            self.coordinate_label.setText("x: -, y: -")
            return

        self.coordinate_label.setText(f"x: {event.xdata:.6g}, y: {event.ydata:.6g}")

    def set_legend_visible(self, visible: bool) -> None:
        """
        Set whether the legend should be visible.
        """
        self.legend_visible = visible

    def clear_plot(self) -> None:
        """
        Clear the plot.
        """
        self.axes.clear()
        self.colorbar_axes.clear()
        self.colorbar_axes.set_visible(False)
        self.axes.grid(True, linestyle="--", linewidth=0.7, alpha=0.35)
        self.canvas.draw_idle()

    def plot_records(self, records, plot_key: str) -> None:
        """
        Plot bead-pull records using the selected plot type.
        """
        plot_records(
            ax=self.axes,
            records=records,
            plot_key=plot_key,
            legend=self.legend_visible,
            colorbar_ax=self.colorbar_axes,
        )

        self.cursor = Cursor(
            self.axes,
            useblit=True,
            horizOn=True,
            vertOn=True,
            linewidth=0.8,
            color="0.35",
        )

        self.toolbar.update()
        self.canvas.draw_idle()