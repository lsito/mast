from __future__ import annotations

from pathlib import Path

import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QFileDialog,
    QButtonGroup,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from src.core.beadpull_analyzer import BeadPullAnalyzer
from src.data_models.bead_config import BeadpullConfig
from src.data_models.bead_record import BeadpullRecord
from src.data_models.meas_config import MeasurementConfig
from src.data_models.rf_structure import RFStructureParams
from src.gui.meas_config_dialog import MeasurementConfigDialog
from src.gui.plot_canvas import MatlabStylePlotCanvas
from src.gui.structure_config_dialog import RFStructureLoaderDialog


class MainWindow(QMainWindow):
    """
    MATLAB-like main window for CLIC bead-pull offline analysis.

    The layout mirrors the original MATLAB application:

    a large plot on the left, numerical result lists and file controls on the
    right, and radio buttons for selecting the displayed plot.
    """

    def __init__(self) -> None:
        """
        Initialize the main window.
        """
        super().__init__()

        self.setWindowTitle("CLIC Bead-pull Offline Analysis")
        self.resize(1450, 900)

        self.RF_params: RFStructureParams | None = None
        self.Meas_params = MeasurementConfig()
        self.BP_options = BeadpullConfig()

        self.analyzer = BeadPullAnalyzer()
        self.records: list[BeadpullRecord] = []
        self.current_record_index: int | None = None
        self.legend_visible = False
        self.current_plot_key = "df_to_tune"

        self._build_actions()
        self._build_menu()
        self._build_ui()

        self.statusBar().showMessage("Ready")

    def _build_actions(self) -> None:
        """
        Build menu actions.
        """
        self.action_structure_design = QAction("Structure RF Design...", self)
        self.action_structure_design.triggered.connect(self.open_structure_dialog)

        self.action_measurement_condition = QAction("Measurement Condition...", self)
        self.action_measurement_condition.triggered.connect(self.open_measurement_dialog)

        self.action_beadpull_files = QAction("Bead-pull files...", self)
        self.action_beadpull_files.triggered.connect(self.add_beadpull_file)

        self.action_exit = QAction("Exit", self)
        self.action_exit.triggered.connect(self.close)

        self.action_about = QAction("About", self)
        self.action_about.triggered.connect(self.show_about)

    def _build_menu(self) -> None:
        """
        Build MATLAB-like menu bar.
        """
        menu_bar = self.menuBar()

        file_menu = menu_bar.addMenu("File")
        file_menu.addAction(self.action_structure_design)
        file_menu.addAction(self.action_measurement_condition)
        file_menu.addAction(self.action_beadpull_files)
        file_menu.addSeparator()
        file_menu.addAction(self.action_exit)

        edit_menu = menu_bar.addMenu("Edit")

        tools_menu = menu_bar.addMenu("Tools")

        help_menu = menu_bar.addMenu("Help")
        help_menu.addAction(self.action_about)

        edit_menu.setEnabled(True)
        tools_menu.setEnabled(True)

    def _build_ui(self) -> None:
        """
        Build the central MATLAB-like layout.
        """
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)

        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        self.plot_canvas = MatlabStylePlotCanvas()
        self.plot_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        right_panel = self._build_right_panel()

        splitter.addWidget(self.plot_canvas)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([820, 580])

    def _build_right_panel(self) -> QWidget:
        """
        Build the right panel with result lists, file list, buttons, and plot options.
        """
        panel = QWidget()
        panel.setMinimumWidth(570)

        layout = QHBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)

        self.df2tune_list = QListWidget()
        self.ds11_list = QListWidget()
        self.file_list = QListWidget()

        self.file_list.currentRowChanged.connect(self._set_current_record_from_row)

        layout.addWidget(
            self._make_list_column(
                title="df to tune\n(MHz)",
                list_widget=self.df2tune_list,
                width=105,
            )
        )

        layout.addWidget(
            self._make_list_column(
                title="sgn*|ds11|\n(mU)",
                list_widget=self.ds11_list,
                width=115,
            )
        )

        file_and_options = QWidget()
        file_and_options_layout = QVBoxLayout(file_and_options)
        file_and_options_layout.setContentsMargins(0, 0, 0, 0)

        screenshot_row = QHBoxLayout()
        screenshot_row.addStretch()

        self.screenshot_button = QPushButton("ScreenShot")
        self.screenshot_button.clicked.connect(self.save_screenshot)
        screenshot_row.addWidget(self.screenshot_button)

        file_and_options_layout.addLayout(screenshot_row)
        file_and_options_layout.addWidget(self.file_list)

        button_row = QHBoxLayout()

        self.add_button = QPushButton("Add")
        self.delete_button = QPushButton("Delete")
        self.export_button = QPushButton("Export")
        self.debug_button = QPushButton("Debug")
        self.legend_button = QPushButton("Leg")
        self.colorbar_button = QPushButton("ColBar")

        self.add_button.clicked.connect(self.add_beadpull_file)
        self.delete_button.clicked.connect(self.delete_selected_record)
        self.export_button.clicked.connect(self.export_current_tuning_table)
        self.debug_button.clicked.connect(self.show_debug_summary)
        self.legend_button.clicked.connect(self.toggle_legend)
        self.colorbar_button.clicked.connect(self.toggle_colorbar_placeholder)

        for button in [
            self.add_button,
            self.delete_button,
            self.export_button,
            self.debug_button,
            self.legend_button,
            self.colorbar_button,
        ]:
            button_row.addWidget(button)

        file_and_options_layout.addLayout(button_row)
        file_and_options_layout.addWidget(self._build_plot_options_group())

        layout.addWidget(file_and_options)

        return panel

    def _make_list_column(
        self,
        title: str,
        list_widget: QListWidget,
        width: int,
    ) -> QWidget:
        """
        Build a narrow MATLAB-like list column with a title.
        """
        widget = QWidget()
        widget.setFixedWidth(width)

        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        label = QLabel(title)
        label.setAlignment(Qt.AlignCenter)

        list_widget.setMinimumHeight(620)

        layout.addWidget(label)
        layout.addWidget(list_widget)

        return widget

    def _build_plot_options_group(self) -> QGroupBox:
        """
        Build the MATLAB-like radio-button plot selector.
        """
        group = QGroupBox("Plot...")
        layout = QGridLayout(group)

        self.plot_button_group = QButtonGroup(self)
        self.plot_button_group.setExclusive(True)

        options = [
            ("df to tune", "df_to_tune"),
            ("phase advance", "phase_advance"),
            ("S11 bead-pull", "s11_beadpull"),
            ("dS11 BP", "ds11_bp"),
            ("|dS11| BP", "abs_ds11_bp"),
            ("|dS11| BP (z)", "abs_ds11_bp_z"),
            ("Mag(E)", "mag_e"),
            ("Mag(peaks(E))", "mag_peaks_e"),
            ("0-Line", "zero_line"),
            ("+/-|dS11|", "pm_abs_ds11"),
            ("phi v.s. freq", "phi_vs_freq"),
            ("local S11", "local_s11"),
            ("local S11(cell)", "local_s11_cell"),
            ("wbn", "wbn"),
            ("wfn", "wfn"),
            ("arg(dS11) BP (z)", "arg_ds11_bp_z"),
            ("|arg(dS11) BP(z)|", "abs_arg_ds11_bp_z"),
        ]

        for idx, (text, key) in enumerate(options):
            button = QRadioButton(text)
            button.setProperty("plot_key", key)
            button.toggled.connect(self._plot_radio_changed)
            self.plot_button_group.addButton(button)

            row = idx % 9
            col = idx // 9

            layout.addWidget(button, row, col)

            if key == "df_to_tune":
                button.setChecked(True)

        return group

    def open_structure_dialog(self) -> None:
        """
        Open the RF structure loader dialog.
        """
        dialog = RFStructureLoaderDialog(self)

        if not dialog.exec():
            return

        if dialog.rf_structure is None:
            QMessageBox.warning(self, "RF structure", "No RF structure was loaded.")
            return

        self.RF_params = dialog.rf_structure
        self.statusBar().showMessage("RF structure loaded")

    def open_measurement_dialog(self) -> None:
        """
        Open the measurement condition dialog.
        """
        dialog = MeasurementConfigDialog(self.Meas_params, self)

        if dialog.exec():
            self.Meas_params = dialog.get_config()
            self.statusBar().showMessage("Measurement configuration updated")

    def add_beadpull_file(self) -> None:
        """
        Add one or more bead-pull files and analyze them immediately.
        """
        if self.RF_params is None:
            QMessageBox.warning(
                self,
                "Missing RF structure",
                "Load Structure RF Design before loading bead-pull files.",
            )
            return

        filenames, _ = QFileDialog.getOpenFileNames(
            self,
            "Load bead-pull files",
            "",
            "CSV files (*.csv);;All files (*)",
        )

        if not filenames:
            return

        for filename in filenames:
            self._analyze_and_add_file(filename)

        if self.records:
            self.file_list.setCurrentRow(len(self.records) - 1)

        self.update_all_views()
        self.statusBar().showMessage("Bead-pull file analysis completed")

    def _analyze_and_add_file(self, filename: str) -> None:
        """
        Analyze one bead-pull file and append it to the record list.
        """
        bdata = BeadpullRecord(
            RF_params=self.RF_params,
            Meas_params=self.Meas_params,
            BP_options=self.BP_options,
            filename=filename,
        )

        self.analyzer.evaluate(bdata)

        self.records.append(bdata)

        item_text = f"{len(self.records):02d}: \"{filename}\""
        item = QListWidgetItem(item_text)
        self.file_list.addItem(item)

    def delete_selected_record(self) -> None:
        """
        Delete the selected bead-pull record.
        """
        row = self.file_list.currentRow()

        if row < 0 or row >= len(self.records):
            return

        self.records.pop(row)
        self.file_list.takeItem(row)

        for idx in range(self.file_list.count()):
            record = self.records[idx]
            self.file_list.item(idx).setText(
                f"{idx + 1:02d}: \"{record.filename}\""
            )

        if self.records:
            self.file_list.setCurrentRow(min(row, len(self.records) - 1))
        else:
            self.current_record_index = None

        self.update_all_views()

    def export_current_tuning_table(self) -> None:
        """
        Export the tuning table of the selected record.
        """
        bdata = self.current_record

        if bdata is None:
            QMessageBox.warning(self, "Export", "No bead-pull record is selected.")
            return

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export tuning table",
            "",
            "CSV files (*.csv);;All files (*)",
        )

        if not filename:
            return

        bdata.tuning_dataframe().to_csv(filename, index=False)
        self.statusBar().showMessage(f"Exported {filename}")

    def show_debug_summary(self) -> None:
        """
        Show a compact debug summary for the selected record.
        """
        bdata = self.current_record

        if bdata is None:
            QMessageBox.information(self, "Debug", "No bead-pull record is selected.")
            return

        summary = bdata.summary()
        text = "\n".join(f"{key}: {value}" for key, value in summary.items())

        QMessageBox.information(self, "Debug summary", text)

    def toggle_legend(self) -> None:
        """
        Toggle plot legend.
        """
        self.legend_visible = not self.legend_visible
        self.plot_canvas.set_legend_visible(self.legend_visible)
        self.update_plot()

    def toggle_colorbar_placeholder(self) -> None:
        """
        Placeholder for MATLAB ColBar button.
        """
        QMessageBox.information(
            self,
            "ColBar",
            "Colorbar support can be added for 2D plots later.",
        )

    def save_screenshot(self) -> None:
        """
        Save a screenshot of the main window.
        """
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save screenshot",
            "",
            "PNG files (*.png);;All files (*)",
        )

        if not filename:
            return

        self.grab().save(filename)
        self.statusBar().showMessage(f"Screenshot saved to {filename}")

    def _set_current_record_from_row(self, row: int) -> None:
        """
        Set the selected record from the file list.
        """
        if row < 0 or row >= len(self.records):
            self.current_record_index = None
        else:
            self.current_record_index = row

        self.update_result_lists()

    @property
    def current_record(self) -> BeadpullRecord | None:
        """
        Return the currently selected bead-pull record.
        """
        if self.current_record_index is None:
            return None

        if self.current_record_index < 0 or self.current_record_index >= len(self.records):
            return None

        return self.records[self.current_record_index]

    def update_all_views(self) -> None:
        """
        Update plot and right-side lists.
        """
        self.update_result_lists()
        self.update_plot()

    def update_result_lists(self) -> None:
        """
        Update the df2tune and ds11 numerical lists for the selected record.
        """
        self.df2tune_list.clear()
        self.ds11_list.clear()

        bdata = self.current_record

        if bdata is None:
            return

        if bdata.df2tune is not None:
            for idx, value in enumerate(bdata.df2tune, start=1):
                self.df2tune_list.addItem(f"{idx:02d}| {value / 1e6:.2f}")

        if bdata.ds11 is not None:
            for idx, value in enumerate(bdata.ds11, start=1):
                self.ds11_list.addItem(f"{idx:02d}| {value * 1e3:.2f}")

    def update_plot(self) -> None:
        """
        Update the main plot using all loaded records.
        """
        self.plot_canvas.plot_records(self.records, self.current_plot_key)

    def _plot_radio_changed(self, checked: bool) -> None:
        """
        Update plot type when a radio button is selected.
        """
        if not checked:
            return

        button = self.sender()

        if button is None:
            return

        self.current_plot_key = button.property("plot_key")
        self.update_plot()

    def show_about(self) -> None:
        """
        Show application information.
        """
        QMessageBox.information(
            self,
            "About",
            "CLIC Bead-pull Offline Analysis\n\nPython implementation.",
        )