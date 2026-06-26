from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QFileDialog,
    QAbstractItemView,
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
from src.gui.beadpull_file_dialog import BeadpullFileDialog
from src.gui.meas_config_dialog import MeasurementConfigDialog
from src.gui.plot_canvas import MatlabStylePlotCanvas
from src.gui.structure_config_dialog import RFStructureLoaderDialog
from src.gui.terminal_widget import TerminalWidget


class MainWindow(QMainWindow):
    """
    MATLAB-like main window for CLIC bead-pull offline analysis.

    The GUI supports selecting any combination of loaded bead-pull files for
    plotting. The selected row controls the numerical lists. The checked rows
    control which files are plotted.

    Bead-pull files are added through `BeadpullFileDialog`, so each file can
    have its own frequency, temperature, measurement direction, and bead-pull
    analysis options.
    """

    def __init__(self) -> None:
        """
        Initialize the main window.
        """
        super().__init__()

        self.setWindowTitle("CLIC Bead-pull Offline Analysis")
        self.resize(1650, 1000)

        self.RF_params: RFStructureParams | None = None
        self.Meas_params = MeasurementConfig()
        self.BP_options = BeadpullConfig()

        self.analyzer = BeadPullAnalyzer()
        self.records: list[BeadpullRecord] = []
        self.current_record_index: int | None = None

        self.legend_visible = False
        self.current_plot_key = "df_to_tune"

        self.structure_done = False
        self.measurement_done = False
        self.beadpull_done = False

        self._build_actions()
        self._build_menu()
        self._build_ui()
        self._apply_style()
        self._update_step_buttons()

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

        self.action_export_current = QAction("Export Current Tuning Table...", self)
        self.action_export_current.triggered.connect(self.export_current_tuning_table)

        self.action_delete_current = QAction("Delete Current Bead-pull", self)
        self.action_delete_current.triggered.connect(self.delete_selected_record)

        self.action_clear_all = QAction("Clear All", self)
        self.action_clear_all.triggered.connect(self.clear_all_records)

        self.action_screenshot = QAction("ScreenShot...", self)
        self.action_screenshot.triggered.connect(self.save_screenshot)

        self.action_toggle_legend = QAction("Legend On/Off", self)
        self.action_toggle_legend.triggered.connect(self.toggle_legend)

        self.action_debug_summary = QAction("Debug Summary", self)
        self.action_debug_summary.triggered.connect(self.show_debug_summary)

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
        file_menu.addAction(self.action_export_current)
        file_menu.addAction(self.action_screenshot)
        file_menu.addSeparator()
        file_menu.addAction(self.action_exit)

        edit_menu = menu_bar.addMenu("Edit")
        edit_menu.addAction(self.action_delete_current)
        edit_menu.addAction(self.action_clear_all)

        tools_menu = menu_bar.addMenu("Tools")
        tools_menu.addAction(self.action_toggle_legend)
        tools_menu.addAction(self.action_debug_summary)

        help_menu = menu_bar.addMenu("Help")
        help_menu.addAction(self.action_about)

    def _build_ui(self) -> None:
        """
        Build the central layout.
        """
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(10)

        main_layout.addWidget(self._build_top_step_bar())

        vertical_splitter = QSplitter(Qt.Vertical)
        main_layout.addWidget(vertical_splitter)

        top_splitter = QSplitter(Qt.Horizontal)

        self.plot_canvas = MatlabStylePlotCanvas()
        self.plot_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        right_panel = self._build_right_panel()

        top_splitter.addWidget(self.plot_canvas)
        top_splitter.addWidget(right_panel)
        top_splitter.setStretchFactor(0, 1)
        top_splitter.setStretchFactor(1, 0)
        top_splitter.setSizes([980, 650])

        self.terminal = TerminalWidget()
        self._update_terminal_namespace()

        vertical_splitter.addWidget(top_splitter)
        vertical_splitter.addWidget(self.terminal)
        vertical_splitter.setStretchFactor(0, 1)
        vertical_splitter.setStretchFactor(1, 0)
        vertical_splitter.setSizes([760, 190])

    def _build_top_step_bar(self) -> QWidget:
        """
        Build top workflow buttons.
        """
        bar = QWidget()
        bar.setObjectName("TopStepBar")

        layout = QHBoxLayout(bar)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(12)

        self.structure_step_button = QPushButton("1  Structure RF Design")
        self.measurement_step_button = QPushButton("2  Measurement Conditions")
        self.beadpull_step_button = QPushButton("3  Bead-pull Files")

        self.structure_step_button.clicked.connect(self.open_structure_dialog)
        self.measurement_step_button.clicked.connect(self.open_measurement_dialog)
        self.beadpull_step_button.clicked.connect(self.add_beadpull_file)

        for button in [
            self.structure_step_button,
            self.measurement_step_button,
            self.beadpull_step_button,
        ]:
            button.setObjectName("StepButton")
            button.setMinimumHeight(38)
            layout.addWidget(button)

        return bar

    def _build_right_panel(self) -> QWidget:
        """
        Build the right panel.

        The top row contains screenshot and file-selection controls.
        The middle contains the file list.
        The bottom contains action buttons in two rows.
        """
        panel = QWidget()
        panel.setObjectName("RightPanel")
        panel.setMinimumWidth(660)

        layout = QHBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        self.df2tune_list = QListWidget()
        self.ds11_list = QListWidget()
        self.file_list = QListWidget()

        self.file_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.file_list.currentRowChanged.connect(self._set_current_record_from_row)
        self.file_list.itemChanged.connect(self._file_item_changed)

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
        file_and_options_layout.setSpacing(8)

        top_button_row = QHBoxLayout()
        top_button_row.setSpacing(8)

        self.screenshot_button = QPushButton("ScreenShot")
        self.check_all_button = QPushButton("All")
        self.check_none_button = QPushButton("None")
        self.check_current_button = QPushButton("Current")

        self.screenshot_button.clicked.connect(self.save_screenshot)
        self.check_all_button.clicked.connect(self.check_all_records)
        self.check_none_button.clicked.connect(self.check_no_records)
        self.check_current_button.clicked.connect(self.check_current_record_only)

        for button in [
            self.screenshot_button,
            self.check_all_button,
            self.check_none_button,
            self.check_current_button,
        ]:
            button.setMinimumWidth(86)
            button.setMinimumHeight(32)

        top_button_row.addWidget(self.screenshot_button)
        top_button_row.addStretch()
        top_button_row.addWidget(self.check_all_button)
        top_button_row.addWidget(self.check_none_button)
        top_button_row.addWidget(self.check_current_button)

        file_and_options_layout.addLayout(top_button_row)

        self.file_list_label = QLabel("Files to plot")
        self.file_list_label.setObjectName("PanelTitle")

        file_and_options_layout.addWidget(self.file_list_label)
        file_and_options_layout.addWidget(self.file_list)

        bottom_button_grid = QGridLayout()
        bottom_button_grid.setHorizontalSpacing(8)
        bottom_button_grid.setVerticalSpacing(8)

        self.add_button = QPushButton("Add")
        self.delete_button = QPushButton("Delete")
        self.export_button = QPushButton("Export")
        self.debug_button = QPushButton("Debug")
        self.legend_button = QPushButton("Legend")
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
            button.setMinimumWidth(90)
            button.setMinimumHeight(32)

        bottom_button_grid.addWidget(self.add_button, 0, 0)
        bottom_button_grid.addWidget(self.delete_button, 0, 1)
        bottom_button_grid.addWidget(self.export_button, 0, 2)

        bottom_button_grid.addWidget(self.debug_button, 1, 0)
        bottom_button_grid.addWidget(self.legend_button, 1, 1)
        bottom_button_grid.addWidget(self.colorbar_button, 1, 2)

        file_and_options_layout.addLayout(bottom_button_grid)
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
        Build a narrow list column.
        """
        widget = QWidget()
        widget.setFixedWidth(width)

        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        label = QLabel(title)
        label.setAlignment(Qt.AlignCenter)
        label.setObjectName("PanelTitle")

        list_widget.setMinimumHeight(620)

        layout.addWidget(label)
        layout.addWidget(list_widget)

        return widget

    def _build_plot_options_group(self) -> QGroupBox:
        """
        Build the radio-button plot selector.
        """
        group = QGroupBox("Plot...")
        layout = QGridLayout(group)
        layout.setHorizontalSpacing(10)
        layout.setVerticalSpacing(4)

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

    def _apply_style(self) -> None:
        """
        Apply a modern light stylesheet.
        """
        self.setStyleSheet(
            """
            QMainWindow {
                background-color: #f5f7fb;
            }

            QWidget {
                font-family: "Segoe UI", "Arial";
                font-size: 10pt;
                color: #1f2937;
            }

            QMenuBar {
                background-color: #ffffff;
                border-bottom: 1px solid #e5e7eb;
                padding: 4px;
            }

            QMenuBar::item {
                background: transparent;
                padding: 6px 10px;
                border-radius: 6px;
            }

            QMenuBar::item:selected {
                background-color: #eef2ff;
            }

            QMenu {
                background-color: #ffffff;
                border: 1px solid #d1d5db;
                padding: 6px;
            }

            QMenu::item {
                padding: 7px 24px;
                border-radius: 5px;
            }

            QMenu::item:selected {
                background-color: #eef2ff;
            }

            QWidget#TopStepBar,
            QWidget#RightPanel {
                background-color: #ffffff;
                border: 1px solid #e5e7eb;
                border-radius: 12px;
            }

            QLabel#PanelTitle {
                font-weight: 600;
                color: #374151;
            }

            QLabel#CoordinateLabel {
                color: #4b5563;
                padding: 4px 8px;
                background-color: #ffffff;
                border-top: 1px solid #e5e7eb;
            }

            QPushButton {
                background-color: #2563eb;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 7px 10px;
                font-weight: 600;
            }

            QPushButton:hover {
                background-color: #1d4ed8;
            }

            QPushButton:pressed {
                background-color: #1e40af;
            }

            QPushButton#StepButton {
                background-color: #e5e7eb;
                color: #374151;
                border: 1px solid #d1d5db;
                font-weight: 700;
            }

            QPushButton#StepButton[state="done"] {
                background-color: #16a34a;
                color: white;
                border: 1px solid #15803d;
            }

            QPushButton#StepButton[state="pending"] {
                background-color: #e5e7eb;
                color: #374151;
                border: 1px solid #d1d5db;
            }

            QListWidget,
            QPlainTextEdit,
            QLineEdit {
                background-color: #ffffff;
                border: 1px solid #d1d5db;
                border-radius: 8px;
                padding: 5px;
                selection-background-color: #dbeafe;
                selection-color: #111827;
            }

            QListWidget::item {
                padding: 4px;
                border-radius: 5px;
            }

            QListWidget::item:selected {
                background-color: #dbeafe;
                color: #111827;
            }

            QGroupBox {
                background-color: #ffffff;
                border: 1px solid #e5e7eb;
                border-radius: 10px;
                margin-top: 12px;
                padding: 10px;
                font-weight: 700;
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
                color: #374151;
            }

            QRadioButton {
                spacing: 6px;
            }

            QStatusBar {
                background-color: #ffffff;
                border-top: 1px solid #e5e7eb;
            }

            QSplitter::handle {
                background-color: #e5e7eb;
            }

            QSplitter::handle:hover {
                background-color: #c7d2fe;
            }
            """
        )

    def _update_step_buttons(self) -> None:
        """
        Update top workflow button states.
        """
        states = {
            self.structure_step_button: self.structure_done,
            self.measurement_step_button: self.measurement_done,
            self.beadpull_step_button: self.beadpull_done,
        }

        for button, done in states.items():
            button.setProperty("state", "done" if done else "pending")
            button.style().unpolish(button)
            button.style().polish(button)
            button.update()

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
        self.structure_done = True

        self._update_step_buttons()
        self._update_terminal_namespace()
        self.statusBar().showMessage("RF structure loaded")
        self.terminal.write("RF structure loaded.")

    def open_measurement_dialog(self) -> None:
        """
        Open the measurement condition dialog.
        """
        dialog = MeasurementConfigDialog(self.Meas_params, self)

        if dialog.exec():
            self.Meas_params = dialog.get_config()
            self.measurement_done = True

            for record in self.records:
                record.Meas_params = self.Meas_params

            self._update_step_buttons()
            self._update_terminal_namespace()
            self.statusBar().showMessage("Measurement configuration updated")
            self.terminal.write("Measurement configuration updated.")

    def add_beadpull_file(self) -> None:
        """
        Add one or more bead-pull files and analyze them immediately.

        The file dialog allows multiple selection using Ctrl/Shift selection.
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

        added_count = 0
        failed_files = []

        for filename in filenames:
            try:
                self._analyze_and_add_file(filename)
                added_count += 1
            except Exception as exc:
                failed_files.append((filename, exc))

                if hasattr(self, "terminal"):
                    self.terminal.write(f"Failed to load {Path(filename).name}: {exc}")

        if self.records:
            self.file_list.setCurrentRow(len(self.records) - 1)

        if hasattr(self, "beadpull_done"):
            self.beadpull_done = len(self.records) > 0

        if hasattr(self, "_update_step_buttons"):
            self._update_step_buttons()

        self.update_all_views()
        self._update_terminal_namespace()

        if failed_files:
            failed_text = "\n".join(
                f"{Path(filename).name}: {exc}" for filename, exc in failed_files
            )

            QMessageBox.warning(
                self,
                "Some bead-pull files failed",
                f"{added_count} file(s) loaded successfully.\n\nFailed files:\n{failed_text}",
            )
        else:
            self.statusBar().showMessage(
                f"{added_count} bead-pull file(s) analyzed successfully"
            )

            if hasattr(self, "terminal"):
                self.terminal.write(
                    f"{added_count} bead-pull file(s) loaded and analyzed."
                )


    def _analyze_and_add_file(self, filename: str) -> None:
        """
        Analyze one bead-pull file and append it to the record list.

        The full path is stored in `BeadpullRecord.filename`.
        The GUI list only displays the filename.
        """
        if self.RF_params is None:
            raise RuntimeError("RF structure must be loaded before analysis.")

        bdata = BeadpullRecord(
            RF_params=self.RF_params,
            Meas_params=self.Meas_params,
            BP_options=self.BP_options,
            filename=filename,
        )

        self.analyzer.evaluate(bdata)

        self.records.append(bdata)

        display_name = Path(filename).name

        item = QListWidgetItem(display_name)
        item.setToolTip(filename)
        item.setData(Qt.UserRole, len(self.records) - 1)
        item.setFlags(
            item.flags()
            | Qt.ItemIsUserCheckable
            | Qt.ItemIsSelectable
            | Qt.ItemIsEnabled
        )
        item.setCheckState(Qt.Checked)

        self.file_list.addItem(item)

        if hasattr(self, "terminal"):
            self.terminal.write(f"Loaded and analyzed: {display_name}")

    def delete_selected_record(self) -> None:
        """
        Delete the selected bead-pull record.
        """
        row = self.file_list.currentRow()

        if row < 0 or row >= len(self.records):
            return

        deleted = self.records[row]

        self.records.pop(row)
        self.file_list.takeItem(row)
        self._refresh_file_items()

        if self.records:
            self.file_list.setCurrentRow(min(row, len(self.records) - 1))
        else:
            self.current_record_index = None

        self.beadpull_done = len(self.records) > 0

        self._update_step_buttons()
        self.update_all_views()
        self._update_terminal_namespace()

        if deleted.filename is not None:
            self.terminal.write(f"Deleted: {Path(deleted.filename).name}")

    def clear_all_records(self) -> None:
        """
        Clear all loaded bead-pull records.
        """
        self.records.clear()
        self.current_record_index = None
        self.beadpull_done = False

        self.file_list.clear()
        self.df2tune_list.clear()
        self.ds11_list.clear()
        self.plot_canvas.clear_plot()

        self._update_step_buttons()
        self._update_terminal_namespace()
        self.terminal.write("Cleared all bead-pull records.")
        self.statusBar().showMessage("All records cleared")

    def _refresh_file_items(self) -> None:
        """
        Refresh file-list item labels and stored indices.
        """
        self.file_list.blockSignals(True)

        checked_filenames = set()

        for idx in range(self.file_list.count()):
            item = self.file_list.item(idx)

            if item.checkState() == Qt.Checked:
                checked_filenames.add(item.text())

        self.file_list.clear()

        for idx, record in enumerate(self.records):
            filename = Path(record.filename).name if record.filename else f"record {idx + 1}"
            item = QListWidgetItem(filename)
            item.setData(Qt.UserRole, idx)
            item.setFlags(
                item.flags()
                | Qt.ItemIsUserCheckable
                | Qt.ItemIsSelectable
                | Qt.ItemIsEnabled
            )

            if filename in checked_filenames:
                item.setCheckState(Qt.Checked)
            else:
                item.setCheckState(Qt.Unchecked)

            self.file_list.addItem(item)

        self.file_list.blockSignals(False)

    def check_all_records(self) -> None:
        """
        Check all loaded records for plotting.
        """
        self.file_list.blockSignals(True)

        for idx in range(self.file_list.count()):
            self.file_list.item(idx).setCheckState(Qt.Checked)

        self.file_list.blockSignals(False)
        self.update_plot()

    def check_no_records(self) -> None:
        """
        Uncheck all loaded records.
        """
        self.file_list.blockSignals(True)

        for idx in range(self.file_list.count()):
            self.file_list.item(idx).setCheckState(Qt.Unchecked)

        self.file_list.blockSignals(False)
        self.update_plot()

    def check_current_record_only(self) -> None:
        """
        Check only the currently selected record.
        """
        current_row = self.file_list.currentRow()

        self.file_list.blockSignals(True)

        for idx in range(self.file_list.count()):
            state = Qt.Checked if idx == current_row else Qt.Unchecked
            self.file_list.item(idx).setCheckState(state)

        self.file_list.blockSignals(False)
        self.update_plot()

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
        self.terminal.write(f"Exported tuning table: {filename}")

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
        self.terminal.write(text)

    def toggle_legend(self) -> None:
        """
        Toggle plot legend.
        """
        self.legend_visible = not self.legend_visible
        self.plot_canvas.set_legend_visible(self.legend_visible)
        self.update_plot()

        if self.legend_visible:
            self.terminal.write("Legend enabled.")
        else:
            self.terminal.write("Legend disabled.")

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
        self.terminal.write(f"Screenshot saved: {filename}")

    def _set_current_record_from_row(self, row: int) -> None:
        """
        Set the selected record from the file list.
        """
        if row < 0 or row >= len(self.records):
            self.current_record_index = None
        else:
            self.current_record_index = row

        self.update_result_lists()
        self._update_terminal_namespace()

    def _file_item_changed(self, item: QListWidgetItem) -> None:
        """
        Update the plot when a file checkbox changes.
        """
        self.update_plot()

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

    @property
    def plotted_records(self) -> list[BeadpullRecord]:
        """
        Return records whose file-list items are checked.
        """
        selected_records = []

        for row in range(self.file_list.count()):
            item = self.file_list.item(row)

            if item.checkState() == Qt.Checked:
                idx = item.data(Qt.UserRole)

                if idx is not None and 0 <= idx < len(self.records):
                    selected_records.append(self.records[idx])

        return selected_records

    def update_all_views(self) -> None:
        """
        Update plot and numerical lists.
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
        Update the main plot using checked records.
        """
        self.plot_canvas.plot_records(self.plotted_records, self.current_plot_key)

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

    def _update_terminal_namespace(self) -> None:
        """
        Update the embedded terminal namespace with the current GUI state.
        """
        if not hasattr(self, "terminal"):
            return

        self.terminal.update_namespace(
            window=self,
            records=self.records,
            bdata=self.current_record,
            plotted_records=self.plotted_records,
            RF_params=self.RF_params,
            Meas_params=self.Meas_params,
            BP_options=self.BP_options,
            analyzer=self.analyzer,
            np=np,
        )

    def show_about(self) -> None:
        """
        Show application information.
        """
        QMessageBox.information(
            self,
            "About",
            "CLIC Bead-pull Offline Analysis\n\nPython implementation.",
        )