from PySide6.QtWidgets import QMainWindow, QPushButton, QWidget, QVBoxLayout

from src.gui.meas_config_dialog import MeasurementConfigDialog
from src.data_models.meas_config import MeasurementConfig

from src.gui.structure_config_dialog import RFStructureLoaderDialog
from src.data_models.rf_structure import RFStructureParams

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Main Window")
        self.resize(800, 600)

        # -------------------------
        # stored application data
        # -------------------------

        self.measurement_config = MeasurementConfig()
        self.rf_structure = RFStructureParams()

        # -------------------------
        # central widget
        # -------------------------

        central = QWidget()
        self.setCentralWidget(central)

        layout = QVBoxLayout(central)

        # -------------------------
        # measurement config button
        # -------------------------

        self.open_meas_dialog_button = QPushButton(
            "Open measurement conditions"
        )

        self.open_meas_dialog_button.clicked.connect(
            self.open_measurement_dialog
        )

        layout.addWidget(self.open_meas_dialog_button)

        # -------------------------
        # RF structure button
        # -------------------------

        self.open_rf_dialog_button = QPushButton(
            "Load RF structure"
        )

        self.open_rf_dialog_button.clicked.connect(
            self.open_rf_structure_dialog
        )

        layout.addWidget(self.open_rf_dialog_button)

        layout.addStretch()

    # ==========================================================
    # measurement config
    # ==========================================================

    def open_measurement_dialog(self):

        dialog = MeasurementConfigDialog(
            self.measurement_config,
            self
        )

        if dialog.exec():

            self.measurement_config = dialog.get_config()

            print(
                "Updated measurement config:",
                self.measurement_config
            )

    # ==========================================================
    # RF structure
    # ==========================================================

    def open_rf_structure_dialog(self):

        dialog = RFStructureLoaderDialog(self)

        if dialog.exec():

            self.rf_structure = dialog.rf_structure

            print(
                "Loaded RF structure:",
                self.rf_structure
            )