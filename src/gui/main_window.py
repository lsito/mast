from PySide6.QtWidgets import QMainWindow, QPushButton, QWidget, QVBoxLayout

from src.gui.meas_config_dialog import MeasurementConfigDialog
from src.data_models.meas_config import MeasurementConfig


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Main Window")
        self.resize(800, 600)

        self.measurement_config = MeasurementConfig()

        central = QWidget()
        self.setCentralWidget(central)

        layout = QVBoxLayout(central)

        self.open_dialog_button = QPushButton("Open measurement conditions")
        self.open_dialog_button.clicked.connect(self.open_measurement_dialog)
        layout.addWidget(self.open_dialog_button)

    def open_measurement_dialog(self):
        dialog = MeasurementConfigDialog(self.measurement_config, self)
        if dialog.exec():
            self.measurement_config = dialog.get_config()
            print("Updated config:", self.measurement_config)
