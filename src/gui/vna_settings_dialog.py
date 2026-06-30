from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)


class VNASettingsDialog(QDialog):
    """
    Dialog for setting the VNA IP address and port.
    """

    def __init__(
        self,
        ip_address: str = "128.11.11.11",
        port: int = 1601,
        parent=None,
    ) -> None:
        """
        Initialize the VNA settings dialog.
        """
        super().__init__(parent)

        self.setWindowTitle("VNA Settings")
        self.setModal(True)
        self.resize(360, 150)

        self.ip_address = str(ip_address)
        self.port = int(port)

        self._build_ui()
        self._load_values()

    def _build_ui(self) -> None:
        """
        Build the dialog UI.
        """
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(12)

        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)

        self.ip_edit = QLineEdit()
        self.port_edit = QLineEdit()

        grid.addWidget(QLabel("IP address"), 0, 0)
        grid.addWidget(self.ip_edit, 0, 1)

        grid.addWidget(QLabel("Port"), 1, 0)
        grid.addWidget(self.port_edit, 1, 1)

        button_row = QHBoxLayout()

        self.ok_button = QPushButton("OK")
        self.cancel_button = QPushButton("Cancel")

        self.ok_button.clicked.connect(self.on_ok)
        self.cancel_button.clicked.connect(self.reject)

        button_row.addStretch()
        button_row.addWidget(self.ok_button)
        button_row.addWidget(self.cancel_button)

        main_layout.addLayout(grid)
        main_layout.addStretch()
        main_layout.addLayout(button_row)

    def _load_values(self) -> None:
        """
        Load current values into the fields.
        """
        self.ip_edit.setText(self.ip_address)
        self.port_edit.setText(str(self.port))

    def on_ok(self) -> None:
        """
        Validate and accept settings.
        """
        ip_address = self.ip_edit.text().strip()
        port_text = self.port_edit.text().strip()

        if not ip_address:
            QMessageBox.warning(self, "VNA Settings", "Enter a VNA IP address.")
            return

        try:
            port = int(port_text)
        except ValueError:
            QMessageBox.warning(self, "VNA Settings", "Enter a valid integer port.")
            return

        if port <= 0:
            QMessageBox.warning(self, "VNA Settings", "Port must be positive.")
            return

        self.ip_address = ip_address
        self.port = port

        self.accept()