import numpy as np

from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QFileDialog,
    QLineEdit,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QMessageBox,
)

from src.data_models.rf_structure import RFStructureParams

class RFStructureLoaderDialog(QDialog):

    def __init__(self, parent=None):
        super().__init__(parent)

        self.rf_structure = None

        self.setWindowTitle("Load RF Structure")
        self.resize(900, 600)

        self.build_ui()

    def build_ui(self):

        main_layout = QVBoxLayout(self)

        # -------------------------
        # file selection row
        # -------------------------

        file_layout = QHBoxLayout()

        file_layout.addWidget(QLabel("File"))

        self.file_edit = QLineEdit()

        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self.browse_file)

        load_button = QPushButton("Load")
        load_button.clicked.connect(self.load_file)

        file_layout.addWidget(self.file_edit)
        file_layout.addWidget(browse_button)
        file_layout.addWidget(load_button)

        main_layout.addLayout(file_layout)

        # -------------------------
        # parameter table
        # -------------------------

        self.table = QTableWidget()
        self.table.setColumnCount(2)

        self.table.setHorizontalHeaderLabels([
            "Parameter",
            "Value",
        ])

        main_layout.addWidget(self.table)

        # -------------------------
        # bottom buttons
        # -------------------------

        button_layout = QHBoxLayout()

        ok_button = QPushButton("OK")
        ok_button.clicked.connect(self.accept)

        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)

        button_layout.addStretch()

        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)

        main_layout.addLayout(button_layout)

    def browse_file(self):

        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Select RF Structure File",
            "",
            "JSON Files (*.json)"
        )

        if filename:
            self.file_edit.setText(filename)

    def load_file(self):

        filename = self.file_edit.text()

        if not filename:
            QMessageBox.warning(
                self,
                "Missing file",
                "Please select a file."
            )
            return

        try:

            self.rf_structure = RFStructureParams.from_json(filename)

            self.populate_table()

        except Exception as exc:

            QMessageBox.critical(
                self,
                "Load error",
                str(exc)
            )

    def populate_table(self):

        params = self.rf_structure

        fields = params.__dataclass_fields__

        self.table.setRowCount(len(fields))

        for row, field_name in enumerate(fields):

            value = getattr(params, field_name)

            self.table.setItem(
                row,
                0,
                QTableWidgetItem(field_name)
            )

            self.table.setItem(
                row,
                1,
                QTableWidgetItem(self.format_value(value))
            )

        self.table.resizeColumnsToContents()

    @staticmethod
    def format_value(value):

        if isinstance(value, np.ndarray):

            if value.size > 10:

                preview = np.array2string(
                    value[:10],
                    precision=4,
                    separator=", "
                )

                return f"{preview} ... shape={value.shape}"

            return np.array2string(
                value,
                precision=4,
                separator=", "
            )

        return str(value)