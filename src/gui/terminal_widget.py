from __future__ import annotations

import code
import contextlib
import io
from typing import Any

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class TerminalWidget(QWidget):
    """
    Small embedded Python terminal for the GUI.

    The terminal evaluates Python commands in a shared namespace. It is useful
    for inspecting `bdata`, `records`, `RF_params`, and other objects while the
    GUI is running.

    This is not a system shell. It is an embedded Python console.
    """

    command_executed = Signal(str)

    def __init__(self, namespace: dict[str, Any] | None = None, parent=None) -> None:
        """
        Initialize the terminal widget.
        """
        super().__init__(parent)

        self.namespace = namespace or {}
        self.console = code.InteractiveConsole(self.namespace)
        self.prompt = ">>> "
        self.more_input_required = False

        self._build_ui()
        self.write("Embedded Python terminal ready.")
        self.write("Available objects will be updated by the main window.")

    def _build_ui(self) -> None:
        """
        Build the terminal layout.
        """
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        header_row = QHBoxLayout()

        self.title_label = QLabel("Terminal")
        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self.clear)

        header_row.addWidget(self.title_label)
        header_row.addStretch()
        header_row.addWidget(self.clear_button)

        self.output = QPlainTextEdit()
        self.output.setReadOnly(True)
        self.output.setMinimumHeight(120)

        input_row = QHBoxLayout()

        self.prompt_label = QLabel(self.prompt)
        self.input_line = QLineEdit()
        self.input_line.returnPressed.connect(self.execute_current_line)

        self.run_button = QPushButton("Run")
        self.run_button.clicked.connect(self.execute_current_line)

        input_row.addWidget(self.prompt_label)
        input_row.addWidget(self.input_line)
        input_row.addWidget(self.run_button)

        layout.addLayout(header_row)
        layout.addWidget(self.output)
        layout.addLayout(input_row)

    def set_namespace(self, namespace: dict[str, Any]) -> None:
        """
        Replace the terminal namespace.
        """
        self.namespace.clear()
        self.namespace.update(namespace)
        self.console = code.InteractiveConsole(self.namespace)

    def update_namespace(self, **kwargs: Any) -> None:
        """
        Add or update objects in the terminal namespace.
        """
        self.namespace.update(kwargs)

    def execute_current_line(self) -> None:
        """
        Execute the current command line.
        """
        line = self.input_line.text()
        self.input_line.clear()

        if not line.strip():
            return

        visible_prompt = "... " if self.more_input_required else ">>> "
        self.write(f"{visible_prompt}{line}")

        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()

        try:
            with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
                self.more_input_required = self.console.push(line)
        except Exception as exc:
            stderr_buffer.write(f"{type(exc).__name__}: {exc}\n")
            self.more_input_required = False

        stdout_text = stdout_buffer.getvalue()
        stderr_text = stderr_buffer.getvalue()

        if stdout_text:
            self.write(stdout_text.rstrip())

        if stderr_text:
            self.write(stderr_text.rstrip())

        self.prompt = "... " if self.more_input_required else ">>> "
        self.prompt_label.setText(self.prompt)

        self.command_executed.emit(line)

    def write(self, text: str) -> None:
        """
        Write text to the terminal output.
        """
        self.output.appendPlainText(text)
        self.output.verticalScrollBar().setValue(
            self.output.verticalScrollBar().maximum()
        )

    def clear(self) -> None:
        """
        Clear the terminal output.
        """
        self.output.clear()