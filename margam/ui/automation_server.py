"""Localhost lifecycle/panel adapted from PyniteGUI (AGPL-3.0)."""
import asyncio
from datetime import datetime
import importlib.util
import json
import secrets
import socket
from threading import Thread

from PySide6.QtCore import QObject, Signal, QTimer, Qt
from PySide6.QtWidgets import (QApplication, QDialog, QFormLayout, QHBoxLayout, QLabel, QLineEdit,
                             QPushButton, QSpinBox, QTreeWidget, QTreeWidgetItem, QPlainTextEdit, QVBoxLayout)
from .automation_core import Permissions, QtCommandBridge, TOOLS


def dependencies_available():
    return all(importlib.util.find_spec(name) is not None for name in ("mcp", "uvicorn"))


class AutomationServer(QObject):
    changed = Signal()
    logged = Signal(str)
    network_event = Signal(str)

    def __init__(self, window):
        super().__init__(window)
        self.permissions = Permissions()
        self.bridge = QtCommandBridge(window, self.permissions)
        self.bridge.logged.connect(self.logged)
        self.network_event.connect(self.network_log)
        self.state = "Stopped"
        self.detail = ""
        self.endpoint = ""
        self.token = ""
        self.require_token = True
        self.requests = 0
        self.last_request = ""
        self.server = self.thread = self.listener = None
        self.failure = False
        self.timer = QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.poll)

    def network_log(self, message):
        if message in ("HTTP | authenticated", "HTTP | accepted"):
            self.requests += 1
            self.last_request = datetime.now().strftime("%H:%M:%S")
            self.changed.emit()
        else:
            self.logged.emit(message)

    def start(self, port=8765):
        if self.thread is not None:
            return False
        if not dependencies_available():
            self.state, self.detail = "Unavailable", "Optional dependencies missing. Install the optional mcp extra in the root environment; see the MCP guide."
            self.changed.emit()
            return False
        if type(port) is not int or not 0 <= port <= 65535:
            self.state, self.detail = "Failed", "Choose a valid localhost port."
            self.changed.emit()
            return False
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            listener.bind(("127.0.0.1", port))
            listener.listen(128)
            listener.setblocking(False)
            port = listener.getsockname()[1]
            from .automation_transport import build_application
            import uvicorn
            token = secrets.token_urlsafe(32) if self.require_token else ""
            app = build_application(self.bridge, self.permissions, token, port, self.network_event,
                                    require_token=self.require_token)
            self.server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, proxy_headers=False,
                access_log=False, log_config=None, log_level="critical", timeout_graceful_shutdown=2))
        except OSError:
            listener.close()
            self.state, self.detail = "Failed", "Could not bind localhost port. Choose another port or stop its existing listener."
            self.changed.emit()
            return False
        except Exception:
            listener.close()
            self.state, self.detail = "Failed", "Could not load the optional MCP server. Reinstall the mcp extra and restart the app."
            self.changed.emit()
            return False
        self.listener, self.token = listener, token
        self.endpoint = f"http://127.0.0.1:{port}/mcp"
        self.requests, self.last_request, self.failure = 0, "", False
        self.state, self.detail = "Starting", ""
        self.bridge.set_active(True)
        self.thread = Thread(target=self.serve, daemon=True, name="Margam automation server")
        self.thread.start()
        self.timer.start()
        self.logged.emit("Server | starting")
        self.changed.emit()
        return True

    def serve(self):
        try:
            asyncio.run(self.server.serve(sockets=[self.listener]))
        except BaseException:
            self.failure = True

    def stop(self):
        self.bridge.set_active(False)
        self.token = ""
        if self.thread is not None:
            self.state = "Stopping"
            self.server.should_exit = True
            self.logged.emit("Server | stopping")
        self.changed.emit()

    def poll(self):
        if self.thread is None:
            return
        if not self.thread.is_alive():
            self.thread.join()
            self.listener.close()
            self.thread = self.server = self.listener = None
            self.bridge.set_active(False)
            self.token = ""
            self.state = "Failed" if self.failure else "Stopped"
            self.detail = "Server exited unexpectedly. Restart the server to reconnect." if self.failure else ""
            self.timer.stop()
            self.logged.emit(f"Server | {self.state.lower()}")
            self.changed.emit()
        elif self.state == "Starting" and self.server.started:
            self.state = "Running"
            self.logged.emit("Server | running")
            self.changed.emit()

    def shutdown(self):
        """Revoke queued commands before stopping the listener on window close."""
        self.stop()
        self.timer.stop()
        if self.thread is not None:
            self.thread.join(3)
            if self.thread.is_alive():
                self.server.force_exit=True
                self.thread.join(1)
            if not self.thread.is_alive():
                self.poll()

    def configuration(self):
        entry = {"url": self.endpoint}
        if self.require_token:
            entry["headers"] = {"Authorization": f"Bearer {self.token}"}
        return {"mcpServers": {"Margam": entry}}


class AutomationPanel(QDialog):
    def __init__(self, window, server):
        super().__init__(window)
        self.server = server
        self.setProperty("margam_non_model_controls", True)
        self.setWindowTitle("Automation Server")
        self.resize(620, 690)
        layout = QVBoxLayout(self)
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.status)
        form = QFormLayout()
        self.port = QSpinBox()
        self.port.setRange(1024, 65535)
        self.port.setValue(8765)
        form.addRow("Localhost port", self.port)
        self.endpoint = QLineEdit()
        self.endpoint.setReadOnly(True)
        form.addRow("Endpoint", self.endpoint)
        self.token = QLineEdit()
        self.token.setReadOnly(True)
        self.token.setEchoMode(QLineEdit.EchoMode.Password)
        form.addRow("Bearer token", self.token)
        layout.addLayout(form)
        row = QHBoxLayout()
        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(lambda: server.start(self.port.value()))
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(server.stop)
        self.copy_token = QPushButton("Copy token")
        self.copy_token.clicked.connect(lambda: QApplication.clipboard().setText(server.token))
        self.copy_config = QPushButton("Copy configuration")
        self.copy_config.clicked.connect(lambda: QApplication.clipboard().setText(json.dumps(server.configuration(), indent=2)))
        for button in (self.start_button, self.stop_button, self.copy_token, self.copy_config):
            row.addWidget(button)
        layout.addLayout(row)
        label = QLabel("A tool needs both its group and individual switch enabled. Permissions apply immediately to discovery and every command. Edits and training start disabled.")
        label.setWordWrap(True)
        layout.addWidget(label)
        self.permissions = QTreeWidget()
        self.permissions.setHeaderLabels(["Permission / Tool"])
        groups = {}
        for name, (group, description) in TOOLS.items():
            if group not in groups:
                item = QTreeWidgetItem(self.permissions, [group])
                item.setData(0, Qt.ItemDataRole.UserRole, ("group", group))
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(0, Qt.CheckState.Checked if server.permissions.groups[group] else Qt.CheckState.Unchecked)
                item.setExpanded(True)
                groups[group] = item
            item = QTreeWidgetItem(groups[group], [name])
            item.setData(0, Qt.ItemDataRole.UserRole, ("tool", name))
            item.setToolTip(0, description)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(0, Qt.CheckState.Checked)
        self.permissions.itemChanged.connect(self.permission_changed)
        layout.addWidget(self.permissions)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(200)
        self.log.setPlaceholderText("Request names/status codes appear here. Tokens, arguments, model values and paths are omitted.")
        layout.addWidget(self.log)
        note = QLabel("This server controls this project window. Starts off; binds only 127.0.0.1. "
                      "Tokens change on every start. Closing this panel leaves the server running; Stop or close the project window to stop it. "
                      "No file-management or arbitrary-code tools are exposed. Edits use revision checks; manual editing stays available when idle.")
        note.setWordWrap(True)
        layout.addWidget(note)
        server.changed.connect(self.update_status)
        server.logged.connect(self.append_log)
        self.update_status()

    def permission_changed(self, item, column):
        kind, name = item.data(0, Qt.ItemDataRole.UserRole)
        enabled = item.checkState(0) == Qt.CheckState.Checked
        if kind == "group":
            self.server.permissions.set_group(name, enabled)
        else:
            self.server.permissions.set_tool(name, enabled)
        self.append_log(f"Permission | {name} | {'enabled' if enabled else 'disabled'}")

    def append_log(self, message):
        self.log.appendPlainText(f"{datetime.now().strftime('%H:%M:%S')} | {message}")

    def update_status(self):
        server = self.server
        connection = f"Last accepted request {server.last_request} | {server.requests} requests" if server.last_request else "Waiting for a client"
        self.status.setText(f"{server.state} | {connection}" + (f"\n{server.detail}" if server.detail else ""))
        self.endpoint.setText(server.endpoint)
        self.token.setText(server.token)
        self.token.setEnabled(server.require_token)
        self.token.setPlaceholderText("" if server.require_token else "Not required")
        self.port.setEnabled(server.thread is None)
        self.start_button.setEnabled(server.thread is None)
        self.stop_button.setEnabled(server.state in ("Starting", "Running"))
        self.copy_token.setEnabled(server.state == "Running" and server.require_token and bool(server.token))
        self.copy_config.setEnabled(server.state == "Running")
