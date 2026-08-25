"""Top-level application window.

Owns the worker thread, the connection bar, the per-motor tabs, the global
E-stop, presets, and the log. The window only ever *posts commands* to the
worker and *reacts to signals* from it - it never performs IO itself.
"""

from __future__ import annotations

import time

from PySide6.QtCore import Qt, QThread, QTimer, Slot
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDockWidget, QHBoxLayout, QInputDialog, QLabel,
    QMainWindow, QMessageBox, QPlainTextEdit, QPushButton, QScrollArea, QSpinBox,
    QTabWidget, QVBoxLayout, QWidget,
)

from .. import protocol as proto
from ..bus import Motor
from ..calibration_store import CalibrationRecord, CalibrationStore
from ..datalog import TelemetryLogger
from ..presets import Preset, PresetStore
from ..protocol import MotorStatus
from ..transport import (
    SerialATTransport, SocketCANTransport,
    auto_detect_serial_port, list_serial_port_details,
)
from .. import worker as wk
from .dashboard import MotorDashboard
from .device_dialog import DeviceDialog
from .motor_panel import MotorPanel
from .sim_dock import SimBridgeDock

PLOT_REFRESH_MS = 33  # ~30 FPS

#: Minimum seconds between error *popups*. Every error still reaches the log;
#: only the modal dialog is throttled, so a failing bus (one error per motor
#: per control cycle) cannot stack hundreds of dialogs and freeze the UI.
ERROR_DIALOG_MIN_INTERVAL_S = 3.0


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RobStride Control")
        self.resize(1180, 780)

        self.panels: dict[int, MotorPanel] = {}
        self.presets = PresetStore().load()
        # Software zero/direction trim per motor, persisted so a captured zero
        # survives a GUI restart (the hardware zero is saved to the motor's flash).
        self.calibrations = CalibrationStore().load()
        # Mirrors the live graph feedback (pos/vel/torque/temp + power) to a
        # separate .txt data file for offline review.
        self.datalog = TelemetryLogger()
        self._connected = False
        # Names of the CAN buses currently connected, in the order added. Each
        # Connect adds one adapter; the list drives the bus picker and label.
        self._bus_names: list[str] = []
        # Motors whose zero was (re)set since their last enable. The next enable
        # for these asks for confirmation: a fresh zero moves the reference
        # frame, so "hold current position" is safe but the operator should
        # still eyeball the rig before energising it.
        self._zeroed_since_enable: set[int] = set()
        self.device_dialog: DeviceDialog | None = None
        # Remembers whether the window was maximized before entering fullscreen
        # so leaving fullscreen restores that state instead of a small window.
        self._was_maximized = False
        # When the last error popup was shown (monotonic); -inf so the first
        # error always gets a dialog.
        self._last_error_dialog_time = float("-inf")

        self._start_worker()
        self._build_ui()
        self._build_shortcuts()
        self._refresh_serial_ports()
        self._reload_preset_combo()

        self._plot_timer = QTimer(self)
        self._plot_timer.timeout.connect(self._refresh_plots)
        self._plot_timer.start(PLOT_REFRESH_MS)

    # -- worker / thread ---------------------------------------------------------

    def _start_worker(self) -> None:
        self.thread = QThread(self)
        self.worker = wk.ControlWorker()
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)

        self.worker.statusUpdated.connect(self._on_status)
        self.worker.powerUpdated.connect(self._on_power)
        self.worker.connectionChanged.connect(self._on_connection_changed)
        self.worker.scanFinished.connect(self._on_scan_finished)
        self.worker.busCollision.connect(self._on_bus_collision)
        self.worker.inventoryReady.connect(self._on_inventory_ready)
        self.worker.motorEnabledChanged.connect(self._on_motor_enabled)
        self.worker.calibrationChanged.connect(self._on_calibration_changed)
        self.worker.rangeLimitsChanged.connect(self._on_range_limits_changed)
        self.worker.motorIdChanged.connect(self._on_motor_id_changed)
        self.worker.sweepStopped.connect(self._on_sweep_stopped)
        self.worker.zeroStateUpdated.connect(self._on_zero_state)
        self.worker.log.connect(self._append_log)
        self.worker.error.connect(self._on_error)

        self.thread.start()

    # -- UI construction ---------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)

        # The connection bar is a long single row; scroll it horizontally so a
        # narrow window can still shrink instead of the toolbar forcing a wide
        # minimum size.
        bar = self._build_connection_bar()
        bar_scroll = QScrollArea()
        bar_scroll.setWidget(bar)
        bar_scroll.setWidgetResizable(True)
        bar_scroll.setFrameShape(QScrollArea.NoFrame)
        bar_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        bar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        bar_scroll.setFixedHeight(bar.sizeHint().height() + 16)  # room for scrollbar
        root.addWidget(bar_scroll)

        self.tabs = QTabWidget()
        self.tabs.setMovable(True)
        root.addWidget(self.tabs, 1)

        # The overview dashboard is the primary operating surface: all motors on
        # one screen. The per-motor detail tabs stay for deep work (MIT gains,
        # sweep, device tools). Kept as tab 0 so it opens first.
        self._build_dashboard()

        self.setCentralWidget(central)
        self._build_log_dock()
        self._build_preset_dock()
        self._build_sim_dock()

    def _build_dashboard(self) -> None:
        self.dashboard = MotorDashboard()
        # The dashboard re-emits each row's high-level signals; route them to the
        # same worker-command handlers the detail panels use, so both surfaces
        # drive the motor identically.
        self.dashboard.enableToggled.connect(self._on_enable_toggled)
        self.dashboard.targetChanged.connect(self._on_target_changed)
        self.dashboard.modeChanged.connect(
            lambda did, m: self.worker.post(wk.SetMode(did, m)))
        self.dashboard.rangeLimitsEdited.connect(self._on_range_limits_edited)
        self.dashboard.lockChanged.connect(self._on_lock_changed)
        self.dashboard.log.connect(self._append_log)
        self.tabs.addTab(self.dashboard, "Overview")

    def _build_shortcuts(self) -> None:
        # F11 toggles full screen; Esc leaves it. Both are no-ops otherwise.
        QShortcut(QKeySequence(Qt.Key_F11), self, activated=self._toggle_fullscreen)
        QShortcut(QKeySequence(Qt.Key_Escape), self, activated=self._exit_fullscreen)

    def _toggle_fullscreen(self) -> None:
        if self.isFullScreen():
            self._exit_fullscreen()
        else:
            # Capture the pre-fullscreen state so Esc/F11 can restore it.
            self._was_maximized = self.isMaximized()
            self.showFullScreen()

    def _exit_fullscreen(self) -> None:
        if not self.isFullScreen():
            return
        # showNormal() would forget a maximized window; restore it explicitly.
        if self._was_maximized:
            self.showMaximized()
        else:
            self.showNormal()

    def _build_connection_bar(self) -> QWidget:
        bar = QWidget()
        lay = QHBoxLayout(bar)

        self.transport_combo = QComboBox()
        self.transport_combo.addItems(["Serial (AT)", "SocketCAN"])
        self.transport_combo.currentIndexChanged.connect(self._on_transport_changed)
        lay.addWidget(QLabel("Transport"))
        lay.addWidget(self.transport_combo)

        self.port_combo = QComboBox()
        self.port_combo.setEditable(True)
        self.port_combo.setMinimumWidth(160)
        lay.addWidget(self.port_combo)

        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.setToolTip("Re-scan for available serial ports")
        self.refresh_btn.clicked.connect(self._refresh_serial_ports)
        lay.addWidget(self.refresh_btn)

        self.port_status = QLabel()
        self.port_status.setToolTip("Whether a usable port was auto-detected")
        lay.addWidget(self.port_status)

        self.connect_btn = QPushButton("Connect")
        self.connect_btn.setToolTip(
            "Connect the selected adapter. Connecting again with a different "
            "port ADDS a second CAN bus alongside the first.")
        self.connect_btn.clicked.connect(self._on_connect_clicked)
        lay.addWidget(self.connect_btn)

        self.disconnect_btn = QPushButton("Disconnect all")
        self.disconnect_btn.setToolTip("Close every connected CAN bus")
        self.disconnect_btn.setEnabled(False)
        self.disconnect_btn.clicked.connect(
            lambda: self.worker.post(wk.Disconnect()))
        lay.addWidget(self.disconnect_btn)

        self.bus_label = QLabel("no bus")
        self.bus_label.setToolTip("CAN buses currently connected")
        self.bus_label.setStyleSheet("color:#888;")
        lay.addWidget(self.bus_label)

        lay.addSpacing(16)
        lay.addWidget(QLabel("Add motor id"))
        self.add_id_spin = QSpinBox()
        self.add_id_spin.setRange(0, 127)  # id 0 is valid; motors ship/reset to it
        self.add_id_spin.setValue(0)
        lay.addWidget(self.add_id_spin)
        self.model_combo = QComboBox()
        self.model_combo.addItems(list(proto.MODELS))
        self.model_combo.setCurrentText(proto.DEFAULT_MODEL)
        lay.addWidget(self.model_combo)
        # Which bus a manually added motor lives on. Scan/Detect fill this in
        # automatically (the router remembers which adapter answered), so this
        # only matters for ids typed in by hand.
        self.bus_combo = QComboBox()
        self.bus_combo.setToolTip("CAN bus this motor is wired to")
        self.bus_combo.setMinimumWidth(90)
        lay.addWidget(self.bus_combo)

        self.add_btn = QPushButton("Add")
        self.add_btn.clicked.connect(self._on_add_motor_clicked)
        lay.addWidget(self.add_btn)

        lay.addWidget(QLabel("Scan"))
        self.scan_start_spin = QSpinBox()
        self.scan_start_spin.setRange(0, 127)  # id 0 is valid; some motors ship/reset to it
        self.scan_start_spin.setValue(0)
        self.scan_end_spin = QSpinBox()
        self.scan_end_spin.setRange(0, 127)
        self.scan_end_spin.setValue(127)  # cover the full id space; motors ship/reset to 127
        lay.addWidget(self.scan_start_spin)
        lay.addWidget(QLabel("to"))
        lay.addWidget(self.scan_end_spin)
        self.scan_btn = QPushButton("Scan")
        self.scan_btn.clicked.connect(self._on_scan_clicked)
        lay.addWidget(self.scan_btn)

        self.setid_btn = QPushButton("Set ID...")
        self.setid_btn.setToolTip("Reassign a motor's CAN id (one motor on the bus)")
        self.setid_btn.clicked.connect(self._on_set_id_clicked)
        lay.addWidget(self.setid_btn)

        self.devices_btn = QPushButton("Devices...")
        self.devices_btn.setToolTip("List motors with their unique ids and assign CAN ids")
        self.devices_btn.clicked.connect(self._on_devices_clicked)
        lay.addWidget(self.devices_btn)

        lay.addStretch(1)

        self.disable_all_btn = QPushButton("Disable all")
        self.disable_all_btn.setToolTip("De-energise every motor (standby / limp)")
        self.disable_all_btn.clicked.connect(self._on_disable_all)
        lay.addWidget(self.disable_all_btn)

        self.zero_all_btn = QPushButton("Zero all")
        self.zero_all_btn.setToolTip(
            "Set every motor's zero to its current position")
        self.zero_all_btn.clicked.connect(self._on_zero_all)
        lay.addWidget(self.zero_all_btn)

        self.estop_btn = QPushButton("E-STOP")
        self.estop_btn.setCheckable(True)
        self.estop_btn.setMinimumWidth(120)
        self.estop_btn.setStyleSheet(
            "QPushButton{background:#b71c1c;color:white;font-weight:bold;}"
            "QPushButton:checked{background:#ff5252;}")
        self.estop_btn.clicked.connect(self._on_estop)
        lay.addWidget(self.estop_btn)
        return bar

    def _build_log_dock(self) -> None:
        self.log_dock = QDockWidget("Log", self)
        container = QWidget()
        lay = QVBoxLayout(container)
        lay.setContentsMargins(4, 4, 4, 4)

        # Telemetry recording is opt-in: nothing is written to disk unless the
        # user ticks this box. The label shows where the data went.
        controls = QHBoxLayout()
        self.record_check = QCheckBox("Record telemetry to file")
        self.record_check.setToolTip(
            "Save the live graph feedback (position/velocity/torque/temp) to a "
            ".txt data file while ticked")
        self.record_check.toggled.connect(self._on_record_toggled)
        controls.addWidget(self.record_check)
        self.record_path_label = QLabel("")
        self.record_path_label.setStyleSheet("color:#888;")
        controls.addWidget(self.record_path_label, 1)
        lay.addLayout(controls)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(2000)
        # Without a minimum height Qt collapses the bottom dock to a thin sliver,
        # so the log is effectively invisible. Give it a usable floor and a
        # sensible initial size.
        self.log_view.setMinimumHeight(120)
        lay.addWidget(self.log_view)

        self.log_dock.setWidget(container)
        self.log_dock.setMinimumHeight(140)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.log_dock)
        self.resizeDocks([self.log_dock], [220], Qt.Vertical)

    def _on_record_toggled(self, checked: bool) -> None:
        if checked:
            path = self.datalog.start()
            if path is None:
                self.record_check.setChecked(False)
                self._append_log("ERROR: could not open telemetry log file")
                return
            self.record_path_label.setText(f"recording -> {path}")
            self._append_log(f"Telemetry recording started: {path}")
        else:
            self.datalog.stop()
            self.record_path_label.setText("")
            self._append_log("Telemetry recording stopped")

    def _build_preset_dock(self) -> None:
        dock = QDockWidget("Presets", self)
        w = QWidget()
        lay = QVBoxLayout(w)
        self.preset_combo = QComboBox()
        lay.addWidget(self.preset_combo)
        apply_btn = QPushButton("Apply to current motor")
        apply_btn.clicked.connect(self._apply_preset)
        save_btn = QPushButton("Save current as preset...")
        save_btn.clicked.connect(self._save_preset)
        del_btn = QPushButton("Delete preset")
        del_btn.clicked.connect(self._delete_preset)
        for b in (apply_btn, save_btn, del_btn):
            lay.addWidget(b)
        lay.addStretch(1)
        dock.setWidget(w)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)

    def _build_sim_dock(self) -> None:
        """Add the MuJoCo bridge on/off dock (tabbed under the Presets dock)."""
        self.sim_dock = SimBridgeDock(self.worker, self._sim_joint_map, parent=self)
        self.addDockWidget(Qt.RightDockWidgetArea, self.sim_dock)
        # Stack it with the Presets dock so it doesn't widen the window.
        for existing in self.findChildren(QDockWidget):
            if existing is not self.sim_dock and \
                    self.dockWidgetArea(existing) == Qt.RightDockWidgetArea:
                self.tabifyDockWidget(existing, self.sim_dock)
                break

    def _sim_joint_map(self) -> dict[str, int]:
        """Map MuJoCo joint names to connected CAN ids for the sim bridge.

        Prefers a ``sim_bindings.json`` (``{joint_name: can_id}``) so an operator
        can pin the real joint<->motor identification; otherwise falls back to
        ``revolute_<id> -> <id>`` for every connected motor, which matches the
        naming the MuJoCo scripts assume. Only ids that are actually connected are
        kept, so a stale bindings file can't command a motor that isn't present.

        The file is looked for at ``mujoco/2_run_bridge/sim_bindings.json`` (its
        home in the staged MuJoCo folder) first, then a plain ``sim_bindings.json``
        in the working directory for backward compatibility.
        """
        import json
        from pathlib import Path

        connected = set(self.panels)
        candidates = [
            Path("mujoco") / "2_run_bridge" / "sim_bindings.json",
            Path("sim_bindings.json"),
        ]
        bindings_path = next((p for p in candidates if p.is_file()), None)
        if bindings_path is not None:
            try:
                raw = json.loads(bindings_path.read_text())
                mapped = {str(name): int(cid) for name, cid in raw.items()
                          if int(cid) in connected}
                if mapped:
                    return mapped
                self._append_log(
                    f"{bindings_path} matched no connected motors; using default map")
            except (ValueError, TypeError, OSError) as exc:
                self._append_log(f"{bindings_path} ignored ({exc}); using default map")
        return {f"revolute_{did}": did for did in sorted(connected)}

    # -- connection bar logic ----------------------------------------------------

    def _is_serial(self) -> bool:
        return self.transport_combo.currentIndex() == 0

    def _on_transport_changed(self, _index: int) -> None:
        self.port_combo.clear()
        if self._is_serial():
            self._refresh_serial_ports()
        else:
            self.port_combo.addItems(["can0", "can1"])
            self.port_status.setText("")
            self._set_connect_enabled(True)

    def _refresh_serial_ports(self) -> None:
        if not self._is_serial():
            return
        self.port_combo.clear()
        ports = list_serial_port_details()
        for info in ports:
            # show a friendly label, but keep the raw device path as the value
            self.port_combo.addItem(info.label(), info.device)

        if ports:
            auto = auto_detect_serial_port()
            idx = self.port_combo.findData(auto)
            self.port_combo.setCurrentIndex(max(idx, 0))
            self._set_port_status(available=True, count=len(ports))
            self._set_connect_enabled(True)
        else:
            # No fabricated default: be honest that nothing is plugged in.
            self._set_port_status(available=False, count=0)
            self._set_connect_enabled(False)

    def _set_port_status(self, *, available: bool, count: int) -> None:
        if available:
            self.port_status.setText(f"✓ {count} port{'s' if count != 1 else ''}")
            self.port_status.setStyleSheet("color:#2e7d32;")  # green
        else:
            self.port_status.setText("✗ no port")
            self.port_status.setStyleSheet("color:#b71c1c;")  # red

    def _set_connect_enabled(self, enabled: bool) -> None:
        # Never block a Disconnect; only gate the initial Connect.
        if not self._connected:
            self.connect_btn.setEnabled(enabled)

    def _current_port(self) -> str:
        data = self.port_combo.currentData()
        if data:
            return str(data)
        return self.port_combo.currentText().strip()

    def _build_transport(self):
        target = self._current_port()
        if self._is_serial():
            if not target:
                raise RuntimeError(
                    "No serial port available. Plug in the USB-CAN adapter and "
                    "press Refresh.")
            return SerialATTransport(target)
        return SocketCANTransport(target or "can0")

    def _derive_bus_name(self, target: str) -> str:
        """A short, unique name for the bus this adapter will become.

        Taken from the device path so it is recognisable in logs and the bus
        picker: "/dev/ttyUSB0" -> "ttyUSB0", "can0" -> "can0". Suffixed if that
        name is already taken, since a bus name is the router's key and a
        duplicate would silently REPLACE the earlier adapter.
        """
        base = (target.rsplit("/", 1)[-1] or "bus").strip()
        name, n = base, 2
        while name in self._bus_names:
            name, n = f"{base}#{n}", n + 1
        return name

    def _on_connect_clicked(self) -> None:
        """Connect the selected adapter, ADDING it to any already connected.

        Connect is additive rather than a toggle: pick the first adapter and
        press Connect, then pick the second and press Connect again, and both
        buses run at once. "Disconnect all" closes them.
        """
        try:
            transport = self._build_transport()
        except Exception as e:
            self._on_error(str(e))
            return
        bus_name = self._derive_bus_name(self._current_port())
        # Only the FIRST bus adopts the motors already tabbed, which preserves
        # the old single-adapter flow (add a motor, then Connect). Later buses
        # start empty: those existing panels belong to an earlier adapter, and
        # handing them over would reroute motors that are already wired up.
        if not self._bus_names:
            if not self.panels:
                self._add_motor(self.add_id_spin.value(),
                                self.model_combo.currentText(), bus_name)
            motors = [Motor(device_id=did, model=p.model)
                      for did, p in self.panels.items()]
        else:
            motors = []
        self._bus_names.append(bus_name)
        self._refresh_bus_widgets()
        self.worker.post(wk.Connect(transport=transport, motors=motors,
                                    bus_name=bus_name))
        self._append_log(f"Connecting bus '{bus_name}' via {transport.name}")

    def _refresh_bus_widgets(self) -> None:
        """Keep the bus label and the per-motor bus picker in step with
        ``_bus_names``, preserving the current pick where possible."""
        self.bus_label.setText(", ".join(self._bus_names) or "no bus")
        previous = self.bus_combo.currentText()
        self.bus_combo.clear()
        self.bus_combo.addItems(self._bus_names)
        index = self.bus_combo.findText(previous)
        self.bus_combo.setCurrentIndex(index if index >= 0 else 0)
        self.disconnect_btn.setEnabled(bool(self._bus_names))

    def _selected_bus(self) -> str | None:
        """Bus chosen for a hand-added motor, or None while only one exists."""
        return self.bus_combo.currentText() or None if len(self._bus_names) > 1 else None

    # -- motors ------------------------------------------------------------------

    def _on_add_motor_clicked(self) -> None:
        # A hand-typed id has never been seen on the wire, so the router cannot
        # know its bus - take it from the picker.
        self._add_motor(self.add_id_spin.value(), self.model_combo.currentText(),
                        self._selected_bus())

    def _on_scan_clicked(self) -> None:
        start = self.scan_start_spin.value()
        end = max(start, self.scan_end_spin.value())
        self.worker.post(wk.Scan(start=start, end=end))

    def _on_devices_clicked(self) -> None:
        if not self._connected:
            self._on_error("Connect to the bus before listing devices")
            return
        if self.device_dialog is None:
            dialog = DeviceDialog(self)
            dialog.detectRequested.connect(self._request_inventory)
            dialog.assignRequested.connect(self._on_device_assign)
            self.device_dialog = dialog
        self.device_dialog.show()
        self.device_dialog.raise_()
        self.device_dialog.activateWindow()
        self._request_inventory()

    def _request_inventory(self) -> None:
        start = self.scan_start_spin.value()
        end = max(start, self.scan_end_spin.value())
        self.worker.post(wk.Inventory(start=start, end=end))

    def _on_device_assign(self, current_id: int, new_id: int) -> None:
        self.worker.post(wk.SetMotorId(current_id=current_id, new_id=new_id))
        # Re-detect so the table reflects the new id after the motor acks.
        self._request_inventory()

    @Slot(list)
    def _on_inventory_ready(self, items: list) -> None:
        if self.device_dialog is not None:
            self.device_dialog.populate(items)
        # Detect alone now surfaces every motor: add a tab per responding CAN id.
        # _add_motor is idempotent (an already-tabbed id is re-focused, not
        # duplicated), so re-running Detect never creates duplicate tabs.
        model = self.model_combo.currentText()
        for entry in items:
            try:
                can_id = int(entry["can_id"])
            except (KeyError, TypeError, ValueError):
                continue
            self._add_motor(can_id, model)

    def _on_set_id_clicked(self) -> None:
        if not self._connected:
            self._on_error("Connect to the bus before changing a motor id")
            return
        cur, ok = QInputDialog.getInt(self, "Set Motor ID", "Current CAN id:",
                                      1, 1, 127)
        if not ok:
            return
        new, ok = QInputDialog.getInt(self, "Set Motor ID", "New CAN id:",
                                      cur, 1, 127)
        if not ok or new == cur:
            return
        confirm = QMessageBox.question(
            self, "Set Motor ID",
            f"Reassign motor {cur} -> {new}?\n\n"
            "Only ONE motor should be on the bus, or every motor sharing id "
            f"{cur} will be changed. Power-cycle afterwards to confirm it sticks.")
        if confirm == QMessageBox.Yes:
            self.worker.post(wk.SetMotorId(current_id=cur, new_id=new))

    def _refresh_sim_map(self) -> None:
        """Rebuild the sim bridge's joint map after the motor set changes, so a
        motor connected while the bridge is streaming is picked up without an
        off/on toggle. Guarded with getattr so it is safe if the motor set changes
        before the sim dock exists."""
        dock = getattr(self, "sim_dock", None)
        if dock is not None:
            dock.refresh_joint_map()

    def _remove_panel(self, device_id: int) -> None:
        self.dashboard.remove_motor(device_id)
        panel = self.panels.pop(device_id, None)
        if panel is None:
            return
        idx = self.tabs.indexOf(panel)
        if idx >= 0:
            self.tabs.removeTab(idx)
        panel.deleteLater()
        self._refresh_sim_map()

    def _add_motor(self, device_id: int, model: str,
                   bus_name: str | None = None) -> MotorPanel:
        if device_id in self.panels:
            self.tabs.setCurrentWidget(self.panels[device_id])
            return self.panels[device_id]
        panel = MotorPanel(device_id, model)
        panel.enableToggled.connect(self._on_enable_toggled)
        panel.zeroRequested.connect(self._on_zero_requested)
        panel.modeChanged.connect(lambda did, m: self.worker.post(wk.SetMode(did, m)))
        panel.targetChanged.connect(self._on_target_changed)
        panel.calibrationChanged.connect(self._on_panel_calibration_changed)
        panel.captureZeroRequested.connect(
            lambda did: self.worker.post(wk.CaptureZero(did)))
        panel.zeroStateRequested.connect(
            lambda did: self.worker.post(wk.ReadZeroState(did)))
        panel.rangeCalibrationToggled.connect(self._on_range_calibration_toggled)
        panel.rangeLimitsEdited.connect(self._on_range_limits_edited)
        panel.sweepChanged.connect(self._on_sweep_changed)
        self.panels[device_id] = panel
        self.tabs.addTab(panel, f"Motor {device_id}")
        self.tabs.setCurrentWidget(panel)
        # Mirror the motor onto the overview dashboard so it shows up there too.
        self.dashboard.add_motor(device_id, model)
        # Restore this motor's saved software calibration, if any, and push it to
        # the worker so display and commands use it from the first frame.
        stored = self.calibrations.get(device_id)
        if stored is not None:
            panel.set_calibration_display(stored.direction, stored.offset)
            self.worker.post(wk.SetCalibration(device_id, stored.direction, stored.offset))
            self.dashboard.set_locked(device_id, stored.locked)
            # Restore the calibrated travel range so inputs and motion are clamped
            # from the first frame, mirroring the zero/direction restore above.
            if stored.pos_min is not None or stored.pos_max is not None:
                panel.set_position_limits(stored.pos_min, stored.pos_max)
                self.dashboard.set_position_limits(device_id, stored.pos_min, stored.pos_max)
                self.worker.post(wk.SetRangeLimits(device_id, stored.pos_min, stored.pos_max))
        self._append_log(f"Added motor {device_id} ({model})")
        # A motor added after Connect must be registered with the worker too,
        # else it gets a tab but is never polled/driven (Connect only seeds the
        # motors open at connect time).
        if self._connected:
            self.worker.post(wk.AddMotor(device_id=device_id, model=model,
                                         bus_name=bus_name))
        # Pick this motor up in the sim bridge's joint map if it is streaming, so
        # a motor connected after the bridge started needs no off/on toggle.
        self._refresh_sim_map()
        return panel

    @Slot(int, bool)
    def _on_zero_requested(self, device_id: int) -> None:
        self.worker.post(wk.SetZero(device_id))
        # Arm the one-shot enable confirmation for this motor.
        self._zeroed_since_enable.add(device_id)

    def _on_enable_toggled(self, device_id: int, enable: bool) -> None:
        if not self._connected:
            self._on_error("Connect to the bus before enabling a motor")
            self.panels[device_id].set_enabled_state(False)
            return
        if enable and device_id in self._zeroed_since_enable:
            if not self._confirm_enable_after_zero(device_id):
                # Operator backed out: leave the motor disabled and the button
                # un-toggled, and keep the flag armed for the next attempt.
                self.panels[device_id].set_enabled_state(False)
                return
            self._zeroed_since_enable.discard(device_id)
        self.worker.post(wk.Enable(device_id) if enable else wk.Disable(device_id))

    def _confirm_enable_after_zero(self, device_id: int) -> bool:
        confirm = QMessageBox.question(
            self, "Enable motor",
            f"Motor {device_id} was zeroed since it was last enabled.\n\n"
            "It will come up HOLDING its current position, but check the rig is "
            "clear before energising it. Enable now?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        return confirm == QMessageBox.Yes

    def _on_sweep_stopped(self, device_id: int) -> None:
        # The worker auto-stopped a sweep (disable / E-stop / range cutout);
        # reflect it so the button doesn't show a phantom "Stop sweep".
        panel = self.panels.get(device_id)
        if panel is not None:
            panel.set_sweep_active(False)

    @Slot(int, object)
    def _on_target_changed(self, device_id: int, changes: dict) -> None:
        self.worker.post(wk.SetTarget(device_id=device_id, **changes))

    @Slot(int, object)
    def _on_sweep_changed(self, device_id: int, cfg: dict) -> None:
        if cfg["enabled"] and not self._connected:
            self._on_error("Connect and enable the motor before starting a sweep")
            self.panels[device_id].set_sweep_active(False)
            return
        self.worker.post(wk.SetSweep(
            device_id=device_id, enabled=cfg["enabled"],
            from_pos=cfg["from_pos"], to_pos=cfg["to_pos"], period=cfg["period"]))

    # -- worker signal slots -----------------------------------------------------

    @Slot(int, object)
    def _on_status(self, device_id: int, status: MotorStatus) -> None:
        panel = self.panels.get(device_id)
        if panel is not None:
            panel.update_status(status)
        self.dashboard.update_status(device_id, status)
        faults = ",".join(name for name, flag in (
            ("stall", status.stalled), ("encoder", status.encoder_fault),
            ("overtemp", status.overtemperature), ("overcur", status.overcurrent),
            ("undervolt", status.undervoltage)) if flag)
        self.datalog.log_status(device_id, status.position, status.velocity,
                                status.torque, status.temperature, faults)

    @Slot(int, object)
    def _on_power(self, device_id: int, power: object) -> None:
        panel = self.panels.get(device_id)
        if panel is not None:
            panel.update_power(power)
        self.dashboard.update_power(device_id, power)
        self.datalog.update_power(device_id, power.vbus, power.iq, power.power)

    @Slot(bool)
    def _on_connection_changed(self, connected: bool) -> None:
        self._connected = connected
        # The adapter selectors stay live while connected: that is how a SECOND
        # bus gets added. Connect is additive now, so the button keeps its label
        # instead of flipping to "Disconnect" (that is its own button).
        self.connect_btn.setEnabled(True)
        self.transport_combo.setEnabled(True)
        self.port_combo.setEnabled(True)
        self.refresh_btn.setEnabled(True)
        if not connected:
            # Teardown closes every bus at once, so drop them all.
            self._bus_names.clear()
            self._refresh_bus_widgets()
            # Re-check availability now that the ports are free again.
            self._refresh_serial_ports()

    @Slot(list)
    def _on_scan_finished(self, ids: list) -> None:
        # Use the model picked in the Model combo, NOT DEFAULT_MODEL. Hardcoding
        # the default silently registered every scanned motor as an rs-04, so an
        # rs-03 (or any non-default model) got rs-04 MIT scaling: its velocity,
        # torque and kp/kd codes were computed against the wrong full-scale
        # values, and the decoded feedback was wrong by the same factor. The
        # motor answers the ping, the tab appears, and then nothing behaves.
        # This mirrors what _on_inventory_ready (the Detect button) already did.
        model = self.model_combo.currentText()
        for device_id in ids:
            self._add_motor(int(device_id), model)

    @Slot(list)
    def _on_bus_collision(self, ids: list) -> None:
        id_str = ", ".join(str(i) for i in ids)
        msg = (
            f"CAN id collision detected on id {id_str}.\n\n"
            "More than one motor is answering on the same CAN id, so every "
            "command drives them together — they cannot be controlled "
            "independently.\n\n"
            "To fix:\n"
            "1. Leave only ONE of the colliding motors powered on the bus.\n"
            "2. Use 'Set ID...' to give it a unique CAN id.\n"
            "3. Power-cycle that motor, then reconnect the others.")
        self._append_log(f"WARNING: CAN id collision on {id_str} - "
                         "multiple motors share an id and move together")
        QMessageBox.warning(self, "Bus ID Collision", msg)

    @Slot(int, bool)
    def _on_motor_enabled(self, device_id: int, enabled: bool) -> None:
        panel = self.panels.get(device_id)
        if panel is not None:
            panel.set_enabled_state(enabled)
        self.dashboard.set_enabled_state(device_id, enabled)

    def _on_panel_calibration_changed(self, device_id: int, direction: int,
                                      offset: float) -> None:
        """User edited invert/offset in the panel: push to the worker and persist."""
        self.worker.post(wk.SetCalibration(device_id, direction, offset))
        self._persist_calibration(device_id, direction, offset)

    def _persist_calibration(self, device_id: int, direction: int, offset: float) -> None:
        # Preserve any calibrated range and lock state already stored for this
        # motor: a zero/direction change must not silently drop the travel limits
        # or unlock calibration.
        prev = self.calibrations.get(device_id)
        pos_min = prev.pos_min if prev is not None else None
        pos_max = prev.pos_max if prev is not None else None
        locked = prev.locked if prev is not None else True
        self.calibrations.upsert(
            CalibrationRecord(device_id, direction, offset, pos_min, pos_max, locked))
        self.calibrations.save()

    def _persist_range(self, device_id: int, pos_min: float | None,
                       pos_max: float | None) -> None:
        # Symmetric to _persist_calibration: keep the existing direction/offset/lock.
        prev = self.calibrations.get(device_id)
        direction = prev.direction if prev is not None else 1
        offset = prev.offset if prev is not None else 0.0
        locked = prev.locked if prev is not None else True
        self.calibrations.upsert(
            CalibrationRecord(device_id, direction, offset, pos_min, pos_max, locked))
        self.calibrations.save()

    def _on_lock_changed(self, device_id: int, locked: bool) -> None:
        """Persist the dashboard calibration lock, keeping all other fields."""
        prev = self.calibrations.get(device_id)
        direction = prev.direction if prev is not None else 1
        offset = prev.offset if prev is not None else 0.0
        pos_min = prev.pos_min if prev is not None else None
        pos_max = prev.pos_max if prev is not None else None
        self.calibrations.upsert(
            CalibrationRecord(device_id, direction, offset, pos_min, pos_max, locked))
        self.calibrations.save()

    def _on_range_calibration_toggled(self, device_id: int, active: bool) -> None:
        if active and not self._connected:
            self._on_error("Connect and enable the motor before calibrating its range")
            self.panels[device_id].range_cal_btn.setChecked(False)
            return
        self.worker.post(wk.StartRangeCalibration(device_id) if active
                         else wk.StopRangeCalibration(device_id))

    def _on_range_limits_edited(self, device_id: int, pos_min, pos_max) -> None:
        """User applied/cleared limits manually: push to the worker (which
        normalises and echoes back via rangeLimitsChanged for display+persist)."""
        self.worker.post(wk.SetRangeLimits(device_id, pos_min, pos_max))

    @Slot(int, object, object)
    def _on_range_limits_changed(self, device_id: int, pos_min, pos_max) -> None:
        panel = self.panels.get(device_id)
        if panel is not None:
            panel.set_position_limits(pos_min, pos_max)
        self.dashboard.set_position_limits(device_id, pos_min, pos_max)
        self._persist_range(device_id, pos_min, pos_max)

    @Slot(int, object)
    def _on_zero_state(self, device_id: int, info: object) -> None:
        panel = self.panels.get(device_id)
        if panel is not None:
            panel.set_zero_state_display(info)

    @Slot(int, int, float)
    def _on_calibration_changed(self, device_id: int, direction: int, offset: float) -> None:
        panel = self.panels.get(device_id)
        if panel is not None:
            panel.set_calibration_display(direction, offset)
        # e.g. a "set zero here" capture originates in the worker; persist it too.
        self._persist_calibration(device_id, direction, offset)

    @Slot(int, int)
    def _on_motor_id_changed(self, old_id: int, new_id: int) -> None:
        model = self.panels[old_id].model if old_id in self.panels else proto.DEFAULT_MODEL
        self._remove_panel(old_id)
        self._add_motor(new_id, model)
        self._append_log(f"Motor {old_id} is now id {new_id}")

    @Slot(str)
    def _on_error(self, message: str) -> None:
        self._append_log("ERROR: " + message)
        # Rate-limit the modal popup: during a bus failure the worker can emit
        # errors far faster than a user can dismiss dialogs, and each unclosed
        # QMessageBox blocks the one behind it. The log above keeps the full
        # history; the dialog is just the attention-getter.
        now = time.monotonic()
        if now - self._last_error_dialog_time < ERROR_DIALOG_MIN_INTERVAL_S:
            return
        self._last_error_dialog_time = now
        QMessageBox.warning(self, "RobStride Control", message)

    def _append_log(self, message: str) -> None:
        self.log_view.appendPlainText(message)

    # -- E-stop ------------------------------------------------------------------

    def _on_estop(self, checked: bool) -> None:
        self.worker.post(wk.EStop(engage=checked))
        self.estop_btn.setText("E-STOP (engaged)" if checked else "E-STOP")
        if checked:
            for panel in self.panels.values():
                panel.set_enabled_state(False)
            for device_id in self.dashboard.rows:
                self.dashboard.set_enabled_state(device_id, False)

    def _on_disable_all(self) -> None:
        """De-energise every motor and reflect the disabled state in the UI."""
        ids = list(self.panels)
        if not ids:
            return
        for device_id in ids:
            self.worker.post(wk.Disable(device_id))
            self.panels[device_id].set_enabled_state(False)
            if device_id in self.dashboard.rows:
                self.dashboard.set_enabled_state(device_id, False)
        self._append_log(f"Disabled all motors ({len(ids)})")

    def _on_zero_all(self) -> None:
        """Set every motor's zero to its current position, after confirmation."""
        if not self._connected:
            self._on_error("Connect to the bus before zeroing motors")
            return
        ids = list(self.panels)
        if not ids:
            return
        confirm = QMessageBox.question(
            self, "Zero all motors",
            f"Set the zero reference of all {len(ids)} motors to their CURRENT "
            "position?\n\nThis re-defines each motor's zero; it does not move "
            "them. Make sure every motor is at the pose you want to call zero.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if confirm != QMessageBox.Yes:
            return
        for device_id in ids:
            self.worker.post(wk.SetZero(device_id))
            # Mirror per-motor zeroing: arm the one-shot enable confirmation so
            # re-enabling after a bulk zero still warns before energising.
            self._zeroed_since_enable.add(device_id)
        self._append_log(f"Zeroed all motors ({len(ids)})")

    # -- presets -----------------------------------------------------------------

    def _reload_preset_combo(self) -> None:
        self.preset_combo.clear()
        self.preset_combo.addItems(self.presets.names())

    def _current_panel(self) -> MotorPanel | None:
        w = self.tabs.currentWidget()
        return w if isinstance(w, MotorPanel) else None

    def _save_preset(self) -> None:
        panel = self._current_panel()
        if panel is None:
            return
        name, ok = QInputDialog.getText(self, "Save preset", "Preset name:")
        if not ok or not name.strip():
            return
        preset = Preset(
            name=name.strip(),
            device_id=panel.device_id,
            mode=panel.mode_combo.currentData(),
            position=panel.pos_spin.value(),
            # vel_spin displays RPM; presets store the canonical rad/s value.
            velocity=panel.vel_spin.value() * proto.RPM_TO_RAD_S,
            current=panel.cur_spin.value(),
            kp=panel.kp_spin.value(),
            kd=panel.kd_spin.value(),
        )
        self.presets.upsert(preset)
        self.presets.save()
        self._reload_preset_combo()
        self._append_log(f"Saved preset '{preset.name}'")

    def _apply_preset(self) -> None:
        panel = self._current_panel()
        name = self.preset_combo.currentText()
        if panel is None or not name:
            return
        preset = self.presets.get(name)
        if preset is None:
            return
        idx = panel.mode_combo.findData(preset.mode)
        if idx >= 0:
            panel.mode_combo.setCurrentIndex(idx)
        panel.pos_spin.setValue(preset.position)
        panel.vel_spin.setValue(preset.velocity * proto.RAD_S_TO_RPM)
        panel.cur_spin.setValue(preset.current)
        panel.kp_spin.setValue(preset.kp)
        panel.kd_spin.setValue(preset.kd)
        self._append_log(f"Applied preset '{name}' to motor {panel.device_id}")

    def _delete_preset(self) -> None:
        name = self.preset_combo.currentText()
        if name and self.presets.remove(name):
            self.presets.save()
            self._reload_preset_combo()
            self._append_log(f"Deleted preset '{name}'")

    # -- plot refresh / shutdown -------------------------------------------------

    def _refresh_plots(self) -> None:
        for panel in self.panels.values():
            panel.plot.refresh()

    def closeEvent(self, event) -> None:
        try:
            self.sim_dock.shutdown()
            self.worker.stop()
            self.thread.quit()
            self.thread.wait(2000)
            self.datalog.close()
        finally:
            super().closeEvent(event)