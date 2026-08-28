#!/usr/bin/env python3
"""Laptop-side control panel: drive motors on a Raspberry Pi over SSH.

Run this on the LAPTOP. It launches ``rpi_agent.py`` on the Pi through SSH and
speaks newline-delimited JSON over the pipe, so no port is opened and no
firewall rule is needed - SSH already provides transport, encryption and auth.

    python publish.py

Everything is in this one file and it imports nothing from ``robstride_gui`` -
only PySide6 - so it runs on a laptop with no CAN hardware and no copy of the
driver stack. The Pi keeps running the 100 Hz control loop next to the motors;
only setpoints go out and telemetry comes back, at a far lower rate, so a
network hiccup delays a setpoint instead of starving the loop.

Safety behaviour worth knowing before using it:

* The position slider snaps to the shaft's ACTUAL angle when telemetry arrives
  after enabling, and no setpoint is sent until it has. Without this, a motor
  resting at 110 degrees would slam to whatever the slider happened to show.
* A keepalive ping goes out every 400 ms. If the agent hears nothing for 1.5 s
  it E-STOPs every motor, so a crashed laptop or a dropped link stops the rig.
* Closing the window disables the motor and shuts the agent down cleanly.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import threading

from PySide6.QtCore import Qt, QObject, QTimer, Signal
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QGridLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QMainWindow, QPlainTextEdit, QPushButton,
    QSlider, QSpinBox, QVBoxLayout, QWidget,
)

#: Ping period. Must stay well under the agent's COMMAND_TIMEOUT_S (1.5 s) or
#: its watchdog reads an idle-but-healthy link as a dead one and stops the
#: motors. A third of the window absorbs ordinary network jitter.
KEEPALIVE_MS = 400

#: How often the current target is re-sent. The Pi's loop runs at 100 Hz on its
#: own; this only has to be smooth to a human eye.
SETPOINT_MS = 50

#: Run modes, paired with the raw value the motor expects. The order matches
#: RunMode in the driver: 0 MIT, 1 profiled position, 2 velocity, 3 current.
#: Position is the default because it is the only mode that both holds a pose
#: and refuses to run away if a setpoint stops arriving.
MODES = [("MIT", 0), ("Position", 1), ("Velocity", 2), ("Current", 3)]


class Link(QObject):
    """A JSON-lines pipe to ``rpi_agent.py``, running wherever SSH puts it."""

        # Qt signals rather than direct calls: the reader runs on a plain thread,
        # and emitting lets Qt hand the payload to the GUI thread as a queued
        # connection. Widgets are then only ever touched from one thread.
    message = Signal(dict)
    closed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.proc: subprocess.Popen | None = None
                # stdin is written from the GUI thread (button presses), the keepalive
                # timer and the setpoint timer. Two interleaved writes would produce a
                # torn JSON line the agent cannot parse, so every write takes the lock.
        self._lock = threading.Lock()
        self.alive = False

    def start(self, host: str, remote_path: str) -> bool:
        # -u is essential: without unbuffered output the agent's telemetry sits
        # in a 4 KB pipe buffer and nothing arrives until the process exits.
        cmd = ["ssh", host, f"python3 -u {remote_path}"]
        try:
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                         stdout=subprocess.PIPE,
                                         text=True, bufsize=1)
        except OSError:
            return False
                # daemon=True so a reader blocked on a dead pipe cannot keep the
                # process alive after the window closes.
        self.alive = True
        threading.Thread(target=self._read, daemon=True).start()
        return True

    def _read(self) -> None:
        """Re-emit each agent line as a Qt signal.

        Runs on a plain thread, but the receivers are widgets in the GUI
        thread, so Qt delivers these as queued connections - the UI is only
        ever touched from the GUI thread.
        """
        assert self.proc is not None
        for line in self.proc.stdout:
            line = line.strip()
            if line:
                try:
                    self.message.emit(json.loads(line))
                except ValueError:
                    pass
                # Falling out of the for loop means EOF: ssh exited, the agent died, or
                # the link dropped. All three mean the same thing to the UI.
        self.alive = False
        self.closed.emit()

    def send(self, **msg) -> None:
        if not self.alive or self.proc is None:
            return
        with self._lock:
            try:
                self.proc.stdin.write(json.dumps(msg) + "\n")
                self.proc.stdin.flush()
                        # A write can fail long before the reader notices EOF, so the
                        # failure is reported here too rather than waiting for the pipe to
                        # close. Reporting twice is harmless; missing it is not.
            except (BrokenPipeError, ValueError, OSError):
                self.alive = False
                self.closed.emit()

    def stop(self) -> None:
        """Shut the agent down cleanly so the motor is disabled promptly.

        Killing SSH alone would also work - the agent stops when its stdin
        closes - but an explicit shutdown does not wait on the watchdog.
        """
        if self.alive:
            self.send(cmd="shutdown")
        self.alive = False
        if self.proc is not None:
            QTimer.singleShot(400, self.proc.terminate)


class Panel(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RobStride remote publisher")
        self.resize(720, 640)
        self.link = Link(self)
        self.link.message.connect(self._on_message)
        self.link.closed.connect(self._on_closed)

                # None until the first status frame. It doubles as the "is anything
                # actually answering at this id" check in _toggle_enable.
        self.position: float | None = None   # newest angle from telemetry, rad
        self.enabled = False
        self._synced = False                 # slider matched to the shaft yet?
        self._sweep_centre = 0.0
        self._sweep_t = 0.0

        self._build()

                # Two independent timers. The keepalive must keep running even when no
                # motor is enabled - the agent's watchdog does not care why the link
                # went quiet, only that it did.
        self.keepalive = QTimer(self)
        self.keepalive.timeout.connect(lambda: self.link.send(cmd="ping"))
        self.setpoint_timer = QTimer(self)
        self.setpoint_timer.timeout.connect(self._tick_setpoint)

    # -- layout ------------------------------------------------------------------

    def _build(self) -> None:
        root = QWidget()
        lay = QVBoxLayout(root)

        conn = QGroupBox("Raspberry Pi")
        g = QGridLayout(conn)
        self.host_edit = QLineEdit("usvz@rs04pi.local")
        self.path_edit = QLineEdit("~/Desktop/Robstride_Control_Gui/rpi_agent.py")
        self.kind_combo = QComboBox()
                # The transport is built on the PI, from this description. Naming it
                # here rather than constructing an object means the laptop never needs
                # CAN hardware or a driver for a bus it cannot see.
        self.kind_combo.addItems(["socketcan", "serial"])
        self.chan_edit = QLineEdit("can4")
        self.link_btn = QPushButton("Start agent")
        self.link_btn.clicked.connect(self._toggle_link)
        self.conn_btn = QPushButton("Connect bus")
        self.conn_btn.setEnabled(False)
        self.conn_btn.clicked.connect(self._connect_bus)
        g.addWidget(QLabel("SSH host"), 0, 0)
        g.addWidget(self.host_edit, 0, 1, 1, 3)
        g.addWidget(QLabel("Agent path"), 1, 0)
        g.addWidget(self.path_edit, 1, 1, 1, 3)
        g.addWidget(QLabel("Transport"), 2, 0)
        g.addWidget(self.kind_combo, 2, 1)
        g.addWidget(QLabel("Channel / port"), 2, 2)
        g.addWidget(self.chan_edit, 2, 3)
        g.addWidget(self.link_btn, 3, 1)
        g.addWidget(self.conn_btn, 3, 3)
        lay.addWidget(conn)

        motor = QGroupBox("Motor")
        m = QHBoxLayout(motor)
        self.id_spin = QSpinBox()
        self.id_spin.setRange(0, 127)
        self.id_spin.setValue(3)
        self.model_combo = QComboBox()
        self.model_combo.addItems(["rs-00", "rs-01", "rs-02", "rs-03",
                                   "rs-04", "rs-05", "rs-06"])
        self.model_combo.setCurrentText("rs-03")
        self.mode_combo = QComboBox()
        for name, value in MODES:
            self.mode_combo.addItem(name, value)
        self.mode_combo.setCurrentIndex(1)                  # Position
        self.mode_combo.currentIndexChanged.connect(self._send_mode)
        self.enable_btn = QPushButton("Enable")
        self.enable_btn.setCheckable(True)
                # Disabled until a "connection" signal arrives. Enabling a motor before
                # the bus is open would silently do nothing and look like a dead motor.
        self.enable_btn.setEnabled(False)
        self.enable_btn.clicked.connect(self._toggle_enable)
        m.addWidget(QLabel("CAN id")); m.addWidget(self.id_spin)
        m.addWidget(QLabel("Model")); m.addWidget(self.model_combo)
        m.addWidget(QLabel("Mode")); m.addWidget(self.mode_combo)
        m.addStretch(1); m.addWidget(self.enable_btn)
        lay.addWidget(motor)

        cmd = QGroupBox("Command")
        c = QGridLayout(cmd)
        self.pos_slider = QSlider(Qt.Horizontal)
                # Degrees in the UI, radians on the wire. The conversion happens only
                # at _tick_setpoint and _sync_slider, so there is no chance of a value
                # in the wrong unit being sent.
        self.pos_slider.setRange(-180, 180)
        self.pos_slider.valueChanged.connect(self._slider_moved)
        self.pos_spin = QDoubleSpinBox()
        self.pos_spin.setRange(-180.0, 180.0)
        self.pos_spin.setSuffix(" deg")
        self.pos_spin.valueChanged.connect(
            lambda v: self.pos_slider.setValue(int(v)))
        self.sweep_check = QCheckBox("Sweep")
        self.sweep_check.toggled.connect(self._toggle_sweep)
        self.amp_spin = QDoubleSpinBox()
        self.amp_spin.setRange(1.0, 180.0)
        self.amp_spin.setValue(20.0)
        self.amp_spin.setSuffix(" deg")
        self.period_spin = QDoubleSpinBox()
        self.period_spin.setRange(0.5, 30.0)
        self.period_spin.setValue(4.0)
        self.period_spin.setSuffix(" s")
        c.addWidget(QLabel("Target"), 0, 0)
        c.addWidget(self.pos_slider, 0, 1, 1, 3)
        c.addWidget(self.pos_spin, 0, 4)
        c.addWidget(self.sweep_check, 1, 0)
        c.addWidget(QLabel("amplitude"), 1, 1)
        c.addWidget(self.amp_spin, 1, 2)
        c.addWidget(QLabel("period"), 1, 3)
        c.addWidget(self.period_spin, 1, 4)
        lay.addWidget(cmd)

        self.readout = QLabel("no telemetry")
        self.readout.setStyleSheet("font-family:monospace; padding:6px;")
        lay.addWidget(self.readout)

                # Deliberately oversized and red, and NOT inside the Command group: it
                # has to be findable without reading anything.
        stop = QPushButton("E-STOP")
        stop.setMinimumHeight(44)
        stop.setStyleSheet(
            "QPushButton{background:#b71c1c;color:white;font-weight:bold;}")
        stop.clicked.connect(self._estop)
        lay.addWidget(stop)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(500)
        lay.addWidget(self.log_view, 1)

                # Everything that can move a motor starts disabled and is re-enabled
                # only once the motor is enabled AND its position is known.
        self.setCentralWidget(root)
        self._set_command_enabled(False)

    def _set_command_enabled(self, on: bool) -> None:
        for w in (self.pos_slider, self.pos_spin, self.sweep_check,
                  self.amp_spin, self.period_spin):
            w.setEnabled(on)

    def log(self, text: str) -> None:
        self.log_view.appendPlainText(text)

    # -- link --------------------------------------------------------------------

    def _toggle_link(self) -> None:
        if self.link.alive:
            self._shutdown()
            return
        if not self.link.start(self.host_edit.text().strip(),
                               self.path_edit.text().strip()):
            self.log("ERROR: could not launch ssh - is it installed and on PATH?")
            return
                # The keepalive starts with the link, not with the motor. The agent
                # arms its watchdog on the first message it receives, so pings have to
                # be flowing before anything else is attempted.
        self.keepalive.start(KEEPALIVE_MS)
        self.link_btn.setText("Stop agent")
        self.conn_btn.setEnabled(True)
        self.log(f"Agent starting on {self.host_edit.text().strip()}")

    def _connect_bus(self) -> None:
        kind = self.kind_combo.currentText()
        target = self.chan_edit.text().strip()
                # channel and port are both sent because the agent picks whichever its
                # transport kind needs; the unused one is ignored. Simpler than making
                # the UI know which field matters for which transport.
        self.link.send(cmd="connect", kind=kind,
                       channel=target, port=target, bus_name=target,
                       motors=[{"device_id": self.id_spin.value(),
                                "model": self.model_combo.currentText()}])
        self.log(f"-> connect {kind} {target} motor {self.id_spin.value()} "
                 f"({self.model_combo.currentText()})")

    def _shutdown(self) -> None:
        # Disable BEFORE closing the link. Once the pipe is gone the only thing
        # stopping the motor is the agent's watchdog, which takes 1.5 s - a
        # long time for a moving arm.
        if self.enabled:
            self.link.send(cmd="disable", device_id=self.id_spin.value())
        self.keepalive.stop()
        self.setpoint_timer.stop()
        self.link.stop()

    def _on_closed(self) -> None:
        self.keepalive.stop()
        self.setpoint_timer.stop()
        self.enabled = False
        self._synced = False
        self.enable_btn.setChecked(False)
        self.enable_btn.setEnabled(False)
        self.enable_btn.setText("Enable")
        self.conn_btn.setEnabled(False)
        self._set_command_enabled(False)
        self.link_btn.setText("Start agent")
        self.log("Agent link closed")

    # -- motor -------------------------------------------------------------------

    def _send_mode(self) -> None:
        if self.link.alive:
            self.link.send(cmd="mode", device_id=self.id_spin.value(),
                           mode=self.mode_combo.currentData())

    def _toggle_enable(self, on: bool) -> None:
        device_id = self.id_spin.value()
        if on:
            if self.position is None:
                # No telemetry means nothing is answering at this id. Enabling
                # blind risks energising a motor that is not the one intended.
                self.log("ERROR: no telemetry yet - check the CAN id and that "
                         "the bus is connected")
                self.enable_btn.setChecked(False)
                return
                        # Mode first, then enable. The motor keeps its previous run-mode
                        # across power cycles, so enabling without asserting the mode can
                        # energise it in MIT and ignore every position setpoint sent after.
            self._send_mode()
            self.link.send(cmd="enable", device_id=device_id)
            self.enabled = True
                        # Cleared on EVERY enable, not just the first: the shaft may have
                        # been moved by hand while it was disabled.
            self._synced = False        # re-match the slider to the live angle
            self._set_command_enabled(True)
            self.enable_btn.setText("Disable")
            self.setpoint_timer.start(SETPOINT_MS)
        else:
                        # Stop the sweep on disable so a later enable holds position
                        # instead of resuming a trajectory the operator has forgotten
                        # about.
            self.setpoint_timer.stop()
            self.sweep_check.setChecked(False)
            self.link.send(cmd="disable", device_id=device_id)
            self.enabled = False
            self._set_command_enabled(False)
            self.enable_btn.setText("Enable")

    def _slider_moved(self, degrees: int) -> None:
        # Guarded to break the slider <-> spinbox feedback loop: each updates
        # the other, so without a deadband they would ping-pong on every move.
        if abs(self.pos_spin.value() - degrees) > 0.5:
            self.pos_spin.setValue(float(degrees))

    def _toggle_sweep(self, on: bool) -> None:
        self.pos_slider.setEnabled(not on)
        self.pos_spin.setEnabled(not on)
        if on:
                        # The sweep is centred on wherever the slider is when it starts, so
                        # turning it on never moves the arm by itself.
            self._sweep_centre = math.radians(self.pos_slider.value())
            self._sweep_t = 0.0
            self.log(f"sweep +/-{self.amp_spin.value():.0f} deg around "
                     f"{self.pos_slider.value()} deg")

    def _tick_setpoint(self) -> None:
        """Send the current target every SETPOINT_MS.

        Held back until ``_synced`` - the slider has been matched to the
        shaft's real angle - so no setpoint can go out before we know where
        the motor actually is.
        """
        if not (self.enabled and self.link.alive and self._synced):
            return
        if self.sweep_check.isChecked():
                        # Phase is accumulated from the timer rather than read off a wall
                        # clock, so a stalled UI slows the sweep instead of making it jump.
            self._sweep_t += SETPOINT_MS / 1000.0
            phase = self._sweep_t / max(self.period_spin.value(), 0.5)
            target = self._sweep_centre + \
                math.radians(self.amp_spin.value()) * math.sin(2 * math.pi * phase)
        else:
            target = math.radians(self.pos_slider.value())
        self.link.send(cmd="target", device_id=self.id_spin.value(),
                       position=round(target, 4))

    def _estop(self) -> None:
        self.setpoint_timer.stop()
        self.sweep_check.setChecked(False)
                # E-STOP is sent even if the local state says nothing is enabled: the
                # Pi's view is the one that matters, and this is the one control that
                # must never be a no-op because of a stale local flag.
        self.link.send(cmd="estop", engage=True)
        self.enabled = False
        self.enable_btn.setChecked(False)
        self.enable_btn.setText("Enable")
        self._set_command_enabled(False)
        self.log("E-STOP sent")

    # -- inbound -----------------------------------------------------------------

    def _on_message(self, msg: dict) -> None:
        sig = msg.get("sig")
        if sig == "status":
            for item in msg.get("items", []):
                                # A status batch carries every motor on the bus; this panel
                                # drives one, so the rest are skipped rather than displayed.
                if int(item["device_id"]) != self.id_spin.value():
                    continue
                self.position = float(item["position"])
                self.readout.setText(
                    f"pos {math.degrees(self.position):+8.2f} deg     "
                    f"vel {item['velocity']:+7.2f} rad/s     "
                    f"torque {item['torque']:+6.2f} Nm     "
                    f"{item['temperature']:.0f} C")
                if self.enabled and not self._synced:
                    self._sync_slider(self.position)
        elif sig == "error":
            self.log("ERROR: " + msg.get("text", ""))
        elif sig == "connection":
                        # Connection state is authoritative from the Pi. A dropped bus
                        # re-locks Enable without the operator having to notice.
            ok = bool(msg.get("connected"))
            self.enable_btn.setEnabled(ok)
            self.log("Bus connected" if ok else "Bus disconnected")
        elif sig == "enabled":
            if int(msg.get("device_id", -1)) == self.id_spin.value() \
                    and not msg.get("enabled") and self.enabled:
                # The Pi disabled it behind our back (range cutout, E-STOP,
                # link watchdog). Reflect that rather than show a phantom
                # Disable button over a motor that is already off.
                self.enabled = False
                self.setpoint_timer.stop()
                self.enable_btn.setChecked(False)
                self.enable_btn.setText("Enable")
                self._set_command_enabled(False)
                self.log("Motor was disabled by the Pi")
        elif sig in ("log", "ready"):
            self.log(msg.get("text", "agent ready"))

    def _sync_slider(self, radians: float) -> None:
        """Match the slider to the shaft's real angle before commanding it.

        Signals are blocked so this does not itself look like a user move.
        Until it runs, ``_tick_setpoint`` sends nothing - otherwise a motor
        resting at 110 degrees would be commanded to whatever the slider
        happened to show, a full-speed slam the moment it was enabled.
        """
                # Clamped to the slider's own range. A shaft parked outside +/-180 deg
                # would otherwise be silently truncated by setValue, and the panel
                # would then command a position that is not where the motor is.
        degrees = max(-180.0, min(180.0, math.degrees(radians)))
        for w in (self.pos_slider, self.pos_spin):
            w.blockSignals(True)
        self.pos_slider.setValue(int(round(degrees)))
        self.pos_spin.setValue(degrees)
        for w in (self.pos_slider, self.pos_spin):
            w.blockSignals(False)
        self._sweep_centre = math.radians(self.pos_slider.value())
        self._synced = True
        self.log(f"slider matched to shaft at {degrees:+.1f} deg - safe to move")

    def closeEvent(self, event) -> None:
        # Closing the window is a stop request. Without this the motor would
        # keep holding its last setpoint until the agent's watchdog fired.
        self._shutdown()
        super().closeEvent(event)


def main() -> int:
    app = QApplication(sys.argv)
    panel = Panel()
    panel.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())