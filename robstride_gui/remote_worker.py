"""Drive a remote Raspberry Pi agent as if it were a local ControlWorker.

:class:`RemoteWorker` presents the exact same surface the GUI already uses -
``post(Command)``, ``run()``, ``stop()`` and the fourteen Qt signals - but
instead of touching CAN hardware it serialises each command to JSON and pipes
it over SSH to ``rpi_agent.py`` running on the Pi. Telemetry comes back the
same way and is re-emitted as the matching Qt signal.

Because ``MainWindow`` only ever calls ``post()`` and connects to signals, it
needs no changes at all: swapping ``ControlWorker()`` for ``RemoteWorker(host)``
is enough to move every motor from the laptop's UI to the Pi's hardware.

Commands are serialised **generically** - class name plus dataclass fields - so
every command type works without a hand-written mapping for each, and any
command added later keeps working. Two need special handling:

* ``Connect`` carries a live ``Transport`` object, which cannot cross a pipe.
  It becomes a descriptor (kind + channel/port) and the Pi builds the real
  transport locally against its own hardware.
* ``SetLimits`` nests a ``SafetyLimits`` dataclass, so it is flattened on the
  way out and rebuilt on the way in.
"""

from __future__ import annotations

import dataclasses
import json
import math
import subprocess
import threading
import time

from PySide6.QtCore import QObject, Signal

from . import worker as wk
from .protocol import MotorStatus
from .transport import SerialATTransport, SocketCANTransport

#: Keepalive period. Must stay well under the agent's COMMAND_TIMEOUT_S or its
#: watchdog reads an idle-but-healthy link as a dead one and E-STOPs the motors.
KEEPALIVE_S = 0.4


def _transport_descriptor(transport) -> dict:
    """Describe a transport in a form the Pi can rebuild from.

    The object itself cannot be sent, and it refers to hardware on the WRONG
    machine anyway - the laptop has no can4. Only the intent travels.
    """
    if isinstance(transport, SocketCANTransport):
        return {"kind": "socketcan", "channel": transport.channel}
    if isinstance(transport, SerialATTransport):
        return {"kind": "serial", "port": transport.port}
    return {"kind": "socketcan", "channel": str(getattr(transport, "channel", "can0"))}


class RemoteWorker(QObject):
    """A ControlWorker-shaped proxy for an agent running on another machine."""

    # Identical to ControlWorker's, so MainWindow's connections are unchanged.
    statusUpdated = Signal(int, object)
    powerUpdated = Signal(int, object)
    connectionChanged = Signal(bool)
    scanFinished = Signal(list)
    busCollision = Signal(list)
    inventoryReady = Signal(list)
    log = Signal(str)
    error = Signal(str)
    motorEnabledChanged = Signal(int, bool)
    calibrationChanged = Signal(int, int, float)
    rangeLimitsChanged = Signal(int, object, object)
    motorIdChanged = Signal(int, int)
    zeroStateUpdated = Signal(int, object)
    sweepStopped = Signal(int)

    def __init__(self, host: str, remote_path: str, parent=None):
        super().__init__(parent)
        self.host = host
        self.remote_path = remote_path
        self.proc: subprocess.Popen | None = None
        self._alive = False
        self._lock = threading.Lock()

    # -- lifecycle ---------------------------------------------------------------

    def run(self) -> None:
        """Launch the agent over SSH and start pumping messages.

        Unlike ControlWorker.run this returns immediately - there is no control
        loop to run here, only a pipe to service - so it must NOT be wired to a
        QThread's started signal.
        """
        # -u is essential: without unbuffered output the agent's telemetry sits
        # in a 4 KB pipe buffer and nothing arrives until the process exits.
        cmd = ["ssh", self.host, f"python3 -u {self.remote_path}"]
        try:
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                         stdout=subprocess.PIPE, text=True, bufsize=1)
        except OSError as e:
            self.error.emit(f"could not start ssh: {e}")
            return
        self._alive = True
        threading.Thread(target=self._read_loop, daemon=True).start()
        threading.Thread(target=self._keepalive_loop, daemon=True).start()
        self.log.emit(f"Remote agent starting on {self.host}")

    def stop(self) -> None:
        """Ask the agent to disable everything, then close the link."""
        if self._alive:
            self._write({"cmd": "shutdown"})
            time.sleep(0.4)
        self._alive = False
        if self.proc is not None:
            self.proc.terminate()

    # -- outbound ----------------------------------------------------------------

    def post(self, command) -> None:
        """Serialise a Command dataclass and send it to the agent."""
        name = type(command).__name__
        payload: dict = {"cmd": "Command", "name": name, "fields": {}}
        for field in dataclasses.fields(command):
            value = getattr(command, field.name)
            if field.name == "transport":
                payload["transport"] = _transport_descriptor(value)
                continue
            if dataclasses.is_dataclass(value):
                payload["fields"][field.name] = dataclasses.asdict(value)
            elif isinstance(value, list):
                payload["fields"][field.name] = [
                    dataclasses.asdict(v) if dataclasses.is_dataclass(v) else v
                    for v in value]
            elif isinstance(value, float) and not math.isfinite(value):
                payload["fields"][field.name] = None   # JSON has no NaN/Infinity
            else:
                payload["fields"][field.name] = value
        self._write(payload)

    def _write(self, payload: dict) -> None:
        if not self._alive or self.proc is None:
            return
        with self._lock:
            try:
                self.proc.stdin.write(json.dumps(payload, default=str) + "\n")
                self.proc.stdin.flush()
            except (BrokenPipeError, ValueError, OSError):
                self._alive = False
                self.connectionChanged.emit(False)
                self.error.emit("Link to the remote agent closed")

    def _keepalive_loop(self) -> None:
        # Ping BEFORE the first sleep. Sleeping first delayed the opening ping
        # by KEEPALIVE_S on top of however long ssh took to authenticate and
        # start python, which was enough for the agent's watchdog to trip and
        # E-STOP before the GUI had done anything at all.
        while self._alive:
            self._write({"cmd": "ping"})
            time.sleep(KEEPALIVE_S)

    # -- inbound -----------------------------------------------------------------

    def _read_loop(self) -> None:
        """Re-emit agent messages as Qt signals.

        Runs on a plain thread, but every receiver lives in the GUI thread, so
        Qt delivers these as queued connections automatically - the widgets are
        only ever touched from the GUI thread.
        """
        assert self.proc is not None
        for line in self.proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            try:
                self._dispatch(msg)
            except Exception as e:
                self.error.emit(f"bad message from agent: {type(e).__name__}: {e}")
        self._alive = False
        self.connectionChanged.emit(False)
        self.log.emit("Remote agent exited")

    def _dispatch(self, msg: dict) -> None:
        sig = msg.get("sig")
        if sig == "status":
            for item in msg.get("items", []):
                self.statusUpdated.emit(int(item["device_id"]), MotorStatus(**item))
        elif sig == "power":
            for item in msg.get("items", []):
                self.powerUpdated.emit(int(item["device_id"]), wk.PowerInfo(**item))
        elif sig == "log":
            self.log.emit(msg.get("text", ""))
        elif sig == "error":
            self.error.emit(msg.get("text", ""))
        elif sig == "connection":
            self.connectionChanged.emit(bool(msg.get("connected")))
        elif sig == "enabled":
            self.motorEnabledChanged.emit(int(msg["device_id"]), bool(msg["enabled"]))
        elif sig == "scan":
            self.scanFinished.emit(list(msg.get("ids", [])))
        elif sig == "collision":
            self.busCollision.emit(list(msg.get("ids", [])))
        elif sig == "inventory":
            self.inventoryReady.emit(list(msg.get("items", [])))
        elif sig == "calibration":
            self.calibrationChanged.emit(int(msg["device_id"]), int(msg["direction"]),
                                        float(msg["offset"]))
        elif sig == "range":
            self.rangeLimitsChanged.emit(int(msg["device_id"]),
                                        msg.get("pos_min"), msg.get("pos_max"))
        elif sig == "motor_id":
            self.motorIdChanged.emit(int(msg["old_id"]), int(msg["new_id"]))
        elif sig == "zero_state":
            self.zeroStateUpdated.emit(
                int(msg["device_id"]),
                wk.ZeroStateInfo(int(msg["device_id"]), msg.get("zero_sta"),
                                 msg.get("mech_offset")))
        elif sig == "sweep_stopped":
            self.sweepStopped.emit(int(msg["device_id"]))
        elif sig == "ready":
            self.log.emit("Remote agent ready")