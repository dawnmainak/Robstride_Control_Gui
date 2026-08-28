"""Headless motor agent for the Raspberry Pi.

Runs on the Pi with the CAN hardware attached. Reads newline-delimited JSON
commands on **stdin**, drives the motors through the same
:class:`~robstride_gui.worker.ControlWorker` the GUI uses, and writes
newline-delimited JSON telemetry back on **stdout**.

Because it speaks stdin/stdout it needs no listening port and no firewall
rules - the laptop just runs::

    ssh pi@raspberrypi.local 'python3 -u rpi_agent.py'

and pipes JSON in. SSH supplies the transport, the encryption and the auth.
The ``-u`` is REQUIRED: without unbuffered output, Python holds telemetry in a
4 KB buffer and the laptop sees nothing until the pipe closes.

Design notes
------------
The control loop keeps running on the Pi at its usual 100 Hz. The network only
carries setpoints in and telemetry out, at a far lower rate, so a slow or
stuttering link delays a setpoint instead of starving the loop and tripping
every motor's canTimeout.

Anything printed to stdout that is not protocol JSON would corrupt the stream,
so all diagnostics go to stderr (which SSH forwards to the laptop's terminal).
"""

from __future__ import annotations

import json
import sys
import threading
import time
from dataclasses import asdict, is_dataclass

from robstride_gui import worker as wk
from robstride_gui.bus import Motor
from robstride_gui.transport import SerialATTransport, SocketCANTransport

#: Disable every motor if no line arrives from the laptop within this long.
#: The Pi's control loop would otherwise happily hold position forever after
#: the laptop crashes or the link drops - the motor-side canTimeout only covers
#: the Pi itself dying, not the laptop. Comfortably longer than the keepalive
#: interval so ordinary jitter never trips it.
COMMAND_TIMEOUT_S = 1.5

#: Telemetry flush rate. The loop produces status at 100 Hz per motor; sending
#: all of it would be hundreds of messages a second for a UI that cannot use
#: them. One batched message at this rate is plenty and keeps the link quiet.
TELEMETRY_HZ = 20.0


def log(message: str) -> None:
    """Diagnostics to stderr - stdout is reserved for protocol JSON."""
    print(f"[agent] {message}", file=sys.stderr, flush=True)


class Agent:
    def __init__(self) -> None:
        self.worker = wk.ControlWorker()
        self._out_lock = threading.Lock()
        # None until the laptop's FIRST message arrives. The watchdog stays
        # disarmed until then, so SSH startup latency - authenticating, spawning
        # python, importing Qt - cannot burn through the timeout before the link
        # is even warm and E-STOP a rig nobody has touched yet.
        self._last_command: float | None = None
        self._pending: dict[int, dict] = {}
        self._last_flush = 0.0
        self._pending_power: dict[int, dict] = {}
        self._last_power_flush = 0.0
        self._stopping = False
        self._connect_signals()

    # -- outbound ---------------------------------------------------------------

    def send(self, **payload) -> None:
        """Write one JSON line to stdout, atomically.

        Locked because signal callbacks fire on the control thread while the
        watchdog thread may also report - two interleaved writes would produce
        a corrupt line that the laptop cannot parse.
        """
        line = json.dumps(payload, default=str)
        with self._out_lock:
            try:
                sys.stdout.write(line + "\n")
                sys.stdout.flush()
            except (BrokenPipeError, ValueError):
                # The laptop went away mid-shutdown. Signals keep firing while
                # the bus tears down, and each one would otherwise raise and
                # print a traceback over an already-closed pipe. Nothing useful
                # can be reported to a reader that is gone - stay quiet and let
                # the teardown finish disabling the motors.
                self._stopping = True

    def _connect_signals(self) -> None:
        w = self.worker
        w.log.connect(lambda text: self.send(sig="log", text=text))
        w.error.connect(lambda text: self.send(sig="error", text=text))
        w.connectionChanged.connect(
            lambda ok: self.send(sig="connection", connected=bool(ok)))
        w.motorEnabledChanged.connect(
            lambda did, on: self.send(sig="enabled", device_id=did, enabled=bool(on)))
        w.scanFinished.connect(lambda ids: self.send(sig="scan", ids=list(ids)))
        w.statusUpdated.connect(self._on_status)
        w.powerUpdated.connect(self._on_power)
        w.busCollision.connect(lambda ids: self.send(sig="collision", ids=list(ids)))
        w.inventoryReady.connect(lambda items: self.send(sig="inventory", items=list(items)))
        w.calibrationChanged.connect(
            lambda did, d, o: self.send(sig="calibration", device_id=did,
                                        direction=d, offset=o))
        w.rangeLimitsChanged.connect(
            lambda did, lo, hi: self.send(sig="range", device_id=did,
                                          pos_min=lo, pos_max=hi))
        w.motorIdChanged.connect(
            lambda old, new: self.send(sig="motor_id", old_id=old, new_id=new))
        w.zeroStateUpdated.connect(
            lambda did, info: self.send(sig="zero_state", device_id=did,
                                        zero_sta=getattr(info, "zero_sta", None),
                                        mech_offset=getattr(info, "mech_offset", None)))
        w.sweepStopped.connect(lambda did: self.send(sig="sweep_stopped", device_id=did))

    def _on_status(self, device_id: int, status) -> None:
        """Buffer status and flush the whole set at TELEMETRY_HZ.

        Batching keeps one timestamp for the group, so the laptop can line up
        every motor's sample instead of guessing from arrival times.
        """
        self._pending[device_id] = asdict(status) if is_dataclass(status) else {}
        now = time.monotonic()
        if now - self._last_flush < 1.0 / TELEMETRY_HZ:
            return
        self._last_flush = now
        items, self._pending = list(self._pending.values()), {}
        self.send(sig="status", t=round(now, 4), items=items)

    def _on_power(self, device_id: int, power) -> None:
        """Buffer power the same way as status and flush at TELEMETRY_HZ."""
        self._pending_power[device_id] = asdict(power) if is_dataclass(power) else {}
        now = time.monotonic()
        if now - self._last_power_flush < 1.0 / TELEMETRY_HZ:
            return
        self._last_power_flush = now
        items, self._pending_power = list(self._pending_power.values()), {}
        self.send(sig="power", items=items)

    # -- inbound ----------------------------------------------------------------

    def _build_transport(self, msg: dict):
        kind = msg.get("kind", "socketcan")
        if kind == "socketcan":
            return SocketCANTransport(msg.get("channel", "can0"))
        if kind == "serial":
            return SerialATTransport(msg.get("port", "/dev/ttyUSB0"))
        if kind == "fake":
            # Loopback stub for bench-testing the link with no hardware wired.
            from robstride_gui.faketransport import FakeTransport
            return FakeTransport()
        raise ValueError(f"unknown transport kind {kind!r}")

    def handle(self, msg: dict) -> None:
        cmd = msg.get("cmd")
        w = self.worker
        if cmd in (None, "ping"):
            return                      # keepalive only; refreshes the watchdog
        if cmd == "Command":
            self.handle_generic(msg)
            return
        if cmd == "connect":
            bus = msg.get("bus_name", "bus0")
            motors = [Motor(device_id=int(m["device_id"]), model=m.get("model", "rs-03"))
                      for m in msg.get("motors", [])]
            w.post(wk.Connect(transport=self._build_transport(msg),
                              motors=motors, bus_name=bus))
        elif cmd == "add_motor":
            w.post(wk.AddMotor(device_id=int(msg["device_id"]),
                               model=msg.get("model", "rs-03"),
                               bus_name=msg.get("bus_name")))
        elif cmd == "scan":
            w.post(wk.Scan(start=int(msg.get("start", 0)), end=int(msg.get("end", 16))))
        elif cmd == "enable":
            w.post(wk.Enable(int(msg["device_id"])))
        elif cmd == "disable":
            w.post(wk.Disable(int(msg["device_id"])))
        elif cmd == "mode":
            w.post(wk.SetMode(int(msg["device_id"]), int(msg["mode"])))
        elif cmd == "target":
            w.post(wk.SetTarget(device_id=int(msg["device_id"]),
                                position=msg.get("position"),
                                velocity=msg.get("velocity"),
                                current=msg.get("current"),
                                kp=msg.get("kp"), kd=msg.get("kd"),
                                torque_ff=msg.get("torque_ff")))
        elif cmd == "zero":
            w.post(wk.SetZero(int(msg["device_id"])))
        elif cmd == "estop":
            w.post(wk.EStop(engage=bool(msg.get("engage", True))))
        elif cmd == "shutdown":
            self._stopping = True
            w.stop()
        else:
            self.send(sig="error", text=f"unknown command {cmd!r}")

    def handle_generic(self, msg: dict) -> None:
        """Rebuild and post a Command sent by RemoteWorker.

        The GUI serialises commands generically as class name + fields, so every
        command type works without a per-command mapping here and anything added
        later keeps working. Two need rebuilding by hand: Connect's transport is
        a descriptor (the real object is built here, against THIS machine's
        hardware) and SetLimits nests a SafetyLimits dataclass.
        """
        name = msg.get("name", "")
        cls = getattr(wk, name, None)
        if cls is None or not isinstance(cls, type) or not issubclass(cls, wk.Command):
            self.send(sig="error", text=f"unknown command class {name!r}")
            return
        fields = dict(msg.get("fields") or {})
        if name == "Connect":
            fields["transport"] = self._build_transport(msg.get("transport", {}))
            fields["motors"] = [Motor(**m) for m in fields.get("motors", [])]
        elif name == "SetLimits":
            from robstride_gui.safety import SafetyLimits
            fields["limits"] = SafetyLimits(**fields["limits"])
        self.worker.post(cls(**fields))

    def read_stdin(self) -> None:
        """Parse one JSON command per line until the pipe closes.

        A closed pipe means SSH dropped or the laptop exited, so the motors
        must not be left holding: fall through to a shutdown.
        """
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            self._last_command = time.monotonic()
            try:
                msg = json.loads(line)
            except ValueError as e:
                self.send(sig="error", text=f"bad JSON: {e}")
                continue
            try:
                self.handle(msg)
            except Exception as e:
                self.send(sig="error", text=f"{type(e).__name__}: {e}")
        log("stdin closed - shutting down")
        self._stopping = True
        self.worker.stop()

    def watchdog(self) -> None:
        """Engage E-STOP if the laptop goes quiet.

        The link failing is exactly when nobody is watching the rig, so the
        safe response is to stop rather than to keep executing the last
        setpoint. E-STOP is used (not a plain disable) because it also LATCHES:
        motors cannot be re-energised until the laptop explicitly clears it.
        """
        tripped = False
        while not self._stopping:
            time.sleep(0.1)
            if self._last_command is None:
                continue          # disarmed: the laptop has not spoken yet
            silent = time.monotonic() - self._last_command
            if silent > COMMAND_TIMEOUT_S and not tripped:
                tripped = True
                log(f"no command for {silent:.1f}s - engaging E-STOP")
                self.send(sig="error",
                          text=f"link silent {silent:.1f}s - motors stopped")
                self.worker.post(wk.EStop(engage=True))
            elif silent <= COMMAND_TIMEOUT_S:
                tripped = False

    def run(self) -> None:
        threading.Thread(target=self.read_stdin, daemon=True).start()
        threading.Thread(target=self.watchdog, daemon=True).start()
        self.send(sig="ready", timeout_s=COMMAND_TIMEOUT_S)
        log("ready - control loop starting")
        self.worker.run()          # blocks; tears the bus down on exit
        log("control loop stopped")


if __name__ == "__main__":
    Agent().run()