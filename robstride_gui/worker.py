"""Background control worker.

The GUI thread must never block on serial / CAN IO, so all bus interaction
happens here on a dedicated :class:`QThread`. The UI talks to the worker by
pushing :class:`Command` objects onto a thread-safe queue; the worker emits Qt
signals back with connection state, per-motor status, and log/error messages.

Per-motor *targets* (mode + setpoints + gains + enabled flag) live in a small
table. Each loop iteration the worker drains pending commands, then for every
enabled motor issues the appropriate frame for its current run-mode and emits
the decoded feedback.
"""

from __future__ import annotations

import math
import queue
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, replace
from typing import Optional

from PySide6.QtCore import QObject, Signal

from . import protocol as proto
from .bus import BusConfig, Motor, RobstrideBus
from .bus_router import BusRouter
from .protocol import MotorMode, MotorStatus, ParameterType, RunMode
from .safety import Calibration, SafetyLimits, SafetyState
from .transport import Transport, TransportError

# Run modes that hold a position setpoint. Enabling one must first seed the
# setpoint to the shaft's current position, or the motor snaps to a stale
# target (see ``ControlWorker._seed_hold_position``).
_POSITION_MODES = (RunMode.POSITION_PP, RunMode.POSITION_CSP)

# How far live feedback may exceed the calibrated range before the safety cutout
# disables the motor. A few degrees of slack absorbs position-hold jitter and
# brief overshoot without nuisance trips, while still stopping a runaway before
# it drives an attached part through a hard stop.
RANGE_TRIP_MARGIN_RAD = math.radians(5.0)


# --- command objects pushed from the UI -----------------------------------------


@dataclass
class Command:
    """Base class for UI -> worker requests."""


@dataclass
class Connect(Command):
    transport: Transport
    motors: list[Motor]
    #: Name for this adapter's bus, e.g. "can0". Posting Connect again with a
    #: DIFFERENT name adds a second bus alongside the first instead of
    #: replacing it, which is how a multi-adapter setup is built up.
    bus_name: str = "bus0"


@dataclass
class AddMotor(Command):
    """Register a motor added/discovered after Connect so it is polled and driveable.

    Connect seeds the bus from the tabs open at connect time; a motor added later
    (e.g. one surfaced by Detect) needs the same registration or it has a tab but
    never gets serviced. Idempotent: re-registering a known id is harmless.
    """

    device_id: int
    model: str = proto.DEFAULT_MODEL
    #: Which bus owns this motor. None is fine while only one bus exists.
    bus_name: Optional[str] = None


@dataclass
class Disconnect(Command):
    pass


@dataclass
class Scan(Command):
    start: int = 1
    end: int = 16


@dataclass
class Inventory(Command):
    """List responding CAN ids and each motor's unique MCU id, for the device panel."""

    start: int = 1
    end: int = 16


@dataclass
class Enable(Command):
    device_id: int


@dataclass
class Disable(Command):
    device_id: int


@dataclass
class SetZero(Command):
    device_id: int


@dataclass
class SetMode(Command):
    device_id: int
    mode: int


@dataclass
class SetTarget(Command):
    """Update one or more setpoints for a motor (None = leave unchanged)."""

    device_id: int
    position: Optional[float] = None
    velocity: Optional[float] = None
    current: Optional[float] = None
    kp: Optional[float] = None
    kd: Optional[float] = None
    torque_ff: Optional[float] = None


@dataclass
class SetSweep(Command):
    """Start/stop a continuous position sweep between two angles.

    While enabled the worker overrides the motor's position setpoint every
    control cycle with a smooth sine that oscillates between ``from_pos`` and
    ``to_pos`` once per ``period`` seconds - a scripted repeating motion for
    soak/simulation. Only takes effect in a position run-mode; angles are in
    radians (user frame), the same units as a normal position setpoint.
    """

    device_id: int
    enabled: bool
    from_pos: float = 0.0
    to_pos: float = 0.0
    period: float = 2.0


@dataclass
class EStop(Command):
    engage: bool


@dataclass
class SetLimits(Command):
    limits: SafetyLimits


@dataclass
class SetCalibration(Command):
    device_id: int
    direction: int = 1     # +1 normal, -1 inverted
    offset: float = 0.0    # rad, raw frame


@dataclass
class CaptureZero(Command):
    """Set this motor's zero-offset to its current measured position."""

    device_id: int


@dataclass
class SetMotorId(Command):
    """Reassign a motor's CAN id (only one motor should be on the bus)."""

    current_id: int
    new_id: int


@dataclass
class ReadZeroState(Command):
    """Read the motor's own persisted zero markers, to show if it remembers zero."""

    device_id: int


@dataclass
class SetRangeLimits(Command):
    """Set (or clear) a motor's calibrated travel range in the user frame.

    ``pos_min``/``pos_max`` are radians; ``None`` clears that bound. The worker
    clamps every position-mode setpoint into this range before it reaches the
    bus, so the motor cannot be driven past the calibrated ends.
    """

    device_id: int
    pos_min: Optional[float] = None
    pos_max: Optional[float] = None


@dataclass
class StartRangeCalibration(Command):
    """Enter range-calibration for a motor (LeRobot-style).

    While active the worker records the min/max of the live position. With
    ``make_limp`` the motor is switched to MIT with zero gains/torque so it can
    be moved by hand; the operator may equally jog it - either way the extremes
    reached are captured. The prior run-mode/gains are restored on stop.
    """

    device_id: int
    make_limp: bool = True


@dataclass
class StopRangeCalibration(Command):
    """Finish range-calibration: commit the captured min/max as the range and
    restore the pre-calibration run-mode and gains."""

    device_id: int


# --- board power telemetry ------------------------------------------------------

#: Read VBUS/Iq once every Nth control loop. The control loop runs at 100 Hz; a
#: divisor of 4 polls the board power registers at ~25 Hz. This is set for
#: power diagnostics - fast enough to catch the voltage dip / current spike when
#: a motor is enabled (a sub-100 ms transient the old ~5 Hz rate stepped over),
#: while not so fast it floods the bus with READ_PARAMETER traffic that competes
#: with setpoint writes. Raise back to ~20 (5 Hz) once the power question is
#: settled to keep the round-trips off the hot path.
POWER_READ_DIVISOR: int = 4

#: Log a VBUS line only when the bus voltage moves at least this many volts from
#: the last value logged for that motor. At the ~25 Hz diagnostic read rate this
#: keeps a steady rail silent and turns a sag+recovery into a couple of lines,
#: instead of flooding the log with every sample. Set well above the ~0.3 V read
#: noise so ordinary jitter does not trip it. Lower it to catch a subtler dip.
VBUS_LOG_DELTA_V: float = 1.0

#: Emit a full per-motor telemetry line to the log every Nth control loop. At the
#: 100 Hz loop rate a divisor of 50 gives ~2 Hz per motor - slow enough to read
#: and copy from the log dock, fast enough to catch a hold dropping when another
#: motor is enabled. This is the "detailed log" for diagnosing by eye; the opt-in
#: telemetry file still keeps the full-rate record. Raise to silence it.
VERBOSE_LOG_DIVISOR: int = 50


# --- communication watchdog ------------------------------------------------------

#: Consecutive transport-level failures in the control loop before the worker
#: declares the link dead and disconnects. At the 100 Hz loop rate this is
#: ~0.1 s of solid failure - long enough to ride out a one-off USB glitch (the
#: transport already retries transient CH340 hiccups internally), short enough
#: that a yanked adapter does not leave enabled motors unsupervised for long.
COMM_FAILURE_LIMIT: int = 10

#: Longest the coordinator waits for one bus to finish its slice of a tick.
#: Generous next to the ~10 ms tick because a bus with several motors makes
#: several blocking round-trips, each able to wait BusConfig.response_timeout
#: (0.15 s) for a reply. Exceeding it means that adapter is wedged: the tick is
#: abandoned for that bus and the others carry on, rather than the whole robot
#: blocking on one bad cable. The stranded motors stop themselves via canTimeout.
BUS_TICK_TIMEOUT_S: float = 0.5

#: Any command whose handler blocks the single-threaded control loop longer than
#: this (milliseconds) is logged with its measured duration. The worker services
#: motors and applies commands on one thread, so this block is dead time for every
#: *other* enabled motor - no hold frame goes out while it runs. A slow
#: enable/mode-switch here is the direct measure of the starvation window that can
#: let a bystander motor trip its firmware watchdog and go limp (the "M6 drops
#: when M5 mode-switches" bug). A normal per-tick setpoint write is well under
#: 1 ms; lower this to see every blocking op.
SLOW_COMMAND_LOG_MS: float = 20.0

#: Max commands ``_drain_commands`` processes in one loop iteration. It used to
#: drain the queue to empty unconditionally, so any burst - regardless of what
#: filled the queue - could starve ``_service_motors`` (and every enabled
#: motor's setpoint/keepalive frame) for the whole burst's duration in one go.
#: Capping it means a burst spreads across multiple ~10 ms ticks instead of
#: blocking one tick indefinitely; leftover commands simply wait in the queue
#: for the next iteration; a handler that raises still lets later commands in
#: the same batch run (see ``_drain_commands``).
MAX_COMMANDS_PER_DRAIN: int = 50

#: Motor-side watchdog: the RAW value written to the ``canTimeout`` register
#: (0x7028) on every Enable. A host-side watchdog cannot help when the *host*
#: dies (process crash, cable pull): in velocity/current mode the motor would
#: keep executing its last setpoint forever. With canTimeout armed the motor
#: stops itself when no frame arrives within the window. Set the worker's
#: ``motor_can_timeout_raw`` to 0 to skip the write.
#:
#: THE UNIT IS NOT MILLISECONDS. It was long *assumed* to be ms, and 1000 was
#: written as "1 s". Field capture (frames.log, 2026-07-10) disproves that:
#: with canTimeout=1000 armed *and read back as 1000*, every motor still tripped
#: to standby at a servicing gap of ~60 ms (median 66 ms across all six motors,
#: no fault bit). So 1000 raw == ~60 ms, i.e. ~0.06 ms per count - the register
#: never widened the window at all, and the keepalive interleaving alone
#: (``_feed_other_motors``) could not hold every bystander's gap under that
#: ~60 ms firmware default once 5-6 motors were enabled at once. That is the
#: real cause of the "auto-disable" cascades: bystanders trip during each
#: enable handshake, the hold-recovery re-enable starves the next motor, and
#: after MAX_HOLD_RECOVERY_ATTEMPTS the worker gives up and leaves them off.
#:
#: 5000 raw targeted ~300 ms at the inferred ~0.06 ms/count, comfortably above
#: the 92 ms worst-case handshake gap seen on the wire - but that only covers
#: starvation from a blocking handshake. :meth:`ControlWorker.measure_can_timeout`
#: run against motor 3 (2026-07-23) measured the REAL scale directly:
#: raw=5000 held through 400 ms of silence and dropped between 400-800 ms
#: (~600 ms, i.e. ~0.12 ms/count - 2x the earlier inferred figure). Separately,
#: reproducing the worker on its real QThread (not a single-threaded test
#: harness) showed hold-drops even at *idle*, with no command burst at all -
#: CPython GIL scheduling contention between the Qt main thread and this
#: control loop's thread can itself produce gaps wide enough to trip even a
#: several-hundred-ms window. 25000 raw (~3 s at the measured 0.12 ms/count,
#: via :func:`recommend_can_timeout_raw`\ (5000, 600.0, 3000.0)) trades a longer
#: host-death stop time for real headroom against that jitter - acceptable on a
#: bench setup with an operator present. VERIFY per hardware with a fresh
#: :meth:`measure_can_timeout` point if this board/firmware differs.
MOTOR_CAN_TIMEOUT_RAW: int = 25000

#: Target watchdog window, in milliseconds, that :data:`MOTOR_CAN_TIMEOUT_RAW`
#: is chosen to hit. Widened from 300 ms (see :data:`MOTOR_CAN_TIMEOUT_RAW`) to
#: clear observed GIL-jitter gaps, not just handshake starvation, while still
#: stopping a genuine runaway within a few seconds of true host death. Feed it
#: to :func:`recommend_can_timeout_raw` with a measured calibration point to get
#: the raw value for a given board.
MOTOR_CAN_TIMEOUT_TARGET_MS: float = 3000.0


def recommend_can_timeout_raw(armed_raw: int, observed_stop_ms: float,
                              target_ms: float = MOTOR_CAN_TIMEOUT_TARGET_MS) -> int:
    """Convert one starve-test data point into the raw canTimeout for ``target_ms``.

    The canTimeout register (0x7028) unit is not documented and is *not*
    milliseconds on this firmware (see :data:`MOTOR_CAN_TIMEOUT_RAW`). Given one
    measured point - you armed ``armed_raw`` and the motor actually stopped
    ``observed_stop_ms`` after the last frame - the count-to-ms scale is
    ``observed_stop_ms / armed_raw`` (assumed linear through the origin, which
    matches a simple down-counter). This inverts it to the raw value that lands
    on ``target_ms``:

        raw = round(target_ms * armed_raw / observed_stop_ms)

    Both inputs must be positive; a zero/negative ``observed_stop_ms`` means the
    starve test never saw a stop (the watchdog is disabled or the value did not
    arm), so there is nothing to scale and we raise rather than return a bogus
    number. The result is clamped to at least 1 (0 would disable the watchdog).
    """
    if armed_raw <= 0:
        raise ValueError(f"armed_raw must be positive, got {armed_raw}")
    if observed_stop_ms <= 0:
        raise ValueError(
            "observed_stop_ms must be positive; a non-positive value means the "
            "starve test saw no stop (watchdog disabled or value did not arm)")
    ms_per_count = observed_stop_ms / armed_raw
    return max(1, round(target_ms / ms_per_count))

#: Consecutive auto-re-enable attempts before the worker gives up on a motor that
#: keeps reverting to standby. A motor that trips its firmware CAN-watchdog (e.g.
#: momentarily starved during another motor's enable) recovers on the first
#: attempt - and the re-enable no longer starves the others, because it feeds them
#: via ``_feed_other_motors`` (fix #1), which is what makes auto-recovery safe
#: instead of a ping-pong. If a motor still reports standby after this many
#: re-enables the cause is persistent (wiring, a too-short canTimeout, a genuine
#: fault), so the worker stops the retry storm, leaves it disabled, and surfaces
#: it to the operator rather than energising it every control tick forever.
MAX_HOLD_RECOVERY_ATTEMPTS: int = 3

#: Consecutive RESET reads required before a hold-drop is treated as real.
#: ``_await`` matches replies by comm-type + device id only (no sequence
#: number), so a single stray/late frame could in principle be misread as a
#: momentary drop. Requiring back-to-back RESET reads, spaced by an actual
#: control tick, absorbs a one-off misread without spending a recovery
#: attempt on it. Raised from 2 to 5 (2026-07-23): reproducing the worker on
#: its real QThread showed CPython GIL scheduling jitter between the Qt main
#: thread and this control loop can span more than 2 consecutive ~10 ms
#: ticks even with no command burst - 2 was not always enough to absorb it.
#: 5 ticks (~50 ms) is still tiny next to the widened
#: :data:`MOTOR_CAN_TIMEOUT_RAW` (~3 s) budget, so a genuine drop is still
#: caught quickly.
HOLD_DROP_DEBOUNCE: int = 5


@dataclass(frozen=True)
class PowerInfo:
    """Electrical telemetry read from the motor control board."""

    device_id: int
    vbus: float       # bus voltage, V
    iq: float         # filtered q-axis current, A
    power: float      # estimated input power VBUS*Iq, W


@dataclass(frozen=True)
class ZeroStateInfo:
    """The motor's own persisted zero markers, read back from its registers.

    ``mech_offset`` is the stored mechanical zero (``mechOffset``, 0x2005) the
    motor keeps in flash; ``zero_sta`` is its zero-state flag (0x7029). Together
    they show whether the motor remembers its absolute zero. Either field is
    ``None`` if that register did not answer.
    """

    device_id: int
    zero_sta: Optional[int]
    mech_offset: Optional[float]


# --- per-motor live target ------------------------------------------------------


@dataclass
class MotorTarget:
    mode: int = RunMode.POSITION_PP
    position: float = 0.0
    velocity: float = 0.0
    current: float = 0.0
    kp: float = 28.0
    kd: float = 6.0
    torque_ff: float = 0.0   # MIT feed-forward torque (Nm); assist/backdrive comp
    enabled: bool = False
    # Continuous position sweep (soak/simulation): when ``sweep_enabled`` the
    # loop drives position along a sine between ``sweep_from`` and ``sweep_to``
    # once per ``sweep_period`` seconds, timed from ``sweep_t0`` (monotonic).
    sweep_enabled: bool = False
    sweep_from: float = 0.0
    sweep_to: float = 0.0
    sweep_period: float = 2.0
    sweep_t0: float = 0.0


class ControlWorker(QObject):
    """Runs the control loop; lives in its own QThread."""

    statusUpdated = Signal(int, object)   # device_id, MotorStatus
    powerUpdated = Signal(int, object)    # device_id, PowerInfo
    connectionChanged = Signal(bool)
    scanFinished = Signal(list)           # list[int]
    busCollision = Signal(list)           # list[int]: ids answered by >1 motor
    inventoryReady = Signal(list)         # list[dict]: {"can_id", "uids"} per id
    log = Signal(str)
    error = Signal(str)
    motorEnabledChanged = Signal(int, bool)
    calibrationChanged = Signal(int, int, float)   # device_id, direction, offset
    rangeLimitsChanged = Signal(int, object, object)  # device_id, pos_min, pos_max (rad|None)
    motorIdChanged = Signal(int, int)              # old_id, new_id
    zeroStateUpdated = Signal(int, object)         # device_id, ZeroStateInfo
    sweepStopped = Signal(int)                     # device_id: sweep auto-stopped

    def __init__(self, rate_hz: float = 100.0):
        super().__init__()
        self._queue: "queue.Queue[Command]" = queue.Queue()
        self._running = True
        self._bus: Optional[RobstrideBus] = None
        self._targets: dict[int, MotorTarget] = {}
        self._calib: dict[int, Calibration] = {}
        # Per-motor calibrated travel range in the user frame: (min, max) rad;
        # either end may be None (uncalibrated -> no clamp on that side).
        self._range: dict[int, tuple[Optional[float], Optional[float]]] = {}
        # Live range-calibration capture, keyed by device_id. Each entry holds
        # the running (min, max) of observed position plus the run-mode/gains to
        # restore when calibration stops.
        self._range_cal: dict[int, dict] = {}
        self._last_raw_pos: dict[int, float] = {}
        # Last-seen set of active fault-flag names per motor, so faults are
        # logged only on change (rising/falling edge) instead of every 100 Hz
        # cycle. Cleared when a motor disables so re-enabling logs afresh.
        self._motor_faults: dict[int, frozenset[str]] = {}
        # Consecutive auto-re-enable attempts for a motor that reverted to standby
        # while we held it enabled, reset the moment it reports running again. Caps
        # the retry storm at MAX_HOLD_RECOVERY_ATTEMPTS (see _recover_dropped_hold).
        self._hold_recovery_attempts: dict[int, int] = {}
        # Consecutive RESET reads seen in a row per motor, reset on any MOTOR
        # (holding) read. See HOLD_DROP_DEBOUNCE.
        self._reset_streak: dict[int, int] = {}
        # Last VBUS value logged per motor, so the diagnostic power log fires only
        # on a significant voltage move (see VBUS_LOG_DELTA_V), not every sample.
        self._last_logged_vbus: dict[int, float] = {}
        # Newest (vbus, iq) read per motor, carried into the verbose telemetry
        # line so each row shows voltage/current alongside position and torque.
        self._last_power: dict[int, tuple[float, float]] = {}
        self._safety = SafetyState(SafetyLimits.for_model(proto.DEFAULT_MODEL))
        # True once the user has explicitly pushed limits via SetLimits, after
        # which the model-derived envelope is never recomputed behind their back.
        self._limits_user_set = False
        # One pool thread per CAN bus, so buses transmit CONCURRENTLY instead of
        # one waiting on the other. Created on connect once a second bus exists;
        # a single-bus setup runs inline and never touches a thread.
        self._pool: Optional[ThreadPoolExecutor] = None
        self._pool_size = 0
        self._rate_hz = rate_hz
        self._loop_count = 0
        self._comm_failures = 0
        self.motor_can_timeout_raw = MOTOR_CAN_TIMEOUT_RAW
        # Motors whose canTimeout arm has already been read back and confirmed
        # this session. The readback is a second blocking register round-trip
        # inside the enable handshake - the exact starvation window that trips a
        # bystander's watchdog - so it runs once per motor, not on every
        # auto-recovery re-enable. Cleared for a motor on disable so a fresh
        # enable re-verifies, and wholesale on disconnect.
        self._watchdog_verified: set[int] = set()

    # -- public, thread-safe API (called from the GUI thread) --------------------

    def post(self, command: Command) -> None:
        self._queue.put(command)

    def stop(self) -> None:
        self._running = False
        self._queue.put(Disconnect())

    # -- main loop ---------------------------------------------------------------

    def run(self) -> None:
        period = 1.0 / self._rate_hz
        while self._running:
            t0 = time.monotonic()
            self._loop_count += 1
            self._drain_commands()
            if self._bus is not None and self._bus.is_open:
                self._service_motors(t0)
            dt = time.monotonic() - t0
            if dt < period:
                time.sleep(period - dt)
        self._teardown()

    # -- command handling --------------------------------------------------------

    def _drain_commands(self) -> None:
        for _ in range(MAX_COMMANDS_PER_DRAIN):
            try:
                cmd = self._queue.get_nowait()
            except queue.Empty:
                return
            t0 = time.monotonic()
            try:
                self._apply(cmd)
            except TransportError as e:
                self.error.emit(str(e))
            except Exception as e:  # never let one bad command kill the loop
                self.error.emit(f"{type(e).__name__}: {e}")
            finally:
                self._log_slow_command(cmd, time.monotonic() - t0)

    def _log_slow_command(self, cmd: "Command", elapsed_s: float) -> None:
        """Log any command whose handler blocked the loop past SLOW_COMMAND_LOG_MS.

        The worker is single-threaded: while this handler runs, no other enabled
        motor gets a hold frame. The measured duration is exactly that starvation
        window, so a slow enable/mode-switch shows up here as a number to compare
        against a motor's firmware watchdog - the mechanism behind a bystander
        motor going limp when another is enabled. Names the command and target so
        the culprit is unambiguous."""
        elapsed_ms = elapsed_s * 1000.0
        if elapsed_ms < SLOW_COMMAND_LOG_MS:
            return
        device_id = getattr(cmd, "device_id", None)
        target = f" M{device_id}" if device_id is not None else ""
        self.log.emit(
            f"[timing]{target} {type(cmd).__name__} blocked the loop "
            f"{elapsed_ms:.0f} ms")

    def _apply(self, cmd: Command) -> None:
        if isinstance(cmd, Connect):
            self._connect(cmd)
        elif isinstance(cmd, AddMotor):
            self._add_motor(cmd.device_id, cmd.model, cmd.bus_name)
        elif isinstance(cmd, Disconnect):
            self._teardown()
        elif isinstance(cmd, Scan):
            self._scan(cmd)
        elif isinstance(cmd, Inventory):
            self._inventory(cmd)
        elif isinstance(cmd, Enable):
            self._enable(cmd.device_id)
        elif isinstance(cmd, Disable):
            self._disable(cmd.device_id)
        elif isinstance(cmd, SetZero):
            self._set_zero(cmd.device_id)
        elif isinstance(cmd, SetMode):
            self._set_mode(cmd.device_id, cmd.mode)
        elif isinstance(cmd, SetTarget):
            self._set_target(cmd)
        elif isinstance(cmd, SetSweep):
            self._set_sweep(cmd)
        elif isinstance(cmd, EStop):
            self._estop(cmd.engage)
        elif isinstance(cmd, SetLimits):
            self._safety.limits = cmd.limits
            self._limits_user_set = True
        elif isinstance(cmd, SetCalibration):
            self._calib[cmd.device_id] = Calibration(int(cmd.direction), float(cmd.offset))
            self.log.emit(f"M{cmd.device_id}: direction={'inverted' if cmd.direction < 0 else 'normal'}, "
                          f"offset={cmd.offset:.3f} rad")
        elif isinstance(cmd, CaptureZero):
            self._capture_zero(cmd.device_id)
        elif isinstance(cmd, SetMotorId):
            self._set_motor_id(cmd.current_id, cmd.new_id)
        elif isinstance(cmd, ReadZeroState):
            self._read_zero_state(cmd.device_id)
        elif isinstance(cmd, SetRangeLimits):
            self._set_range_limits(cmd.device_id, cmd.pos_min, cmd.pos_max)
        elif isinstance(cmd, StartRangeCalibration):
            self._start_range_calibration(cmd.device_id, cmd.make_limp)
        elif isinstance(cmd, StopRangeCalibration):
            self._stop_range_calibration(cmd.device_id)

    def _refresh_model_limits(self) -> None:
        """Rebuild the safety envelope from the models actually on the bus.

        The envelope used to be derived once from ``DEFAULT_MODEL`` ("rs-04") in
        ``__init__`` and never revisited, so it stayed an rs-04 envelope no
        matter what was connected. On an rs-03 that is wrong in both directions:
        the velocity cap (0.6 x rs-04's 15 rad/s = 9) throttles a motor rated far
        higher, while the torque cap (0.5 x rs-04's 120 Nm = 60 Nm) equals the
        rs-03's *entire* rated torque, i.e. no headroom protection at all.

        With a mixed bus there is still one global envelope, so take the
        strictest cap across the registered models - the only choice that is
        safe for every motor present. A user SetLimits always wins and is never
        overwritten here.
        """
        if self._bus is None or self._limits_user_set:
            return
        models = {m.model for m in self._bus.motors.values()} or {proto.DEFAULT_MODEL}
        envelopes = [SafetyLimits.for_model(model) for model in sorted(models)]
        strictest = envelopes[0]
        for env in envelopes[1:]:
            strictest = strictest.with_(
                position_min=max(strictest.position_min, env.position_min),
                position_max=min(strictest.position_max, env.position_max),
                velocity_max=min(strictest.velocity_max, env.velocity_max),
                current_max=min(strictest.current_max, env.current_max),
                torque_max=min(strictest.torque_max, env.torque_max),
                kp_max=min(strictest.kp_max, env.kp_max),
                kd_max=min(strictest.kd_max, env.kd_max),
            )
        self._safety.limits = strictest

    def _ensure_pool(self) -> None:
        """Size the bus thread pool to the number of registered buses.

        Skipped entirely for a single bus: with nothing to run in parallel the
        pool would only add hand-off cost, and the inline path keeps
        single-adapter behaviour exactly as it was. The pool only grows - a
        smaller reconnect leaves spare idle threads, which is cheaper and safer
        than tearing down a pool that may still hold a running tick.
        """
        n = len(self._bus.buses) if self._bus is not None else 0
        if n <= 1 or (self._pool is not None and self._pool_size >= n):
            return
        old = self._pool
        self._pool = ThreadPoolExecutor(max_workers=n, thread_name_prefix="bus")
        self._pool_size = n
        if old is not None:
            old.shutdown(wait=False)

    def _connect(self, cmd: Connect) -> None:
        self._comm_failures = 0
        # Additive: a second Connect under a new bus_name joins the existing
        # router rather than discarding the first adapter. Reconnecting the
        # SAME name replaces just that bus.
        if self._bus is None:
            self._bus = BusRouter()
        self._bus.add_bus(cmd.bus_name, RobstrideBus(cmd.transport, BusConfig()))
        for m in cmd.motors:
            self._bus.add_motor(m, cmd.bus_name)
            self._targets.setdefault(m.device_id, MotorTarget())
        self._refresh_model_limits()
        self._ensure_pool()
        self._bus.open()
        self.connectionChanged.emit(True)
        self.log.emit(f"Connected via {cmd.transport.name}")

    def _add_motor(self, device_id: int, model: str,
                   bus_name: Optional[str] = None) -> None:
        if self._bus is None:
            return
        self._bus.add_motor(Motor(device_id=device_id, model=model), bus_name)
        self._targets.setdefault(device_id, MotorTarget())
        self._refresh_model_limits()

    def _teardown(self) -> None:
        if self._bus is not None:
            for device_id, target in self._targets.items():
                if target.enabled:
                    try:
                        self._bus.disable(device_id)
                    except Exception:
                        pass
                    target.enabled = False
            try:
                self._bus.close()
            except Exception:
                pass
            self._bus = None
            if self._pool is not None:
                # wait=False: never block the control loop on a wedged bus
                # thread. Its transport is closed above, so it unwinds shortly.
                self._pool.shutdown(wait=False)
                self._pool = None
                self._pool_size = 0
            self._comm_failures = 0
            self._motor_faults.clear()
            self._hold_recovery_attempts.clear()
            self._reset_streak.clear()
            self._watchdog_verified.clear()
            self._last_logged_vbus.clear()
            self._last_power.clear()
            self.connectionChanged.emit(False)
            self.log.emit("Disconnected")

    def _scan(self, cmd: Scan) -> None:
        if not self._bus:
            return
        found = self._bus.scan(cmd.start, cmd.end)
        self.scanFinished.emit(found)
        self.log.emit(f"Scan found motors: {found or 'none'}")
        collisions = self._bus.find_collisions(found)
        if collisions:
            self.busCollision.emit(collisions)
            ids = ", ".join(str(d) for d in collisions)
            self.log.emit(f"WARNING: CAN id collision on {ids} - "
                          "multiple motors share an id and move together")

    def _inventory(self, cmd: Inventory) -> None:
        if not self._bus:
            return
        items = self._bus.inventory(cmd.start, cmd.end)
        payload = [{"can_id": did, "uids": [u.hex() for u in uids]}
                   for did, uids in items]
        self.inventoryReady.emit(payload)
        self.log.emit(f"Device inventory: {len(payload)} id(s) responding")

    def _enable(self, device_id: int) -> None:
        if not self._bus:
            return
        target = self._targets.setdefault(device_id, MotorTarget())
        # Hard safety latch: never energise a motor while E-STOP is engaged. The
        # control loop already skips commanding an estopped motor, but enabling
        # still applies holding torque - the operator expects a dead motor.
        if self._safety.estop:
            self.error.emit(
                f"M{device_id}: E-STOP engaged - clear it before enabling")
            self.motorEnabledChanged.emit(device_id, False)
            return
        # Assert the run-mode before enabling: the motor keeps its previous mode
        # across power cycles, so without this it may sit in MIT mode and ignore
        # every position/velocity/current setpoint we send.
        self._bus.set_run_mode(device_id, target.mode)
        self._feed_other_motors(device_id)
        # Safe-enable: seed the setpoint to the shaft's *current* position so the
        # motor comes up HOLDING where it is. Without this it snaps to the last
        # (default 0.0) position target the instant it is enabled, which can slam
        # attached hardware into its surroundings. Only position/MIT modes hold a
        # position; velocity/current default to 0 (no motion) and need no seed.
        if target.mode in _POSITION_MODES or target.mode == RunMode.MIT:
            self._seed_hold_position(device_id, target)
        # The seed above can be several blocking round-trips; feed the others
        # before pressing on so their watchdogs do not starve (see
        # _feed_other_motors).
        self._feed_other_motors(device_id)
        self._configure_motor_limits(device_id)
        self._configure_motor_watchdog(device_id)
        self._feed_other_motors(device_id)
        self._bus.enable(device_id)
        target.enabled = True
        self._feed_other_motors(device_id)
        # Re-assert the hold setpoint AFTER enabling. The pre-enable seed above is
        # a belt-and-suspenders: RobStride ignores a loc_ref (POSITION_TARGET)
        # write while the motor is disabled, so on its own it keeps the internal
        # target from the *previous* enable and profiles straight back to that old
        # angle the instant it is energised (the "jumps to the last position, not
        # the new zero" bug - most visible after a Set Zero at a hand-moved spot).
        # Writing loc_ref once more here, now that the motor accepts it, pins the
        # hold to the shaft's current spot and closes the window before the first
        # control-loop tick.
        if target.mode in _POSITION_MODES:
            raw = self._last_raw_pos.get(device_id)
            if raw is not None:
                self._bus.set_position(device_id, raw, self._safety.limits.velocity_max)
        self.motorEnabledChanged.emit(device_id, True)
        self.log.emit(f"M{device_id}: enabled (mode {RunMode.NAMES.get(target.mode, target.mode)})")

    def _seed_hold_position(self, device_id: int, target: MotorTarget) -> None:
        """Make the shaft's current position the hold setpoint before enabling.

        The primary reading comes from ``poll_status`` so it is in the SAME frame
        as the live feedback and the motor's mechanical zero. A plain ``mechPos``
        param read can lag a just-applied Set Zero: it would seed the setpoint in
        the old frame and swing the shaft back to its pre-zero position on enable.
        The motor is always disabled at this point, so the zero-gain status frame
        commands no motion.

        ``poll_status`` relies on an operation-control frame, though, and a
        disabled or fault-latched motor can decline to ack that while still
        answering a plain parameter read. So if the poll comes back empty, fall
        back to reading ``mechPos`` (comm 17) before giving up: that keeps us from
        enabling blind - and from mislabelling a reachable motor as a dead link -
        whenever the operation-frame poll is the only thing failing. The fallback
        runs only when the poll fails, so the fresh-after-Set-Zero case still uses
        ``poll_status`` and is unaffected. For profiled/CSP modes we also pre-load
        ``loc_ref`` so the first hold is the current spot. A read that fails BOTH
        ways is surfaced loudly: enabling blind risks the jump this guards against.
        """
        status = self._bus.poll_status(device_id)
        raw = float(status.position) if status is not None \
            else self._read_mech_pos(device_id)
        if raw is None:
            self.error.emit(
                f"M{device_id}: could not read position before enable - the "
                "motor may jump to its last target. Check power and wiring.")
            return
        c = self._calib.get(device_id) or Calibration()
        self._last_raw_pos[device_id] = raw
        target.position = c.pos_from_raw(raw)
        if target.mode in _POSITION_MODES:
            self._bus.set_position(device_id, raw, self._safety.limits.velocity_max)

    def _read_mech_pos(self, device_id: int) -> Optional[float]:
        """Read the shaft's mechanical position via a parameter read (comm 17).

        Unlike ``poll_status``' operation-control frame, a plain READ_PARAMETER is
        answered by a disabled or fault-latched motor, so it is a reliable
        fallback for seeding the pre-enable hold. Returns ``None`` if the motor
        does not answer or the bus errors.
        """
        try:
            val = self._bus.read_param(device_id, ParameterType.MEASURED_POSITION)
        except TransportError:
            return None
        return None if val is None else float(val)

    def _disable(self, device_id: int) -> None:
        if not self._bus:
            return
        self._bus.disable(device_id)
        target = self._targets.get(device_id)
        if target is not None:
            target.enabled = False
            # Stop any sweep so the next enable holds position instead of
            # resuming the trajectory - that would defeat safe-enable.
            self._stop_sweep(device_id, target)
        # Forget the last fault set so a re-enable reports its faults fresh.
        self._motor_faults.pop(device_id, None)
        # An operator disable is a clean slate: forget any in-progress hold-drop
        # recovery so a later manual enable starts its attempt count from zero.
        self._hold_recovery_attempts.pop(device_id, None)
        self._reset_streak.pop(device_id, None)
        # Re-verify the canTimeout arm on the next enable (the motor may have
        # been power-cycled or reconfigured while disabled).
        self._watchdog_verified.discard(device_id)
        self.motorEnabledChanged.emit(device_id, False)
        self.log.emit(f"M{device_id}: disabled")

    def _set_zero(self, device_id: int) -> None:
        """Rewrite the motor's mechanical zero at the current spot and persist it.

        SET_ZERO_POSITION is meant for a *stopped* motor: issuing it while a
        position loop is actively holding shifts the encoder frame out from under
        the setpoint, so the shaft twitches, and the ~0.25 s save (see
        ``RobstrideBus.set_zero``) freezes the readout mid-move - the operator
        sees the feedback wander instead of snapping to 0. So if the motor is
        enabled we disable it around the zero, then re-seed the hold at the new
        zero and re-enable: the shaft stays put and the readout lands cleanly on
        0. If it was already disabled this is a plain zero with no motion.
        """
        if self._bus is None:
            return
        target = self._targets.get(device_id)
        was_enabled = target is not None and target.enabled
        # Stop the active position loop before redefining its reference frame.
        if was_enabled:
            self._disable(device_id)
        self._bus.set_zero(device_id)
        self.log.emit(f"M{device_id}: zero set and saved to flash")
        # A hardware zero remakes the motor's mechanical frame at the current
        # spot. Any software offset would now double-count, so the readout would
        # show -offset instead of 0. Clear it (keep the direction/invert) so the
        # two frames agree; the change is persisted via calibrationChanged.
        calib = self._calib.setdefault(device_id, Calibration())
        calib.offset = 0.0
        self.calibrationChanged.emit(device_id, calib.direction, 0.0)
        # The new zero is the current spot, so any stored hold setpoint (e.g. one
        # frozen mid-swing on disable) is now stale and would swing the shaft back
        # there on the next enable. Reset it to the new zero.
        if target is not None:
            target.position = 0.0
        # Refresh the readout once so the new zero shows immediately. The motor is
        # disabled at this point (either it already was, or we just disabled it),
        # so poll_status is safe - it will not inject a brake frame into a live
        # control loop.
        status = self._bus.poll_status(device_id)
        if status is not None:
            self.statusUpdated.emit(
                device_id, self._decalibrate(device_id, status, calib))
        # Read the motor's own zero back so the UI confirms it stuck.
        self._read_zero_state(device_id)
        # Bring the motor back up holding the new zero if it was running before;
        # _enable re-seeds the hold from the (now ~0) live position, so the shaft
        # stays put rather than jumping.
        if was_enabled:
            self._enable(device_id)

    def _stop_sweep(self, device_id: int, target: MotorTarget) -> None:
        """Clear a running sweep, freezing its setpoint where it left off. Emits
        ``sweepStopped`` so the UI's sweep button clears too. No-op if idle."""
        if not target.sweep_enabled:
            return
        target.position = self._sweep_position(target, time.monotonic())
        target.sweep_enabled = False
        self.sweepStopped.emit(device_id)

    def _set_mode(self, device_id: int, mode: int) -> None:
        if not self._bus:
            return
        target = self._targets.setdefault(device_id, MotorTarget())
        # A run-mode switch is only reliably accepted while the motor is
        # disabled; bracket the change so an enabled motor ends up enabled again
        # in the new mode (this is what makes the jog buttons work).
        was_enabled = target.enabled
        if was_enabled:
            self._bus.disable(device_id)
            # This motor is now briefly down; feed the others while we
            # reconfigure it so their watchdogs stay fed (see _feed_other_motors).
            self._feed_other_motors(device_id)
        self._bus.set_run_mode(device_id, mode)
        target.mode = mode
        if was_enabled:
            # Safe-enable on the re-enable too: switching an enabled motor into a
            # position mode would otherwise snap it to the stale setpoint. Seed
            # the current position first, exactly as _enable does.
            if mode in _POSITION_MODES or mode == RunMode.MIT:
                self._seed_hold_position(device_id, target)
            self._feed_other_motors(device_id)
            # Re-push the current/torque limits: a run-mode rewrite can reset the
            # motor's limit registers to a default (or zero), and without this the
            # motor re-enables with no torque budget and sits LIMP (exactly 0.00 Nm
            # / 0.0 A) - the "motor goes dead after a mode switch" bug. _enable
            # writes these on every enable; the mode-switch re-enable must too.
            self._configure_motor_limits(device_id)
            # Re-arm the motor-side watchdog too: this disable/enable pulse is
            # a full enable, and firmware may not keep canTimeout across it.
            self._configure_motor_watchdog(device_id)
            self._feed_other_motors(device_id)
            self._bus.enable(device_id)
            self._feed_other_motors(device_id)
            # Re-assert the hold setpoint AFTER enabling, exactly as _enable does.
            # RobStride ignores a loc_ref (POSITION_TARGET) write while the motor
            # is disabled, so the _seed_hold_position call above - done while
            # disabled - does NOT latch the target. Without this the position loop
            # comes up with no valid hold and the motor sits limp (zero torque)
            # and drifts instead of holding, which reads as "the motor dropped its
            # hold after a mode switch". Writing loc_ref once more here, now that
            # the motor accepts it, pins the hold to the seeded spot. See
            # robstride-locref-disabled-quirk and the matching step in _enable.
            if mode in _POSITION_MODES:
                raw = self._last_raw_pos.get(device_id)
                if raw is not None:
                    self._bus.set_position(device_id, raw, self._safety.limits.velocity_max)
        self.log.emit(f"M{device_id}: mode -> {RunMode.NAMES.get(mode, mode)}")

    def _set_target(self, cmd: SetTarget) -> None:
        t = self._targets.setdefault(cmd.device_id, MotorTarget())
        if cmd.position is not None:
            t.position = cmd.position
        if cmd.velocity is not None:
            t.velocity = cmd.velocity
        if cmd.current is not None:
            t.current = cmd.current
        if cmd.kp is not None:
            t.kp = cmd.kp
        if cmd.kd is not None:
            t.kd = cmd.kd
        if cmd.torque_ff is not None:
            t.torque_ff = cmd.torque_ff

    def _set_sweep(self, cmd: SetSweep) -> None:
        t = self._targets.setdefault(cmd.device_id, MotorTarget())
        t.sweep_from = cmd.from_pos
        t.sweep_to = cmd.to_pos
        # Guard the divisor: a zero/negative period would divide by zero in the
        # sine phase. 50 ms is already far faster than any real motor can track.
        t.sweep_period = max(float(cmd.period), 0.05)
        t.sweep_enabled = cmd.enabled
        if cmd.enabled:
            t.sweep_t0 = time.monotonic()
            self.log.emit(
                f"M{cmd.device_id}: sweep {math.degrees(cmd.from_pos):+.1f} -> "
                f"{math.degrees(cmd.to_pos):+.1f} deg, period {t.sweep_period:.2f}s")
        else:
            # Freeze the setpoint where the sweep left off so the motor holds
            # instead of jumping back to a stale manual position.
            t.position = self._sweep_position(t, time.monotonic())
            self.log.emit(f"M{cmd.device_id}: sweep stopped")

    @staticmethod
    def _sweep_position(t: MotorTarget, now: Optional[float] = None) -> float:
        """Sine setpoint for the current time: ``from`` at t0, ``to`` at the
        half-period, back to ``from`` at the full period. Continuous in velocity
        (no snap at the endpoints), which is what makes it smooth."""
        if now is None:
            now = time.monotonic()
        mid = (t.sweep_from + t.sweep_to) / 2.0
        amp = (t.sweep_to - t.sweep_from) / 2.0
        phase = (now - t.sweep_t0) / t.sweep_period
        return mid - amp * math.cos(2.0 * math.pi * phase)

    def _estop(self, engage: bool) -> None:
        if engage:
            self._safety.engage_estop()
            self.log.emit("E-STOP engaged")
            if self._bus:
                for device_id, target in self._targets.items():
                    self._stop_sweep(device_id, target)
                    if target.enabled:
                        try:
                            self._bus.disable(device_id)
                        except Exception:
                            pass
                        target.enabled = False
                        self.motorEnabledChanged.emit(device_id, False)
        else:
            self._safety.clear_estop()
            self.log.emit("E-STOP cleared")

    def _set_motor_id(self, current_id: int, new_id: int) -> None:
        if not self._bus:
            self.error.emit("Connect before changing a motor id")
            return
        if new_id == current_id:
            return
        live_id = self._bus.set_motor_id(current_id, new_id)
        if live_id is None:
            self.error.emit(f"Motor {current_id} did not respond after the id change. "
                            "Power-cycle the motor and press Detect.")
            return
        if live_id == current_id:
            # The motor still answers at its old id: the new id is not active
            # yet. Leave the panel where it is so live feedback keeps flowing.
            self.log.emit(f"Motor still at ID {current_id}: new id not active yet. "
                          "Power-cycle the motor, then Detect.")
            return
        # Follow the motor to the id it actually responds at now.
        for table in (self._targets, self._calib, self._range,
                      self._range_cal, self._last_raw_pos):
            if current_id in table:
                table[live_id] = table.pop(current_id)
        self.motorIdChanged.emit(current_id, live_id)
        target = self._targets.get(live_id)
        if target is not None and target.enabled:
            self.motorEnabledChanged.emit(live_id, True)
        self.log.emit(f"Motor {current_id} -> ID {live_id} (verified and saved).")

    def _capture_zero(self, device_id: int) -> None:
        raw = self._last_raw_pos.get(device_id)
        if raw is None:
            self.log.emit(f"M{device_id}: no feedback yet - enable the motor first")
            return
        calib = self._calib.setdefault(device_id, Calibration())
        calib.offset = raw
        self.calibrationChanged.emit(device_id, calib.direction, calib.offset)
        self.log.emit(f"M{device_id}: zero captured at {raw:.3f} rad")

    # -- range calibration -------------------------------------------------------

    def _set_range_limits(self, device_id: int, pos_min: Optional[float],
                          pos_max: Optional[float]) -> None:
        """Store a motor's travel range, normalising so min <= max.

        An inverted-direction motor can report the two ends in either order; sort
        them so the clamp always has a well-formed [lo, hi]. Either end may stay
        ``None`` to leave that side unbounded.
        """
        lo = None if pos_min is None else float(pos_min)
        hi = None if pos_max is None else float(pos_max)
        if lo is not None and hi is not None and lo > hi:
            lo, hi = hi, lo
        self._range[device_id] = (lo, hi)
        self.rangeLimitsChanged.emit(device_id, lo, hi)
        span = ("unbounded" if lo is None and hi is None
                else f"[{'-inf' if lo is None else f'{lo:.3f}'}, "
                     f"{'+inf' if hi is None else f'{hi:.3f}'}] rad")
        self.log.emit(f"M{device_id}: range limits {span}")

    def _start_range_calibration(self, device_id: int, make_limp: bool) -> None:
        t = self._targets.setdefault(device_id, MotorTarget())
        state = {"min": None, "max": None, "prev_mode": t.mode,
                 "prev_kp": t.kp, "prev_kd": t.kd, "prev_tq": t.torque_ff,
                 "limp": make_limp}
        # Seed from the current position if we already have feedback, so a motor
        # sitting still still yields a (degenerate) range rather than nothing.
        c = self._calib.get(device_id) or Calibration()
        raw = self._last_raw_pos.get(device_id)
        if raw is not None:
            user = c.pos_from_raw(raw)
            state["min"] = state["max"] = user
        self._range_cal[device_id] = state
        if make_limp:
            # MIT with zero gains/torque = no holding effort, so the shaft can be
            # moved by hand. Goes through _set_mode so an enabled motor is cleanly
            # re-armed in the new mode.
            self._set_mode(device_id, RunMode.MIT)
            t.kp = 0.0
            t.kd = 0.0
            t.torque_ff = 0.0
        self.log.emit(f"M{device_id}: range calibration started - move the motor "
                      "through its travel (by hand or jog), then Stop")

    def _stop_range_calibration(self, device_id: int) -> None:
        state = self._range_cal.pop(device_id, None)
        if state is None:
            return
        t = self._targets.get(device_id)
        if t is not None and state.get("limp"):
            # Restore the gains we zeroed, then the run-mode we came from.
            t.kp = state["prev_kp"]
            t.kd = state["prev_kd"]
            t.torque_ff = state["prev_tq"]
            self._set_mode(device_id, state["prev_mode"])
        lo, hi = state["min"], state["max"]
        if lo is None or hi is None:
            self.log.emit(f"M{device_id}: range calibration stopped - no motion "
                          "captured (enable the motor and move it), range unchanged")
            return
        self._set_range_limits(device_id, lo, hi)

    def _note_range_sample(self, device_id: int, user_pos: float) -> None:
        """Fold one live position sample into an active calibration's min/max."""
        state = self._range_cal.get(device_id)
        if state is None:
            return
        if state["min"] is None or user_pos < state["min"]:
            state["min"] = user_pos
        if state["max"] is None or user_pos > state["max"]:
            state["max"] = user_pos

    def _clamp_to_range(self, device_id: int, value: float) -> float:
        lo, hi = self._range.get(device_id, (None, None))
        if lo is not None and value < lo:
            return lo
        if hi is not None and value > hi:
            return hi
        return value

    def _read_zero_state(self, device_id: int) -> None:
        """Read the motor's persisted zero markers and emit them for display."""
        if not self._bus:
            return
        info = self._bus.read_zero_state(device_id)
        if info is None:
            self.log.emit(f"M{device_id}: zero state unavailable (no reply)")
            return
        self.zeroStateUpdated.emit(
            device_id, ZeroStateInfo(device_id, info["zero_sta"], info["mech_offset"]))
        self.log.emit(f"M{device_id}: motor zero_sta={info['zero_sta']}, "
                      f"mechOffset={info['mech_offset']}")

    def _configure_motor_limits(self, device_id: int) -> None:
        """Push the software current/torque caps into the motor's own limit
        registers so they apply in *every* run-mode.

        Only velocity mode wrote ``limit_cur`` before (via ``set_velocity``), so
        in position/MIT mode the motor kept whatever current/torque limit it
        powered up with - a ~10 A default throttles an rs-04 to ~15 Nm, far
        below ``torque_max``. Writing them on each enable makes the software caps
        authoritative regardless of mode. Best-effort: a motor that does not ack
        still enables.
        """
        lim = self._safety.limits
        if lim.current_max is not None:
            self._bus.write_param(device_id, ParameterType.CURRENT_LIMIT,
                                  float(lim.current_max))
        if lim.torque_max is not None:
            self._bus.write_param(device_id, ParameterType.TORQUE_LIMIT,
                                  float(lim.torque_max))

    def _configure_motor_watchdog(self, device_id: int) -> None:
        """Arm the motor's own CAN watchdog before enabling it.

        Writes ``motor_can_timeout_raw`` to the canTimeout register (0x7028) so
        the motor stops itself if the host goes silent - the failure a
        host-side watchdog cannot cover - and widens the firmware watchdog past
        its short (~60 ms) stored default so ordinary servicing gaps no longer
        trip a bystander (see :data:`MOTOR_CAN_TIMEOUT_RAW` for why the old
        1000-as-1s value did not, and what the raw unit actually is).
        Best-effort: a motor that does not ack the write still enables, but the
        gap is surfaced as an error so the operator knows the safety net is
        missing.

        The write goes out on *every* enable (cheap, one frame). The readback
        that confirms it stuck is a second blocking register round-trip - dead
        air for the other motors, inside the very starvation window that trips
        their watchdogs - so it runs only the first time each motor is armed
        this session, not on every hold-recovery re-enable. A readback of 0/None
        means the arm silently failed; a different non-zero value means the
        firmware clamped or rescaled it. Only the starve test
        (:meth:`measure_can_timeout`) proves the real timeout duration.
        """
        raw = int(self.motor_can_timeout_raw)
        if raw <= 0:
            return
        ack = self._bus.write_param(device_id, ParameterType.CAN_TIMEOUT, raw)
        if ack is None:
            # error (not just log): an unarmed motor-side watchdog is the most
            # safety-relevant gap here - the motor will NOT stop on its own if
            # the link to the host drops. The operator must see this.
            self.error.emit(
                f"M{device_id}: canTimeout write not acked - the motor will "
                "NOT stop on its own if the link to the host drops")
            self._watchdog_verified.discard(device_id)  # re-verify next enable
            return
        if device_id in self._watchdog_verified:
            return  # already confirmed this session - skip the blocking readback
        try:
            stored = self._bus.read_param(device_id, ParameterType.CAN_TIMEOUT)
        except TransportError:
            return  # readback is confirmation only; a hiccup here is not fatal
        if not stored:
            # Reads back 0 or nothing: the write did not stick, so the watchdog
            # is NOT armed despite the ack - as safety-relevant as no ack at all.
            self.error.emit(
                f"M{device_id}: canTimeout did not stick (reads back {stored}) - "
                "the motor will NOT stop on its own if the host link drops")
        elif int(stored) != raw:
            # Armed, but not with our value: the firmware reinterpreted it, so the
            # real stop time is unknown until verified by the starve test.
            self.log.emit(
                f"M{device_id}: canTimeout wrote {raw}, motor stored "
                f"{int(stored)} - verify the actual stop time")
            self._watchdog_verified.add(device_id)
        else:
            self._watchdog_verified.add(device_id)

    def measure_can_timeout(self, device_id: int, candidate_raw: int,
                            probes_ms: "Optional[list[float]]" = None,
                            ) -> "Optional[float]":
        """Measure the real canTimeout window for ``candidate_raw`` by starving it.

        The register unit is not milliseconds (see :data:`MOTOR_CAN_TIMEOUT_RAW`),
        so the only way to know what a raw value buys is to arm it and see how
        long the motor holds with no frames. This automates that "unplug test"
        *without* unplugging: for each silence duration in ``probes_ms``
        (ascending) it arms ``candidate_raw``, re-enables the motor so it is
        running and holding, then sends **nothing** for that long and polls once.
        A poll frame resets the watchdog, so it is sent only at the *end* of the
        silence - during the wait the motor is truly starved. The first probe
        after which the motor reports standby brackets the timeout; the returned
        estimate is the midpoint between that probe and the previous (still
        holding) one. Feed it to :func:`recommend_can_timeout_raw` to get the raw
        value for :data:`MOTOR_CAN_TIMEOUT_TARGET_MS`.

        Returns the estimated window in ms, or ``None`` if the motor never
        dropped within the longest probe (the value is already very wide, or -
        less likely - a poll resets a watchdog that a control frame would not).
        Blocking and operator-invoked: run it on a *single* motor with the bus
        otherwise idle, never during normal operation. Best-effort; a bus error
        aborts and returns ``None``.
        """
        if self._bus is None or self._safety.estop or candidate_raw <= 0:
            return None
        probes = sorted(probes_ms or [50.0, 100.0, 200.0, 400.0, 800.0, 1600.0])
        saved_raw = self.motor_can_timeout_raw
        self.motor_can_timeout_raw = candidate_raw
        prev = 0.0
        try:
            for wait_ms in probes:
                self._watchdog_verified.discard(device_id)  # force a fresh arm
                self._enable(device_id)                     # arms + energises + holds
                time.sleep(wait_ms / 1000.0)                # dead silence: no frames
                status = self._bus.poll_status(device_id)   # resets watchdog, reads mode
                if status is not None and status.mode == MotorMode.RESET:
                    estimate = (prev + wait_ms) / 2.0
                    self.log.emit(
                        f"M{device_id}: canTimeout={candidate_raw} raw dropped "
                        f"between {prev:.0f} and {wait_ms:.0f} ms silence "
                        f"(~{estimate:.0f} ms)")
                    return estimate
                prev = wait_ms
        except TransportError:
            return None
        finally:
            self.motor_can_timeout_raw = saved_raw
            self._disable(device_id)
        self.log.emit(
            f"M{device_id}: canTimeout={candidate_raw} raw still holding after "
            f"{probes[-1]:.0f} ms - window is at least that wide")
        return None

    def _feed_other_motors(self, exclude_id: int) -> None:
        """Send one hold frame to every *other* enabled motor.

        The worker is single-threaded, so a multi-step enable / mode-switch
        handshake for one motor is dead air for all the others: no setpoint
        frame goes out while it runs (the whole sequence is drained before the
        service loop gets a turn). A bystander motor whose firmware CAN-watchdog
        window is shorter than that handshake reverts to standby and goes limp -
        the "M6 drops to mode 0 when M5 is enabled" bug, seen on the wire as a
        mode 2 -> mode 0 flip with *no* fault bit and a steady VBUS. Calling this
        between the blocking steps of the handshake keeps the bystanders'
        watchdogs fed: each gets the same hold frame the service loop would send,
        capping any one motor's starvation to a single handshake step instead of
        the whole sequence.

        Best-effort: a bus error on a keepalive is swallowed rather than torn
        down here, so a transient hiccup cannot abort the in-progress enable
        mid-sequence. The 100 Hz service loop and its comm-failure watchdog own
        link-death detection.
        """
        if self._bus is None or self._safety.estop:
            return
        for device_id, target in list(self._targets.items()):
            if device_id == exclude_id or not target.enabled:
                continue
            try:
                self._command_motor(device_id, target)
            except TransportError:
                pass
            if self._bus is None:
                return

    def _recover_dropped_hold(self, device_id: int, target: MotorTarget,
                              status: MotorStatus) -> None:
        """Re-enable a motor that silently reverted to standby while enabled.

        A RobStride motor whose firmware CAN-watchdog trips (starved during
        another motor's blocking enable, before fix #1's keepalives closed that
        window - or a canTimeout shorter than a servicing gap) de-energises to
        :data:`MotorMode.RESET`: limp, ignoring setpoints, yet still ACKing our
        writes, so nothing else notices it stopped holding. It sets NO fault bit
        and the supply stays flat; the mode field is the only tell.

        On seeing that drop we re-run the full safe-enable, which re-seeds the
        hold at the shaft's *current* position (no snap-back) and, via
        ``_feed_other_motors``, does not starve the other motors in turn - the
        reason this is safe now and was a ping-pong hazard before. If the motor
        still reports standby after :data:`MAX_HOLD_RECOVERY_ATTEMPTS` re-enables
        the cause is persistent: stop, leave it disabled, and tell the operator.

        The attempt counter resets the instant the motor reports running again,
        so an isolated drop that recovers on the first try never counts against a
        later, unrelated one.
        """
        if self._safety.estop or device_id in self._range_cal:
            return
        if status.mode == MotorMode.MOTOR:
            self._hold_recovery_attempts.pop(device_id, None)  # holding again
            self._reset_streak.pop(device_id, None)
            return
        if status.mode != MotorMode.RESET:
            return  # transient CALI or unknown state - do not act on it
        streak = self._reset_streak.get(device_id, 0) + 1
        if streak < HOLD_DROP_DEBOUNCE:
            self._reset_streak[device_id] = streak
            return  # one RESET read alone - wait for a second before acting
        self._reset_streak[device_id] = 0
        attempts = self._hold_recovery_attempts.get(device_id, 0)
        if attempts >= MAX_HOLD_RECOVERY_ATTEMPTS:
            self._hold_recovery_attempts.pop(device_id, None)
            try:
                self._bus.disable(device_id)
            except Exception:
                pass
            target.enabled = False
            self._stop_sweep(device_id, target)
            self.motorEnabledChanged.emit(device_id, False)
            self.error.emit(
                f"M{device_id}: keeps dropping to standby after {attempts} "
                "re-enable attempts - left disabled. Check the motor's CAN "
                "watchdog (canTimeout), wiring, and bus load.")
            return
        self._hold_recovery_attempts[device_id] = attempts + 1
        self.log.emit(
            f"M{device_id}: hold dropped to standby (no fault) - auto "
            f"re-enabling (attempt {attempts + 1}/{MAX_HOLD_RECOVERY_ATTEMPTS})")
        self._enable(device_id)

    # -- host-side communication watchdog -----------------------------------------

    def _note_comm_failure(self, message: str) -> None:
        """Count a transport-level failure; disconnect once the limit is hit.

        Only the first failure of a burst is surfaced (the UI would otherwise
        get one error per motor per 100 Hz cycle). Reaching
        :data:`COMM_FAILURE_LIMIT` consecutive failures means the link is dead,
        not glitching: tear down instead of spinning on it while enabled motors
        run unsupervised - their own canTimeout watchdog stops them.
        """
        self._comm_failures += 1
        if self._comm_failures == 1:
            self.error.emit(message)
        elif self._comm_failures >= COMM_FAILURE_LIMIT:
            self.error.emit(
                f"Connection lost: {self._comm_failures} consecutive bus "
                f"failures (last: {message}). Disconnecting - enabled motors "
                "stop via their canTimeout watchdog.")
            self._teardown()

    # -- per-loop servicing ------------------------------------------------------

    def _service_motors(self, t_tick: Optional[float] = None) -> None:
        """One control tick: command every enabled motor, then publish results.

        Split into two phases so multiple CAN buses run at the same time:

        1. IO phase - each bus's motors are serviced on their OWN thread, all
           buses in parallel. Pure blocking IO, which releases the GIL, so this
           is real concurrency. Nothing here emits Qt signals.
        2. Publish phase - back on this thread, in one pass, results are turned
           into signals and follow-up actions (fault logs, range cutout, hold
           recovery). Keeping every emit on one thread means the UI sees exactly
           the same single-threaded signal stream it always did.

        Within a bus, motors are still commanded one after another - CAN is one
        wire and carries one frame at a time. What changes is that can1 no longer
        waits for can0 to finish.
        """
        if t_tick is None:
            t_tick = time.monotonic()
        read_power = self._loop_count % POWER_READ_DIVISOR == 0
        # The failure counter may only reset after a *fully clean* pass. If it
        # were reset on any motor's success, one healthy motor would clear the
        # count that a dead one keeps accruing in the same cycle, and the
        # watchdog could never trip (nor its error dedup engage).
        failures_before = self._comm_failures
        records = self._run_bus_batches(self._batch_by_bus(), t_tick, read_power)
        self._publish_records(records)
        if self._bus is not None and self._comm_failures == failures_before:
            self._comm_failures = 0  # clean pass: the link is healthy again

    def _batch_by_bus(self) -> "dict[str, list[tuple[int, MotorTarget]]]":
        """Group the enabled motors by the bus that owns them."""
        if self._bus is None or self._safety.estop:
            return {}
        enabled = [(d, t) for d, t in list(self._targets.items()) if t.enabled]
        buses = getattr(self._bus, "buses", None)
        if not buses:
            # A plain RobstrideBus rather than a BusRouter (a direct caller or a
            # test double). Treat it as one unnamed bus: every motor in a single
            # batch, serviced inline, exactly as before routing existed.
            return {"": enabled} if enabled else {}
        # Reverse map bus object -> name. Rebuilt each tick rather than cached:
        # it is a handful of entries, and a cache would go stale on reconnect.
        name_of = {id(bus): name for name, bus in buses.items()}
        batches: "dict[str, list[tuple[int, MotorTarget]]]" = {}
        for device_id, target in enabled:
            try:
                bus = self._bus.bus_for(device_id)
            except KeyError:
                continue  # not registered on any bus - nothing to command
            name = name_of.get(id(bus))
            if name is not None:
                batches.setdefault(name, []).append((device_id, target))
        return batches

    def _run_bus_batches(self, batches: dict, t_tick: float,
                         read_power: bool) -> list:
        """Run every bus's batch, in parallel when there is more than one."""
        if not batches:
            return []
        if len(batches) == 1 or self._pool is None:
            # Single bus: run inline. No thread hand-off, and the behaviour is
            # byte-for-byte what it was before threads existed.
            batch = next(iter(batches.values()))
            return self._service_bus(batch, t_tick, read_power)
        # Submit EVERY bus before waiting on any of them - that is what makes
        # them overlap. Submitting and collecting one at a time would still be
        # sequential, which is the whole thing this exists to avoid.
        futures = [self._pool.submit(self._service_bus, batch, t_tick, read_power)
                   for batch in batches.values()]
        records: list = []
        for fut in futures:
            try:
                records.extend(fut.result(timeout=BUS_TICK_TIMEOUT_S))
            except FutureTimeout:
                self.log.emit(
                    f"[timing] a CAN bus did not finish its tick within "
                    f"{BUS_TICK_TIMEOUT_S * 1000:.0f} ms - skipped this cycle")
            except Exception as e:
                self._note_comm_failure(f"{type(e).__name__}: {e}")
        return records

    def _service_bus(self, batch: list, t_tick: float,
                     read_power: bool) -> list:
        """Command one bus's motors. RUNS ON A POOL THREAD.

        Must not emit Qt signals or touch worker state shared across buses -
        results are returned for the caller to publish on the control thread.
        The per-device dict writes it does make (``_last_raw_pos`` inside
        ``_decalibrate``, the sweep setpoint on the motor's own target) are to
        keys owned exclusively by this bus, since CAN ids are unique across
        buses, so they cannot race another bus's thread.
        """
        records = []
        for device_id, target in batch:
            if self._bus is None or self._safety.estop or not target.enabled:
                continue
            try:
                status = self._command_motor(device_id, target, t_tick)
            except TransportError as e:
                records.append((device_id, target, None, None, str(e)))
                continue
            power = None
            if read_power:
                try:
                    power = self._read_power_values(device_id)
                except TransportError as e:
                    records.append((device_id, target, status, None, str(e)))
                    continue
            records.append((device_id, target, status, power, None))
        return records

    def _publish_records(self, records: list) -> None:
        """Turn one tick's results into signals and follow-up actions.

        Runs on the control thread, after every bus has finished, so signal
        order and the actions that can disable a motor (range cutout, hold
        recovery) stay single-threaded exactly as before.
        """
        for device_id, target, status, power, error in records:
            if error is not None:
                self._note_comm_failure(error)
                continue
            if status is not None:
                self.statusUpdated.emit(device_id, status)
                self._log_motor_faults(device_id, status)
                self._log_verbose_status(device_id, status)
                self._enforce_range_cutout(device_id, target, status.position)
                # After the range cutout (which may have just disabled it): if the
                # motor is still meant to be enabled but reports standby, its hold
                # dropped - re-enable it. Guarded by target.enabled so a motor the
                # cutout just tripped is not immediately re-energised.
                if target.enabled:
                    self._recover_dropped_hold(device_id, target, status)
            if power is not None:
                vbus, iq = power
                self._last_power[device_id] = (vbus, iq)
                self.powerUpdated.emit(
                    device_id, PowerInfo(device_id, vbus, iq, vbus * iq))
                self._log_power_change(device_id, vbus, iq)

    #: Human-readable labels for the fault bits in a status frame, in priority
    #: order. ``undervoltage``/``overcurrent`` come first: those are the ones
    #: that fire when a shared supply can't hold the current of an extra motor.
    _FAULT_LABELS = (
        ("undervoltage", "UNDERVOLTAGE (supply sagged below motor minimum)"),
        ("overcurrent", "OVERCURRENT (demanded more current than allowed)"),
        ("stalled", "STALLED (commanded torque but could not move)"),
        ("overtemperature", "OVERTEMPERATURE"),
        ("encoder_fault", "ENCODER FAULT"),
    )

    def _log_motor_faults(self, device_id: int, status: MotorStatus) -> None:
        """Log a motor's fault bits when they change, not every cycle.

        Every feedback frame carries the fault flags (undervoltage, overcurrent,
        stalled, ...). Logging them raw at 100 Hz would flood the UI, so we track
        the last-seen active set per motor and emit only on a rising edge (a new
        fault appears) or falling edge (all faults clear). The message includes
        live VBUS-adjacent context - torque and temperature from the same frame -
        so a power sag is visible in one line: e.g. a motor that sets
        ``undervoltage`` the instant a third motor is enabled is the direct CAN
        evidence that the supply, not the software, dropped the hold.
        """
        active = frozenset(
            name for name, _ in self._FAULT_LABELS if getattr(status, name)
        )
        if active == self._motor_faults.get(device_id, frozenset()):
            return
        self._motor_faults[device_id] = active
        if not active:
            self.log.emit(f"M{device_id}: faults cleared")
            return
        labels = ", ".join(
            label for name, label in self._FAULT_LABELS if name in active
        )
        self.error.emit(
            f"M{device_id}: FAULT {labels} "
            f"[torque={status.torque:+.2f} Nm, temp={status.temperature:.0f}C]")

    def _read_power_values(self, device_id: int) -> "Optional[tuple[float, float]]":
        """Read (VBUS, Iq) for one motor. RUNS ON A POOL THREAD.

        The IO half of the old ``_read_power``: plain READ_PARAMETER
        round-trips (never a motion frame), so they are safe to interleave with
        the control loop without disturbing the motor. Split out so the reads
        happen inside the parallel phase while the emitting stays on the
        control thread. Returns None if either register does not answer;
        TransportError propagates to the caller, which records it.
        """
        if self._bus is None:
            return None
        vbus = self._bus.read_param(device_id, ParameterType.VBUS)
        iq = self._bus.read_param(device_id, ParameterType.IQ_FILTERED)
        if vbus is None or iq is None:
            return None
        return (float(vbus), float(iq))

    def _read_power(self, device_id: int) -> None:
        """Read power for one motor and emit it immediately (control thread).

        Retained for callers outside the tick; the tick itself uses
        ``_read_power_values`` plus ``_publish_records``.
        """
        try:
            values = self._read_power_values(device_id)
        except TransportError as e:
            self._note_comm_failure(str(e))
            return
        if values is None:
            return
        vbus, iq = values
        self._last_power[device_id] = (vbus, iq)
        self.powerUpdated.emit(device_id, PowerInfo(device_id, vbus, iq, vbus * iq))
        self._log_power_change(device_id, vbus, iq)

    def _log_verbose_status(self, device_id: int, status: MotorStatus) -> None:
        """Emit a throttled, human-readable telemetry line to the log dock.

        One line per motor at ~2 Hz (see VERBOSE_LOG_DIVISOR) with position,
        board voltage/current (from the newest power read), torque, temperature
        and the active fault set - the "detailed log" you can watch and copy
        straight from the GUI while reproducing a hold drop. VBUS/Iq show ``---``
        until the first power read for that motor lands.
        """
        if self._loop_count % VERBOSE_LOG_DIVISOR != 0:
            return
        power = self._last_power.get(device_id)
        power_s = (f"VBUS {power[0]:5.1f}V Iq {power[1]:+5.1f}A"
                   if power is not None else "VBUS   ---  Iq   ---")
        faults = ",".join(
            name for name, _ in self._FAULT_LABELS if getattr(status, name)
        ) or "ok"
        self.log.emit(
            f"M{device_id}: pos {math.degrees(status.position):+7.1f}deg  "
            f"{power_s}  torque {status.torque:+6.2f}Nm  "
            f"{status.temperature:4.0f}C  [{faults}]")

    def _log_power_change(self, device_id: int, vbus: float, iq: float) -> None:
        """Log VBUS/Iq only when the bus voltage moves at least VBUS_LOG_DELTA_V
        from the last logged value, so a supply sag when another motor is enabled
        shows up in the log timeline (dip and recovery) without flooding it. The
        first reading just seeds the baseline silently."""
        prev = self._last_logged_vbus.get(device_id)
        self._last_logged_vbus[device_id] = vbus
        if prev is None or abs(vbus - prev) < VBUS_LOG_DELTA_V:
            if prev is not None:
                # Not a significant move: keep the baseline where it was so small
                # drifts accumulate against a fixed reference instead of ratcheting.
                self._last_logged_vbus[device_id] = prev
            return
        self.log.emit(
            f"M{device_id}: VBUS {prev:.1f} -> {vbus:.1f} V, Iq {iq:+.1f} A")

    def _enforce_range_cutout(self, device_id: int, target: MotorTarget,
                              user_pos: float) -> None:
        """Disable the motor if live feedback leaves the calibrated range.

        Position/MIT setpoints are already clamped to the range, but velocity and
        current modes have NO position clamp - this feedback-driven cutout is the
        only thing that stops the shaft driving through an end-stop in those
        modes. Skipped while range-calibrating, since that deliberately moves the
        shaft past the old bounds to redefine them.
        """
        if device_id in self._range_cal:
            return
        lo, hi = self._range.get(device_id, (None, None))
        if lo is None and hi is None:
            return
        m = RANGE_TRIP_MARGIN_RAD
        if (lo is not None and user_pos < lo - m) or \
           (hi is not None and user_pos > hi + m):
            try:
                self._bus.disable(device_id)
            except Exception:
                pass
            target.enabled = False
            self._stop_sweep(device_id, target)
            self.motorEnabledChanged.emit(device_id, False)
            self.error.emit(
                f"M{device_id}: position {math.degrees(user_pos):+.1f} deg left "
                "the calibrated range - motor disabled for safety")

    def _command_motor(self, device_id: int, t: MotorTarget,
                       now: Optional[float] = None) -> Optional[MotorStatus]:
        """Clamp the target in the *user* frame, convert to the *raw* motor frame
        via the motor's calibration, send it, and de-calibrate the reply back to
        the user frame for display."""
        if now is None:
            now = time.monotonic()
        s = self._safety
        c = self._calib.get(device_id) or Calibration()
        # A running sweep drives the position setpoint itself, but only in a
        # position mode - in velocity/current/MIT the position field is unused,
        # so leave it alone. Writing t.position keeps the readout/graph honest.
        if t.sweep_enabled and t.mode in (RunMode.POSITION_PP, RunMode.POSITION_CSP):
            t.position = self._sweep_position(t, now)
        if t.mode == RunMode.MIT:
            raw = self._bus.operation(
                device_id,
                c.pos_to_raw(self._clamp_to_range(
                    device_id, s.clamp_position(t.position))),
                c.signed_to_raw(s.clamp_velocity(t.velocity)),
                s.clamp_kp(t.kp),
                s.clamp_kd(t.kd),
                s.clamp_torque(t.torque_ff),
            )
            return self._decalibrate(device_id, raw, c)
        if t.mode in (RunMode.POSITION_PP, RunMode.POSITION_CSP):
            status = self._bus.set_position(
                device_id, c.pos_to_raw(self._clamp_to_range(
                    device_id, s.clamp_position(t.position))),
                s.limits.velocity_max)
        elif t.mode == RunMode.VELOCITY:
            status = self._bus.set_velocity(
                device_id, c.signed_to_raw(s.clamp_velocity(t.velocity)),
                s.limits.current_max)
        elif t.mode == RunMode.CURRENT:
            status = self._bus.set_current(
                device_id, c.signed_to_raw(s.clamp_current(t.current)))
        else:
            status = None
        # Feedback comes from the status frame the setpoint write is acked with.
        # Do NOT call poll_status here: it sends an MIT operation_control frame
        # which, in position/velocity/current mode, commands zero and brakes the
        # motor every control cycle - the motor would twitch and never sustain
        # motion. (This was the root cause of "motors don't rotate".)
        return self._decalibrate(device_id, status, c)

    def _decalibrate(self, device_id: int, status: Optional[MotorStatus],
                     c: Calibration) -> Optional[MotorStatus]:
        """Record the raw position (for capture-zero) and convert feedback to the
        user frame for display."""
        if status is None:
            return None
        self._last_raw_pos[device_id] = status.position
        user_pos = c.pos_from_raw(status.position)
        # While range-calibrating, every live sample widens the captured min/max.
        self._note_range_sample(device_id, user_pos)
        return replace(
            status,
            position=user_pos,
            velocity=c.signed_from_raw(status.velocity),
            torque=c.signed_from_raw(status.torque),
        )