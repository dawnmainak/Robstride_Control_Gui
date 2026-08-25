"""Route motor commands to whichever CAN bus owns the motor.

The worker was written against a single :class:`RobstrideBus`. Supporting a
second adapter by hand would mean touching every one of its ~38
``self._bus.<method>(device_id, ...)`` call sites - a large, hard-to-review
change with plenty of room to miss one.

Instead this class presents the *same interface* as ``RobstrideBus`` and
forwards each call to the bus that owns the target motor, so the worker keeps
talking to ``self._bus`` exactly as before and needs no per-call-site edits.

This assumes **CAN ids are unique across every bus** - id 3 lives on exactly one
adapter. That is what keeps the whole stack (the worker's ten device-keyed
dicts, its Qt signals, the saved calibrations, the UI tabs) keyed on a plain
int. Reusing ids across buses would require a composite key everywhere and is
deliberately not supported; assign each motor a distinct id with "Set ID...".

Routing only. Every bus is still serviced by the one worker thread, so motors
on different buses are still commanded one after another - concurrency comes
later. With a single bus registered, behaviour is identical to before.
"""

from __future__ import annotations

import logging
from typing import Optional

from .bus import Motor, RobstrideBus

logger = logging.getLogger(__name__)

#: Methods whose FIRST positional argument is the target ``device_id``. These
#: are forwarded verbatim to the owning bus by :meth:`BusRouter.__getattr__`.
#: Kept as an explicit allowlist rather than blind delegation so a typo raises
#: AttributeError instead of silently reaching an unrouted method - and so the
#: set of things that route by device is auditable in one place.
_DEVICE_ROUTED = frozenset({
    "enable", "disable", "set_zero", "set_run_mode", "operation",
    "set_position", "set_velocity", "set_current", "poll_status",
    "read_param", "write_param", "read_zero_state", "ping",
    "identify", "count_responders", "set_motor_id",
})


class BusRouter:
    """A ``RobstrideBus``-shaped facade over one or more real buses."""

    def __init__(self):
        #: bus name (e.g. "can0") -> the real bus behind it
        self.buses: dict[str, RobstrideBus] = {}
        #: device_id -> owning bus, built as motors are registered
        self._owner: dict[int, RobstrideBus] = {}

    # -- composition -------------------------------------------------------------

    def add_bus(self, name: str, bus: RobstrideBus) -> None:
        """Register a bus under ``name``. Re-adding a name replaces it."""
        self.buses[name] = bus

    def bus_for(self, device_id: int) -> RobstrideBus:
        """The bus owning ``device_id``.

        Falls back to the only bus when just one is registered, so a
        single-adapter setup behaves exactly as it did before this class
        existed - including for a motor discovered by Scan that has not been
        explicitly registered yet.
        """
        owner = self._owner.get(device_id)
        if owner is not None:
            return owner
        if len(self.buses) == 1:
            return next(iter(self.buses.values()))
        raise KeyError(
            f"motor {device_id} is not registered on any bus - add it with "
            f"add_motor(motor, bus_name=...) before commanding it")

    def add_motor(self, motor: Motor, bus_name: Optional[str] = None) -> None:
        """Register ``motor`` and record which bus owns it.

        ``bus_name`` may be omitted when exactly one bus exists. Registering an
        id that already belongs to a *different* bus is a configuration error
        (duplicate ids across buses), so it is logged loudly rather than
        silently rerouting the motor.
        """
        if bus_name is None:
            if len(self.buses) != 1:
                raise ValueError(
                    f"bus_name is required when {len(self.buses)} buses are "
                    f"registered")
            bus_name = next(iter(self.buses))
        bus = self.buses[bus_name]
        previous = self._owner.get(motor.device_id)
        if previous is not None and previous is not bus:
            logger.warning(
                "motor id %d is claimed by two buses - CAN ids must be unique "
                "across every adapter; keeping the newer one (%s)",
                motor.device_id, bus_name)
        self._owner[motor.device_id] = bus
        bus.add_motor(motor)

    # -- RobstrideBus-shaped surface ---------------------------------------------

    @property
    def motors(self) -> dict[int, Motor]:
        """Every motor across every bus, merged. Ids are unique, so no clash."""
        merged: dict[int, Motor] = {}
        for bus in self.buses.values():
            merged.update(bus.motors)
        return merged

    @property
    def is_open(self) -> bool:
        """True when at least one bus is open.

        Deliberately ``any`` rather than ``all``: one failed adapter must not
        stop the worker servicing the motors on a healthy one.
        """
        return any(bus.is_open for bus in self.buses.values())

    def open(self) -> None:
        """Open every bus, collecting failures so one bad adapter does not
        prevent the others from coming up. Raises only if *all* of them fail."""
        errors = []
        for name, bus in self.buses.items():
            try:
                bus.open()
            except Exception as e:
                errors.append(f"{name}: {e}")
        if errors and not self.is_open:
            raise RuntimeError("; ".join(errors))
        for err in errors:
            logger.warning("bus failed to open - %s", err)

    def close(self) -> None:
        for bus in self.buses.values():
            try:
                bus.close()
            except Exception:
                pass

    def model_of(self, device_id: int) -> str:
        return self.bus_for(device_id).model_of(device_id)

    # -- fan-out operations ------------------------------------------------------

    def scan(self, start: int = 1, end: int = 16) -> list[int]:
        """Scan every bus and return the union of responding ids, sorted.

        Also records which bus each id answered on, so a motor found by Scan is
        routable immediately without a separate registration step.
        """
        found: list[int] = []
        for bus in self.buses.values():
            for device_id in bus.scan(start, end):
                self._owner.setdefault(device_id, bus)
                if device_id not in found:
                    found.append(device_id)
        return sorted(found)

    def inventory(self, start: int = 1, end: int = 16) -> list[tuple[int, list[bytes]]]:
        items: list[tuple[int, list[bytes]]] = []
        for bus in self.buses.values():
            for device_id, uids in bus.inventory(start, end):
                self._owner.setdefault(device_id, bus)
                items.append((device_id, uids))
        return items

    def find_collisions(self, device_ids: list[int]) -> list[int]:
        """Ids answered by more than one motor.

        Checked per owning bus: two motors sharing an id on ONE bus is the
        collision this detects. The same id on two different buses is a
        configuration error caught by :meth:`add_motor` instead.
        """
        return [did for did in device_ids
                if self.bus_for(did).count_responders(did) > 1]

    # -- device-routed delegation -------------------------------------------------

    def __getattr__(self, name: str):
        """Forward an allowlisted device-routed method to the owning bus.

        Only reached for attributes not defined above, so the explicit methods
        keep priority. ``_DEVICE_ROUTED`` gates it so an unknown name fails
        loudly rather than being silently swallowed.
        """
        if name not in _DEVICE_ROUTED:
            raise AttributeError(
                f"{type(self).__name__!r} object has no attribute {name!r}")

        def call(device_id: int, *args, **kwargs):
            return getattr(self.bus_for(device_id), name)(device_id, *args, **kwargs)

        return call