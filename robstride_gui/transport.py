"""Wire transports for the RobStride protocol.

Two interchangeable backends implement the same :class:`Transport` interface so
the rest of the app never cares how frames reach the motor:

* :class:`SerialATTransport` - USB-CAN adapter in AT mode (``/dev/ttyUSB*``).
  This is the path proven by the vendor ``motor_zero.py`` on this hardware.
* :class:`SocketCANTransport` - a kernel CAN interface (``can0``) via
  ``python-can``. More "standard" but needs the interface brought up first.

Both are import-safe even when their optional dependency is missing; the error
only surfaces when you actually try to :meth:`open` that backend.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from . import protocol as proto
from .protocol import Frame

logger = logging.getLogger(__name__)

#: substrings (case-insensitive) seen in the description/hwid of the USB-serial
#: chips commonly used by USB-CAN adapters. Used only to *rank* candidates - a
#: non-matching port is still offered, just lower in the list.
USB_CAN_HINTS = ("can", "ch340", "ch341", "cp210", "ftdi", "ft232",
                 "usb-serial", "usb2.0-serial", "1a86:", "0403:", "10c4:")


class TransportError(RuntimeError):
    """Raised for connection / IO failures in a transport."""


class Transport(ABC):
    """Abstract send/receive channel for protocol :class:`Frame` objects."""

    #: human label shown in the UI
    name: str = "transport"

    @abstractmethod
    def open(self) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    @property
    @abstractmethod
    def is_open(self) -> bool: ...

    @abstractmethod
    def send(self, frame: Frame) -> None: ...

    @abstractmethod
    def recv(self, timeout: float = 0.1) -> Optional[Frame]:
        """Return the next received :class:`Frame`, or ``None`` on timeout."""

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *exc):
        self.close()


# --- serial "AT" transport -------------------------------------------------------


class SerialATTransport(Transport):
    """USB-CAN adapter speaking the AT framing over a serial port."""

    name = "Serial (AT)"

    def __init__(self, port: str, baud: int = proto.DEFAULT_SERIAL_BAUD,
                 read_timeout: float = 0.02):
        self.port = port
        self.baud = baud
        self.read_timeout = read_timeout
        self._ser = None
        self._rx = b""          # leftover bytes from partial frames
        self._pending: list[Frame] = []  # fully parsed frames awaiting recv()

    def open(self) -> None:
        try:
            import serial  # pyserial
        except ImportError as e:  # pragma: no cover - dependency guard
            raise TransportError("pyserial is not installed (pip install pyserial)") from e
        try:
            # exclusive=True (POSIX) takes an advisory lock so a second process
            # cannot also open this port. Without it a stale GUI instance keeps
            # the port open and silently consumes the motor's reply frames, so
            # Detect reports "0 responding" with nothing actually wrong. Better
            # to fail loudly here than to race two readers on one adapter.
            self._ser = serial.Serial(self.port, self.baud,
                                      timeout=self.read_timeout, exclusive=True)
        except Exception as e:
            raise TransportError(f"Cannot open serial port {self.port}: {e}") from e
        time.sleep(0.2)
        self._rx = b""
        self._pending = []
        # Do NOT send an "AT+AT" handshake: on the official RobStride USB-CAN
        # module that command switches it into AT *command* mode, which stops
        # it forwarding CAN frames (the motor then never replies). The module
        # powers up ready to pass "AT"-framed CAN frames through, exactly like
        # the vendor GUI, which sends no handshake. Just clear any stale bytes.
        try:
            self._ser.reset_input_buffer()
        except Exception:
            pass

    def close(self) -> None:
        if self._ser is not None:
            try:
                self._ser.close()
            finally:
                self._ser = None

    @property
    def is_open(self) -> bool:
        return self._ser is not None and getattr(self._ser, "is_open", False)

    def send(self, frame: Frame) -> None:
        if self._ser is None:
            raise TransportError("serial transport is not open")
        try:
            self._ser.write(proto.encode_at(frame))
        except Exception as e:
            raise TransportError(f"serial write failed: {e}") from e

    def recv(self, timeout: float = 0.1) -> Optional[Frame]:
        if self._ser is None:
            raise TransportError("serial transport is not open")
        deadline = time.monotonic() + timeout
        while True:
            if self._pending:
                return self._pending.pop(0)
            frames, self._rx = proto.decode_at(self._rx)
            if frames:
                self._pending.extend(frames[1:])
                return frames[0]
            try:
                waiting = self._ser.in_waiting
                chunk = self._ser.read(waiting or 1)
            except Exception as e:
                # CH340 USB-serial adapters intermittently raise "device reports
                # readiness to read but returned no data" while the motor draws
                # current. It is a transient glitch, not a disconnect: retry
                # within the deadline instead of tearing down the connection.
                if "returned no data" in str(e) and time.monotonic() < deadline:
                    time.sleep(0.001)
                    continue
                raise TransportError(f"serial read failed: {e}") from e
            if chunk:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("recv raw RX %dB: %s", len(chunk), chunk.hex())
                self._rx += chunk
            elif time.monotonic() >= deadline:
                return None


# --- SocketCAN transport ---------------------------------------------------------


class SocketCANTransport(Transport):
    """Kernel CAN interface via ``python-can``."""

    name = "SocketCAN"

    def __init__(self, channel: str = "can0", bitrate: int = proto.DEFAULT_CAN_BITRATE):
        self.channel = channel
        self.bitrate = bitrate
        self._bus = None

    #: sysfs IFF_UP bit. A CAN interface can exist and be bindable while
    #: administratively DOWN, so this flag - not mere existence - is what says
    #: the kernel will actually transmit on it.
    _IFF_UP = 0x1

    def _interface_state(self) -> tuple[bool, bool]:
        """Return ``(exists, is_up)`` for ``self.channel``, read from sysfs.

        python-can binds happily to a down CAN interface and does NOT bring it
        up (the ``bitrate=`` argument is ignored on Linux - only ``ip link``
        sets it). So open() appears to succeed and then every send() fails with
        ENETDOWN, "Failed to transmit: Network is down". Checking the flag here
        turns that into one actionable error at connect time.

        If the state cannot be determined (no sysfs, e.g. a non-Linux host),
        report ``(True, True)`` so we never block a connection we cannot verify.
        """
        from pathlib import Path

        iface = Path("/sys/class/net") / self.channel
        if not iface.is_dir():
            return (False, False)
        try:
            flags = int((iface / "flags").read_text().strip(), 16)
        except (OSError, ValueError):
            return (True, True)   # unknown - do not stand in the way
        return (True, bool(flags & self._IFF_UP))

    def open(self) -> None:
        try:
            import can
        except ImportError as e:  # pragma: no cover - dependency guard
            raise TransportError("python-can is not installed (pip install python-can)") from e
        # Fail here, loudly, rather than letting every later send() raise
        # ENETDOWN one frame at a time (see _interface_state).
        exists, is_up = self._interface_state()
        if not exists:
            raise TransportError(
                f"SocketCAN interface '{self.channel}' does not exist.\n"
                f"  Load the driver and create the interface:\n"
                f"    sudo modprobe can can_raw\n"
                f"    sudo ip link set {self.channel} up type can "
                f"bitrate {self.bitrate}\n"
                f"  If your adapter is the serial USB-CAN dongle, switch the\n"
                f"  Transport dropdown to 'Serial (AT)' instead.")
        if not is_up:
            raise TransportError(
                f"SocketCAN interface '{self.channel}' exists but is DOWN.\n"
                f"  Every transmit would fail with 'Network is down'. Bring it up:\n"
                f"    sudo ip link set {self.channel} up type can "
                f"bitrate {self.bitrate}\n"
                f"  Then confirm the bitrate matches the motor "
                f"(RobStride default 1 Mbit/s):\n"
                f"    ip -details link show {self.channel}")
        try:
            self._bus = can.interface.Bus(interface="socketcan",
                                          channel=self.channel, bitrate=self.bitrate)
        except Exception as e:
            raise TransportError(
                f"Cannot open SocketCAN '{self.channel}': {e}\n"
                f"  Bring the interface up first, e.g.:\n"
                f"    sudo ip link set {self.channel} up type can bitrate {self.bitrate}"
            ) from e

    def close(self) -> None:
        if self._bus is not None:
            try:
                self._bus.shutdown()
            finally:
                self._bus = None

    @property
    def is_open(self) -> bool:
        return self._bus is not None

    def send(self, frame: Frame) -> None:
        if self._bus is None:
            raise TransportError("socketcan transport is not open")
        import can
        msg = can.Message(arbitration_id=frame.ext_id, is_extended_id=True,
                          dlc=len(frame.data), data=frame.data)
        try:
            self._bus.send(msg)
        except Exception as e:
            raise TransportError(f"socketcan send failed: {e}") from e

    def recv(self, timeout: float = 0.1) -> Optional[Frame]:
        if self._bus is None:
            raise TransportError("socketcan transport is not open")
        try:
            msg = self._bus.recv(timeout=timeout)
        except Exception as e:
            raise TransportError(f"socketcan recv failed: {e}") from e
        if msg is None or not msg.is_extended_id:
            return None
        comm_type, extra_data, device_id = proto.split_ext_id(msg.arbitration_id)
        return Frame(comm_type, extra_data, device_id, bytes(msg.data))


#: ARPHRD_CAN - the sysfs ``type`` value that marks a network interface as CAN.
#: Distinguishes a real CAN port from an ethernet/loopback device whose name
#: happens to start with "can".
_ARPHRD_CAN = 280

#: Offered when no CAN interface can be enumerated (non-Linux host, or the hub's
#: driver not loaded yet). Covers the RobStride CAN hub, which exposes can0..can4.
FALLBACK_CAN_CHANNELS = ("can0", "can1", "can2", "can3", "can4")


def list_can_interfaces() -> list[str]:
    """CAN interfaces present on this machine, naturally sorted.

    Read from sysfs rather than hardcoded, so every port a multi-bus adapter
    exposes shows up - a RobStride CAN hub presents can0..can4, and a fixed
    two-entry list silently hides three of them. Sorted numerically so can10
    follows can9 rather than can1.

    Returns :data:`FALLBACK_CAN_CHANNELS` when nothing can be enumerated, so the
    dropdown is never empty on a host without sysfs.
    """
    from pathlib import Path

    root = Path("/sys/class/net")
    if not root.is_dir():
        return list(FALLBACK_CAN_CHANNELS)
    found = []
    for iface in root.iterdir():
        try:
            if int((iface / "type").read_text().strip()) != _ARPHRD_CAN:
                continue
        except (OSError, ValueError):
            continue
        found.append(iface.name)
    if not found:
        return list(FALLBACK_CAN_CHANNELS)
    return sorted(found, key=lambda n: (len(n), n))


def can_interface_is_up(channel: str) -> bool:
    """True when ``channel`` is administratively UP (see SocketCANTransport)."""
    return SocketCANTransport(channel)._interface_state()[1]


def _is_usb_serial(device: str, hwid: str) -> bool:
    """True for a hot-plugged USB serial device, False for built-in ttyS* ports.

    Legacy 8250/16550 motherboard ports enumerate as ``/dev/ttyS*`` with an
    ``n/a`` hwid even when nothing is attached; the USB-CAN adapter shows up as
    ``ttyUSB*``/``ttyACM*`` (Linux) or ``cu.usb*`` (macOS) with a real vid:pid.
    """
    dev = device.lower()
    if any(tok in dev for tok in ("ttyusb", "ttyacm", "usbmodem", "usbserial")):
        return True
    hw = hwid.lower()
    return bool(hw) and hw != "n/a" and "usb" in hw


@dataclass(frozen=True)
class SerialPortInfo:
    """One detected serial port and how likely it is to be the USB-CAN adapter."""

    device: str            # e.g. "/dev/ttyUSB0"
    description: str        # human label from the OS, "" if unknown
    hwid: str              # vid:pid / serial-number string, "" if unknown
    is_usb: bool          # a real hot-plugged USB device (not a built-in ttyS*)
    is_likely_can: bool   # matched a USB_CAN_HINTS substring

    def label(self) -> str:
        """Combined label for a UI dropdown, e.g. '/dev/ttyUSB0 - USB Serial'."""
        if self.description and self.description.lower() != "n/a":
            return f"{self.device} - {self.description}"
        return self.device


def list_serial_port_details(usb_only: bool = True) -> list[SerialPortInfo]:
    """Detected serial ports, best (most CAN-like) first.

    By default only real USB serial devices are returned, so the dozens of
    built-in ``/dev/ttyS*`` placeholders do not drown out the actual adapter.
    Pass ``usb_only=False`` to include every enumerated port. An empty list
    means "no port available".
    """
    try:
        from serial.tools import list_ports
    except ImportError:
        return []

    infos: list[SerialPortInfo] = []
    for p in list_ports.comports():
        hwid = p.hwid or ""
        is_usb = _is_usb_serial(p.device, hwid)
        if usb_only and not is_usb:
            continue
        haystack = f"{p.description or ''} {hwid}".lower()
        infos.append(SerialPortInfo(
            device=p.device,
            description=p.description or "",
            hwid=hwid,
            is_usb=is_usb,
            is_likely_can=any(hint in haystack for hint in USB_CAN_HINTS),
        ))
    # CAN-like first, then other USB devices, then a stable order by device name.
    infos.sort(key=lambda i: (not i.is_likely_can, not i.is_usb, i.device))
    return infos


def list_serial_ports(usb_only: bool = True) -> list[str]:
    """Best-effort list of candidate serial port device names (CAN-like first)."""
    return [info.device for info in list_serial_port_details(usb_only=usb_only)]


def auto_detect_serial_port() -> Optional[str]:
    """Return the most likely USB-CAN serial port, or ``None`` if none found.

    ``None`` is the honest "no port available" answer - callers should surface
    that rather than guessing a default device that does not exist.
    """
    details = list_serial_port_details(usb_only=True)
    return details[0].device if details else None