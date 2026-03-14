"""Unified gamepad support via evdev + Xbox One USB (pyusb) fallback.

Provides :class:`GamepadState` which handles device discovery, background
USB reading, and polling.  Both ``play_nes.py`` and ``collect.py`` use
this instead of duplicating controller logic.
"""
from __future__ import annotations

import select
import struct
import threading
from dataclasses import dataclass

try:
    import evdev
except ImportError:
    evdev = None  # type: ignore[assignment]

try:
    import usb.core
    import usb.util
    _HAS_PYUSB = True
except ImportError:
    _HAS_PYUSB = False

# -- Xbox One GIP protocol constants --
_XB1_VIDS_PIDS = [
    (0x24C6, 0x581A),  # ThrustMaster XB1 Classic Controller
    (0x045E, 0x02D1),  # Microsoft Xbox One Controller
    (0x045E, 0x02DD),  # Microsoft Xbox One Controller (firmware 2015)
    (0x045E, 0x02E3),  # Microsoft Xbox Elite Controller
    (0x045E, 0x02EA),  # Microsoft Xbox One S Controller
    (0x045E, 0x0B00),  # Microsoft Xbox Elite Series 2
    (0x045E, 0x0B12),  # Microsoft Xbox Series X|S Controller
    (0x0738, 0x4A01),  # Mad Catz FightStick TE 2
    (0x0E6F, 0x0139),  # PDP Afterglow Prismatic
    (0x0E6F, 0x013A),  # PDP Xbox One Controller
    (0x24C6, 0x541A),  # PowerA Xbox One
    (0x24C6, 0x542A),  # PowerA Xbox One Spectra
    (0x24C6, 0x543A),  # PowerA Xbox One Mini
]
_GIP_INIT = bytes([0x05, 0x20, 0x00, 0x01, 0x00])
_GIP_INPUT_TYPE = 0x20


@dataclass
class ButtonState:
    """Snapshot of gamepad directional + button state."""
    right: bool = False
    left: bool = False
    up: bool = False
    down: bool = False
    a_btn: bool = False
    b_btn: bool = False
    start: bool = False
    select: bool = False


class GamepadState:
    """Manages evdev / USB gamepad discovery, polling, and state reading.

    Call :meth:`poll` each frame, then :meth:`read_buttons` to get the
    current directional + button state from the gamepad (keyboard
    state is handled by the caller).

    Parameters
    ----------
    extended_button_codes : bool
        If *True*, A/B/Start/Select map to the wider set of evdev codes
        used in ``play_nes.py`` (BTN_SOUTH, BTN_WEST, BTN_EAST, BTN_START,
        BTN_SELECT, etc.).  If *False* (default), only BTN_THUMB → A and
        BTN_TRIGGER → B are recognised (the mapping used in ``collect.py``).
    """

    def __init__(self, *, extended_button_codes: bool = False):
        self._evdev_device = None
        self._usb_device = None
        self._evdev_axes: dict[str, float] = {"ABS_X": 128, "ABS_Y": 128}
        self._evdev_buttons: set[int] = set()
        self._evdev_buttons_transient: set[int] = set()  # presses since last read
        self._evdev_axis_ranges: dict[str, tuple[int, int]] = {}
        self._usb_lock: threading.Lock | None = None
        self._extended = extended_button_codes
        self._init_device()

    # ------------------------------------------------------------------
    # Device discovery
    # ------------------------------------------------------------------

    def _init_device(self) -> None:
        if evdev is None:
            self._init_usb_gamepad()
            return
        try:
            for path in evdev.list_devices():
                try:
                    dev = evdev.InputDevice(path)
                except PermissionError:
                    print(f"Permission denied: {path}")
                    print(f"  Fix: sudo chmod 666 {path}")
                    print(f"  Permanent: sudo usermod -aG input $(whoami)  (then restart WSL)")
                    continue
                caps = dev.capabilities(verbose=False)
                has_abs = evdev.ecodes.EV_ABS in caps
                has_key = evdev.ecodes.EV_KEY in caps
                if has_abs and has_key:
                    self._evdev_device = dev
                    for abs_code, abs_info in caps.get(evdev.ecodes.EV_ABS, []):
                        name = evdev.ecodes.ABS.get(abs_code, f"ABS_{abs_code}")
                        if isinstance(name, list):
                            name = name[0]
                        mid = (abs_info.min + abs_info.max) // 2
                        self._evdev_axes[name] = mid
                        self._evdev_axis_ranges[name] = (abs_info.min, abs_info.max)
                    print(f"Initialized evdev gamepad: {dev.name} ({dev.path})")
                    print(f"  Axes: {list(self._evdev_axis_ranges.keys())}")
                    return
        except Exception as exc:
            print(f"evdev gamepad probe failed: {exc}")
        if self._init_usb_gamepad():
            return
        print("No gamepad found via evdev or USB. Falling back to keyboard.")
        print("  Hint: ensure /dev/input/event* is readable (sudo chmod 666 /dev/input/event*)")
        print("  For Xbox One controllers in WSL2: pip install pyusb && run with sudo")

    def _init_usb_gamepad(self) -> bool:
        """Try to open an Xbox One controller via pyusb (WSL2 fallback)."""
        if not _HAS_PYUSB:
            return False
        for vid, pid in _XB1_VIDS_PIDS:
            dev = usb.core.find(idVendor=vid, idProduct=pid)
            if dev is not None:
                try:
                    if dev.is_kernel_driver_active(0):
                        dev.detach_kernel_driver(0)
                    dev.set_configuration()
                    dev.write(0x02, _GIP_INIT)
                    self._usb_device = dev
                    self._evdev_axis_ranges = {
                        "ABS_X": (-32768, 32767),
                        "ABS_Y": (-32768, 32767),
                        "ABS_HAT0X": (-1, 1),
                        "ABS_HAT0Y": (-1, 1),
                    }
                    self._evdev_axes = {
                        "ABS_X": 0, "ABS_Y": 0,
                        "ABS_HAT0X": 0, "ABS_HAT0Y": 0,
                    }
                    self._usb_lock = threading.Lock()
                    thread = threading.Thread(
                        target=self._usb_reader_loop, daemon=True,
                    )
                    thread.start()
                    print(f"Initialized Xbox One controller via USB ({vid:04x}:{pid:04x})")
                    return True
                except Exception as exc:
                    print(f"USB gamepad init failed for {vid:04x}:{pid:04x}: {exc}")
        return False

    # ------------------------------------------------------------------
    # USB background reader
    # ------------------------------------------------------------------

    def _usb_reader_loop(self) -> None:
        """Background thread: continuously read Xbox One GIP reports."""
        dev = self._usb_device
        while dev is not None and self._usb_device is not None:
            try:
                data = dev.read(0x81, 64, timeout=100)
            except usb.core.USBTimeoutError:
                continue
            except (OSError, IOError, usb.core.USBError):
                print("USB gamepad disconnected!")
                self._usb_device = None
                return
            if len(data) < 18 or data[0] != _GIP_INPUT_TYPE:
                continue
            btn1, btn2 = data[4], data[5]
            lx, ly = struct.unpack_from("<hh", data, 10)
            hat_x = (1 if btn2 & 0x08 else 0) - (1 if btn2 & 0x04 else 0)
            hat_y = (1 if btn2 & 0x02 else 0) - (1 if btn2 & 0x01 else 0)
            with self._usb_lock:
                self._evdev_axes["ABS_X"] = lx
                self._evdev_axes["ABS_Y"] = -ly  # invert Y (GIP: up=+, evdev: up=-)
                self._evdev_axes["ABS_HAT0X"] = hat_x
                self._evdev_axes["ABS_HAT0Y"] = hat_y
                if btn1 & 0x10:
                    self._evdev_buttons.add(evdev.ecodes.BTN_THUMB)
                    self._evdev_buttons_transient.add(evdev.ecodes.BTN_THUMB)
                else:
                    self._evdev_buttons.discard(evdev.ecodes.BTN_THUMB)
                if btn1 & 0x40 or btn1 & 0x20:
                    self._evdev_buttons.add(evdev.ecodes.BTN_TRIGGER)
                    self._evdev_buttons_transient.add(evdev.ecodes.BTN_TRIGGER)
                else:
                    self._evdev_buttons.discard(evdev.ecodes.BTN_TRIGGER)
                if btn1 & 0x04:  # Menu (Start)
                    self._evdev_buttons.add(getattr(evdev.ecodes, 'BTN_START', 315))
                    self._evdev_buttons_transient.add(getattr(evdev.ecodes, 'BTN_START', 315))
                else:
                    self._evdev_buttons.discard(getattr(evdev.ecodes, 'BTN_START', 315))
                if btn1 & 0x08:  # View (Select)
                    self._evdev_buttons.add(getattr(evdev.ecodes, 'BTN_SELECT', 314))
                    self._evdev_buttons_transient.add(getattr(evdev.ecodes, 'BTN_SELECT', 314))
                else:
                    self._evdev_buttons.discard(getattr(evdev.ecodes, 'BTN_SELECT', 314))

    # ------------------------------------------------------------------
    # Polling
    # ------------------------------------------------------------------

    def poll(self) -> None:
        """Read any pending evdev events (no-op for USB, which uses a thread)."""
        dev = self._evdev_device
        if dev is None:
            return
        try:
            while select.select([dev.fd], [], [], 0)[0]:
                for event in dev.read():
                    if event.type == evdev.ecodes.EV_ABS:
                        name = evdev.ecodes.ABS.get(event.code, f"ABS_{event.code}")
                        if isinstance(name, list):
                            name = name[0]
                        self._evdev_axes[name] = event.value
                    elif event.type == evdev.ecodes.EV_KEY:
                        if event.value >= 1:
                            self._evdev_buttons.add(event.code)
                            self._evdev_buttons_transient.add(event.code)
                        else:
                            self._evdev_buttons.discard(event.code)
        except (OSError, IOError):
            print("Gamepad disconnected!")
            self._evdev_device = None

    # ------------------------------------------------------------------
    # State reading
    # ------------------------------------------------------------------

    @property
    def connected(self) -> bool:
        return self._evdev_device is not None or self._usb_device is not None

    def _norm(self, axis: str) -> float:
        lo, hi = self._evdev_axis_ranges.get(axis, (0, 255))
        mid = (lo + hi) / 2.0
        half = max((hi - lo) / 2.0, 1)
        return (self._evdev_axes.get(axis, mid) - mid) / half

    def read_buttons(self) -> ButtonState:
        """Return the current gamepad button / directional state.

        Automatically acquires the USB lock if needed.  Call :meth:`poll`
        before this each frame.
        """
        if not self.connected:
            return ButtonState()

        lock = self._usb_lock
        if lock is not None:
            lock.acquire()
        try:
            x = self._norm("ABS_X")
            y = self._norm("ABS_Y")
            right = x > 0.5
            left = x < -0.5
            up = y < -0.5
            down = y > 0.5

            hx = self._norm("ABS_HAT0X")
            hy = self._norm("ABS_HAT0Y")
            right = right or hx > 0.5
            left = left or hx < -0.5
            up = up or hy < -0.5
            down = down or hy > 0.5

            btns = self._evdev_buttons | self._evdev_buttons_transient
            self._evdev_buttons_transient.clear()

            if self._extended:
                a_btn = any(b in btns for b in [
                    getattr(evdev.ecodes, 'BTN_SOUTH', 304),
                    getattr(evdev.ecodes, 'BTN_THUMB', 289),
                ])
                b_btn = any(b in btns for b in [
                    getattr(evdev.ecodes, 'BTN_WEST', 308),
                    getattr(evdev.ecodes, 'BTN_EAST', 305),
                    getattr(evdev.ecodes, 'BTN_TRIGGER', 288),
                ])
                start = any(b in btns for b in [
                    getattr(evdev.ecodes, 'BTN_START', 315),
                    297,  # BTN_BASE4 on generic USB gamepads
                ])
                sel = any(b in btns for b in [
                    getattr(evdev.ecodes, 'BTN_SELECT', 314),
                    296,  # BTN_BASE3 on generic USB gamepads
                ])
            else:
                a_btn = evdev.ecodes.BTN_THUMB in btns
                b_btn = evdev.ecodes.BTN_TRIGGER in btns
                start = False
                sel = False
        finally:
            if lock is not None:
                lock.release()

        return ButtonState(
            right=right, left=left, up=up, down=down,
            a_btn=a_btn, b_btn=b_btn, start=start, select=sel,
        )
