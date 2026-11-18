"""Windows client-area capture and focus-gated mouse input."""

import ctypes as C
import os
import time
from concurrent.futures import ThreadPoolExecutor
from ctypes import wintypes as W
from dataclasses import replace
from types import SimpleNamespace

import cv2
import mss
import numpy as np
import pywintypes
import win32api
import win32con
import win32gui


class MouseInput(C.Structure):
    _fields_ = [
        ("dx", W.LONG),
        ("dy", W.LONG),
        ("mouseData", W.DWORD),
        ("dwFlags", W.DWORD),
        ("time", W.DWORD),
        ("dwExtraInfo", C.c_size_t),
    ]


class InputUnion(C.Union):
    # MOUSEINPUT is the largest member of the native INPUT union.
    _fields_ = [("mi", MouseInput)]


class Input(C.Structure):
    _anonymous_ = ("data",)
    _fields_ = [("type", W.DWORD), ("data", InputUnion)]


_user32 = C.WinDLL("user32", use_last_error=True)
_user32.SendInput.argtypes = [W.UINT, C.POINTER(Input), C.c_int]
_user32.SendInput.restype = W.UINT


def _button(*flags):
    events = (Input * len(flags))(*(Input(type=0, mi=MouseInput(dwFlags=flag)) for flag in flags))
    sent = _user32.SendInput(len(events), events, C.sizeof(Input))
    if sent != len(events):
        if len(events) == 2 and sent == 1:
            _button(win32con.MOUSEEVENTF_LEFTUP)
        raise RuntimeError(
            "Windows rejected mouse input; run the game without administrator rights."
        )


def _post_button(hwnd, point, pressed):
    x, y = point
    position = (x & 0xFFFF) | ((y & 0xFFFF) << 16)
    message = win32con.WM_LBUTTONDOWN if pressed else win32con.WM_LBUTTONUP
    win32gui.PostMessage(hwnd, message, win32con.MK_LBUTTON if pressed else 0, position)


def _cursor_position():
    try:
        return win32api.GetCursorPos()
    except pywintypes.error as error:
        if error.winerror == 5:  # The interactive desktop can disappear during sleep.
            return None
        raise


class CaptureUnavailable(RuntimeError):
    """DXGI needs to reacquire a fresh desktop presentation."""


class GameWindow:
    def __init__(
        self,
        title=None,
        *,
        batch_jumps=False,
        message_jumps=False,
        native_vision=False,
        vision_roi=None,
    ):
        # Keep capture pixels and client-to-screen input coordinates in the same units.
        try:
            _user32.SetProcessDpiAwarenessContext(C.c_void_p(-4))
        except AttributeError:
            _user32.SetProcessDPIAware()
        candidates = []

        def visit(hwnd, _):
            if not win32gui.IsWindowVisible(hwnd) or win32gui.IsIconic(hwnd):
                return
            name = win32gui.GetWindowText(hwnd)
            cls = win32gui.GetClassName(hwnd)
            _, _, width, height = win32gui.GetClientRect(hwnd)
            matches = (
                title.casefold() in name.casefold()
                if title
                else "跃动方块" in name or "wechatappex" in cls.casefold()
            )
            if matches and width > 200 and height > width:
                candidates.append((hwnd, name, width * height))

        win32gui.EnumWindows(visit, None)
        if not candidates:
            raise RuntimeError(
                "No visible portrait game window found. Open 跃动方块 in WeChat, "
                "or select its window with --title."
            )
        if len(candidates) > 1 and title is None:
            raise RuntimeError("Multiple game windows found. Select one with --title.")
        hwnd, self.title, _ = max(candidates, key=lambda c: c[2])
        self.window = SimpleNamespace(id=hwnd)
        self.capture_kind = (
            "native-dxgi" if native_vision else os.environ.get("JUMPBOT_CAPTURE", "mss")
        )
        self.last_capture_time = None
        self._capture_region = None
        if native_vision:
            from .native_vision import NativeVision

            self.capture_backend = NativeVision(vision_roi)
            self._last_ui_sample = None
        elif self.capture_kind == "dxgi":
            import dxcam

            self.capture_backend = dxcam.create(output_color="BGR", max_buffer_len=2)
        elif self.capture_kind == "mss":
            self.capture_backend = mss.mss()
        else:
            raise RuntimeError("JUMPBOT_CAPTURE must be dxgi or mss")
        self.last_press_time = None
        self.last_press_timing = None
        self.batch_jumps = batch_jumps
        self.message_jumps = message_jumps
        self._release_worker = ThreadPoolExecutor(max_workers=1) if message_jumps else None
        self._pending_release = None
        self.closed = False

    def focused(self):
        return win32gui.GetForegroundWindow() == self.window.id

    def focus(self):
        if self.focused():
            return
        try:
            win32gui.SetForegroundWindow(self.window.id)
        except pywintypes.error as error:
            raise RuntimeError("Windows denied focus. Bring the game to the foreground.") from error

    def stopped(self):
        if win32api.GetAsyncKeyState(win32con.VK_F8) & 0x8000:
            return True
        position = _cursor_position()
        return position is not None and 0 <= position[0] < 3 and 0 <= position[1] < 3

    def _geometry(self):
        hwnd = self.window.id
        if not win32gui.IsWindow(hwnd) or not win32gui.IsWindowVisible(hwnd):
            raise RuntimeError("Game window was closed or hidden.")
        if win32gui.IsIconic(hwnd):
            raise RuntimeError("Game window was minimized.")
        _, _, width, height = win32gui.GetClientRect(hwnd)
        left, top = win32gui.ClientToScreen(hwnd, (0, 0))
        if width <= 0 or height <= 0:
            raise RuntimeError("Game window has no drawable client area.")
        return left, top, width, height

    def capture(self):
        left, top, width, height = self._geometry()
        self._frame_size = (width, height)
        if self.capture_kind == "dxgi":
            region = (left, top, left + width, top + height)
            if not (
                0 <= left
                and 0 <= top
                and region[2] <= self.capture_backend.width
                and region[3] <= self.capture_backend.height
            ):
                raise RuntimeError(
                    "Keep the game fully inside the primary monitor for DXGI capture"
                )
            if region != self._capture_region:
                if self.capture_backend.is_capturing:
                    self.capture_backend.stop()
                self.capture_backend.start(region=region, target_fps=120)
                self._capture_region = region
            result = self.capture_backend.get_latest_frame(with_timestamp=True)
            if result is None:
                self._capture_region = None
                raise CaptureUnavailable("DXGI lost the desktop presentation")
            frame, timestamp = result
            if abs(time.perf_counter() - timestamp) > 2:
                self._capture_region = None
                raise CaptureUnavailable("DXGI returned an invalid or stale presentation timestamp")
            self.last_capture_time = timestamp
            return frame
        shot = self.capture_backend.grab(
            {"left": left, "top": top, "width": width, "height": height}
        )
        return cv2.cvtColor(np.asarray(shot), cv2.COLOR_BGRA2BGR)

    def observe_native(self):
        from .native_vision import CaptureLost

        region = self._geometry()
        self._frame_size = region[2:]
        if region != self._capture_region:
            self._capture_region = region
            self._last_ui_sample = None
        try:
            sample = self.capture_backend.poll(region)
        except CaptureLost as error:
            self._last_ui_sample = None
            raise CaptureUnavailable(str(error)) from error
        if sample is None:
            # A still dialog may have no new desktop presentations between
            # UI checks. Never reuse a gameplay pose for a movement decision.
            cached = self._last_ui_sample
            if cached and time.perf_counter() - cached.presented_at < 0.15:
                return replace(cached, capture_seconds=0, detection_seconds=0)
        else:
            self._last_ui_sample = sample if sample.ui else None
        return sample

    def save_native_capture(self, path):
        self.capture_backend.save(path)

    def click(self, direction, roi, *, not_before=None, expires_at=None, blocking_release=False):
        if direction not in (-1, 1) or not self.focused() or self.stopped():
            return False
        x, y, width, height = roi
        _, _, client_width, client_height = self._geometry()
        if not (0 <= x < x + width <= client_width and 0 <= y < y + height <= client_height):
            return False
        return self.click_at(
            x + round(width * (0.20 if direction < 0 else 0.80)),
            y + round(height * 0.75),
            not_before=not_before,
            expires_at=expires_at,
            batch=self.batch_jumps,
            post_message=self.message_jumps,
            blocking_release=blocking_release,
        )

    def click_at(
        self,
        x,
        y,
        *,
        not_before=None,
        expires_at=None,
        batch=False,
        post_message=False,
        blocking_release=False,
    ):
        self.last_press_timing = None
        if self._pending_release is not None:
            self._pending_release.result()
            self._pending_release = None
        entered = time.perf_counter() if not_before is not None else None
        if not self.focused() or self.stopped():
            return False
        if expires_at is not None and (
            time.perf_counter() >= expires_at
            or (not_before is not None and not_before >= expires_at)
        ):
            return False
        left, top, width, height = self._geometry()
        if getattr(self, "_frame_size", (width, height)) != (width, height):
            return False
        if not (2 <= x < width - 2 and 2 <= y < height - 2):
            return False
        target = (left + round(x), top + round(y))
        # Do not click another window covering the client area.
        hit = win32gui.WindowFromPoint(target)
        if win32gui.GetAncestor(hit, win32con.GA_ROOT) != self.window.id:
            return False
        try:
            win32api.SetCursorPos(target)
        except pywintypes.error as error:
            if error.winerror == 5:
                return False
            raise
        if not self.focused() or self.stopped():
            return False
        wait_start = time.perf_counter() if entered is not None else None
        while not_before is not None and time.perf_counter() < not_before:
            if not self.focused() or self.stopped():
                return False
            time.sleep(min(0.005, max(0.0, not_before - time.perf_counter())))
        wait_end = time.perf_counter() if entered is not None else None
        if (
            not self.focused()
            or self.stopped()
            or (expires_at is not None and time.perf_counter() >= expires_at)
        ):
            return False
        if not_before is not None:
            # The user or another window can change the target while waiting.
            if self._geometry() != (left, top, width, height):
                return False
            if _cursor_position() != target:
                return False
            hit = win32gui.WindowFromPoint(target)
            if win32gui.GetAncestor(hit, win32con.GA_ROOT) != self.window.id:
                return False
            if expires_at is not None and time.perf_counter() >= expires_at:
                return False
        if post_message:
            child_point = win32gui.ScreenToClient(hit, target)
        send_start = time.perf_counter() if entered is not None else None
        if post_message:
            _post_button(hit, child_point, pressed=True)
        elif batch:
            _button(win32con.MOUSEEVENTF_LEFTDOWN, win32con.MOUSEEVENTF_LEFTUP)
        else:
            _button(win32con.MOUSEEVENTF_LEFTDOWN)
        self.last_press_time = time.perf_counter()
        if entered is not None:
            self.last_press_timing = {
                "prepare_ms": (wait_start - entered) * 1000,
                "wait_ms": (wait_end - wait_start) * 1000,
                "wait_end_error_ms": (wait_end - not_before) * 1000,
                "validate_ms": (send_start - wait_end) * 1000,
                "send_ms": (self.last_press_time - send_start) * 1000,
            }
        if batch and not post_message:
            return True
        if post_message and not blocking_release:
            try:
                self._pending_release = self._release_worker.submit(
                    self._release_message, hit, child_point, self.last_press_time + 0.008
                )
            except RuntimeError:
                self._release_message(hit, child_point, self.last_press_time + 0.008)
                raise
            return True
        try:
            time.sleep(0.008)
        finally:
            release_start = time.perf_counter() if entered is not None else None
            if post_message:
                _post_button(hit, child_point, pressed=False)
            else:
                _button(win32con.MOUSEEVENTF_LEFTUP)
            if self.last_press_timing is not None:
                self.last_press_timing["hold_ms"] = (release_start - self.last_press_time) * 1000
                self.last_press_timing["release_ms"] = (time.perf_counter() - release_start) * 1000
        return True

    @staticmethod
    def _release_message(hwnd, point, deadline):
        # Release on the same target even if focus changes after button-down.
        remaining = deadline - time.perf_counter()
        if remaining > 0:
            time.sleep(remaining)
        try:
            _post_button(hwnd, point, pressed=False)
        except pywintypes.error as error:
            if error.winerror != 1400:  # The target window was destroyed.
                raise

    def close(self):
        if not self.closed:
            self.closed = True
            try:
                if self._release_worker:
                    self._release_worker.shutdown(wait=True)
                if self._pending_release is not None:
                    self._pending_release.result()
            finally:
                if self.capture_kind == "dxgi":
                    self.capture_backend.release()
                else:
                    self.capture_backend.close()
