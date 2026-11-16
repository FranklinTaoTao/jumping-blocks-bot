"""Direct X11 memory capture and XTEST input. No screenshot files or subprocesses."""

import time

from Xlib import X, display
from Xlib.ext import xtest
from Xlib.protocol import event

from .capture import NativeCapture


class GameWindow:
    def __init__(self, title=None):
        self.display = display.Display()
        self.root = self.display.screen().root
        self.net_name = self.display.intern_atom("_NET_WM_NAME")
        self.net_active = self.display.intern_atom("_NET_ACTIVE_WINDOW")
        candidates = []
        for w in self.root.query_tree().children:
            try:
                cls = " ".join(w.get_wm_class() or ())
                prop = w.get_full_property(self.net_name, X.AnyPropertyType)
                name = (
                    prop.value.decode("utf-8", errors="replace")
                    if prop is not None
                    else w.get_wm_name() or ""
                )
                g = w.get_geometry()
                if (
                    "wechatappex" in cls.lower()
                    and g.width > 200
                    and g.height > g.width
                    and w.get_attributes().map_state == X.IsViewable
                    and (title is None or title.casefold() in name.casefold())
                ):
                    candidates.append((w, name, g.width * g.height))
            except Exception:
                continue
        if not candidates:
            self.display.close()
            raise RuntimeError("No visible portrait WeChatAppEx game window found.")
        if len(candidates) > 1 and title is None:
            self.display.close()
            raise RuntimeError("Multiple game windows found. Select one with --title.")
        self.window, self.title, _ = max(candidates, key=lambda c: c[2])
        self.capture_backend = NativeCapture()
        self.escape_key = self.display.keysym_to_keycode(0xFFC5)  # F8
        self.closed = False
        self.last_press_time = None

    def focused(self):
        prop = self.root.get_full_property(self.net_active, X.AnyPropertyType)
        return prop is not None and len(prop.value) and int(prop.value[0]) == self.window.id

    def focus(self):
        request = event.ClientMessage(
            window=self.window, client_type=self.net_active, data=(32, [1, X.CurrentTime, 0, 0, 0])
        )
        self.root.send_event(
            request, event_mask=X.SubstructureRedirectMask | X.SubstructureNotifyMask
        )
        self.display.sync()

    def stopped(self):
        keys = self.display.query_keymap()
        pressed = bool(keys[self.escape_key // 8] & (1 << (self.escape_key % 8)))
        pointer = self.root.query_pointer()
        return pressed or (pointer.root_x < 3 and pointer.root_y < 3)

    def capture(self):
        g = self.window.get_geometry()
        if self.window.get_attributes().map_state != X.IsViewable:
            raise RuntimeError("Game window was hidden or minimized.")
        # XGetImage returns the current window pixels directly into Python
        # memory. A synchronous pull has no queued old frames to drain.
        return self.capture_backend.grab(self.window.id, g.width, g.height)

    def click(self, direction, roi):
        if direction not in (-1, 1) or not self.focused() or self.stopped():
            return False
        x, y, w, h = roi
        g = self.window.get_geometry()
        if not (0 <= x < x + w <= g.width and 0 <= y < y + h <= g.height):
            return False
        return self.click_at(x + round(w * (0.20 if direction < 0 else 0.80)), y + round(h * 0.75))

    def click_at(self, x, y):
        if not self.focused() or self.stopped():
            return False
        g = self.window.get_geometry()
        if not (2 <= x < g.width - 2 and 2 <= y < g.height - 2):
            return False
        origin = self.root.translate_coords(self.window, 0, 0)
        target_x, target_y = origin.x + round(x), origin.y + round(y)
        if not (
            0 <= target_x < self.display.screen().width_in_pixels
            and 0 <= target_y < self.display.screen().height_in_pixels
        ):
            return False
        xtest.fake_input(self.display, X.MotionNotify, x=target_x, y=target_y)
        xtest.fake_input(self.display, X.ButtonPress, 1)
        self.last_press_time = time.monotonic()
        try:
            self.display.sync()
            # A short real press survives games which poll button state each tick.
            time.sleep(0.008)
        finally:
            xtest.fake_input(self.display, X.ButtonRelease, 1)
            self.display.sync()
        return True

    def close(self):
        if not self.closed:
            self.closed = True
            self.capture_backend.close()
            self.display.close()
