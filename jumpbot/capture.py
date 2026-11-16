"""Native XGetImage capture: X11 drawable -> NumPy, entirely in memory.

Use libX11 directly for pixel replies. python-xlib is used for small control
messages elsewhere, but its large-image protocol decoding adds avoidable latency.
"""

import ctypes as C
from ctypes.util import find_library

import cv2
import numpy as np


class XImage(C.Structure):
    _fields_ = [
        ("width", C.c_int),
        ("height", C.c_int),
        ("xoffset", C.c_int),
        ("format", C.c_int),
        ("data", C.c_void_p),
        ("byte_order", C.c_int),
        ("bitmap_unit", C.c_int),
        ("bitmap_bit_order", C.c_int),
        ("bitmap_pad", C.c_int),
        ("depth", C.c_int),
        ("bytes_per_line", C.c_int),
        ("bits_per_pixel", C.c_int),
        ("red_mask", C.c_ulong),
        ("green_mask", C.c_ulong),
        ("blue_mask", C.c_ulong),
    ]


class NativeCapture:
    def __init__(self):
        library = find_library("X11")
        if library is None:
            raise RuntimeError(
                "libX11 is required for live capture. Install your system's X11 libraries."
            )
        self.lib = C.CDLL(library)
        self.lib.XOpenDisplay.argtypes = [C.c_char_p]
        self.lib.XOpenDisplay.restype = C.c_void_p
        self.lib.XCloseDisplay.argtypes = [C.c_void_p]
        self.lib.XGetImage.argtypes = [
            C.c_void_p,
            C.c_ulong,
            C.c_int,
            C.c_int,
            C.c_uint,
            C.c_uint,
            C.c_ulong,
            C.c_int,
        ]
        self.lib.XGetImage.restype = C.POINTER(XImage)
        self.lib.XDestroyImage.argtypes = [C.POINTER(XImage)]
        self.display = self.lib.XOpenDisplay(None)
        if not self.display:
            raise RuntimeError("Cannot open X11 display for capture.")

    def grab(self, window_id, width, height):
        ptr = self.lib.XGetImage(
            self.display, window_id, 0, 0, width, height, C.c_ulong(-1).value, 2
        )
        if not ptr:
            raise RuntimeError("XGetImage failed.")
        try:
            im = ptr.contents
            if (
                im.bits_per_pixel != 32
                or im.byte_order != 0
                or im.red_mask != 0xFF0000
                or im.green_mask != 0xFF00
                or im.blue_mask != 0xFF
            ):
                raise RuntimeError("Capture requires little-endian 32-bit BGRX X11 pixels.")
            storage = (C.c_ubyte * (im.bytes_per_line * height)).from_address(im.data)
            rows = np.ctypeslib.as_array(storage).reshape(height, im.bytes_per_line)
            pixels = rows[:, : width * 4].reshape(height, width, 4)
            # cvtColor owns its output: the XImage can be freed immediately.
            return cv2.cvtColor(pixels, cv2.COLOR_BGRA2BGR)
        finally:
            self.lib.XDestroyImage(ptr)

    def close(self):
        if self.display:
            self.lib.XCloseDisplay(self.display)
            self.display = None
