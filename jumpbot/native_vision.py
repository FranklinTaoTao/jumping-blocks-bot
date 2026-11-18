"""Windows DXGI and OpenCV backend that returns detections through a C ABI."""

import ctypes as C
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .motion import Shape
from .ui import UIAction
from .vision import Observation, Rect


class CaptureLost(RuntimeError):
    pass


class _Box(C.Structure):
    _fields_ = [(key, C.c_int) for key in ("x", "y", "w", "h")]

    def rect(self):
        return Rect(self.x, self.y, self.w, self.h)


class _Shape(C.Structure):
    _fields_ = [
        ("points", C.POINTER(C.c_int)),
        ("count", C.c_int),
        ("x", C.c_double),
        ("y", C.c_double),
        ("angle", C.c_double),
        ("area", C.c_double),
    ]


class _Detection(C.Structure):
    _fields_ = [
        *((field, C.c_int) for field in ("width", "height", "board_height", "active", "score")),
        ("roi", _Box),
        ("player", _Box),
        ("obstacles", C.POINTER(_Box)),
        ("obstacle_count", C.c_int),
        ("shapes", C.POINTER(_Shape)),
        ("shape_count", C.c_int),
        ("ui", C.c_char_p),
        *(
            (field, C.c_double)
            for field in (
                "ui_x",
                "ui_y",
                "ui_confidence",
                "presented_at",
                "capture_seconds",
                "detection_seconds",
            )
        ),
    ]


@dataclass(slots=True)
class Sample:
    frame_size: tuple[int, int]
    board_height: int
    observation: Observation | None
    score: int | None
    ui: UIAction | None
    presented_at: float
    capture_seconds: float
    detection_seconds: float


def library_path():
    root = Path(__file__).resolve().parent.parent
    default = root / "build" / "native-vision" / "install" / "jumpbot_vision.dll"
    return Path(os.environ.get("JUMPBOT_NATIVE_DLL", default)).resolve()


def _load_library():
    path = library_path()
    if not path.is_file():
        raise RuntimeError(
            f"Native vision DLL missing at {path}. See docs/windows-native-vision.md."
        )
    with os.add_dll_directory(str(path.parent)):
        library = C.CDLL(str(path))
    version = getattr(library, "jumpbot_abi_version", None)
    if version is None or version() != 1:
        raise RuntimeError(
            f"Incompatible native vision DLL at {path}. Rebuild it from this checkout."
        )
    pointer, integer = C.c_void_p, C.c_int
    functions = {
        "create": (pointer, (integer,) * 4),
        "destroy": (None, (pointer,)),
        "error": (C.c_char_p, (pointer,)),
        "add_template": (integer, (pointer, C.c_char_p, pointer, integer, integer)),
        "poll": (integer, (pointer, integer, integer, integer, integer, C.POINTER(_Detection))),
        "inspect": (
            integer,
            (pointer, pointer, integer, integer, C.c_size_t, integer, C.POINTER(_Detection)),
        ),
        "save": (integer, (pointer, C.c_wchar_p)),
    }
    for name, (restype, argtypes) in functions.items():
        function = getattr(library, f"jumpbot_{name}")
        function.restype = restype
        function.argtypes = argtypes
    return library


class NativeVision:
    """Own one capture and detector on the control-loop thread."""

    def __init__(self, roi=None):
        self.library = _load_library()
        self.handle = self.library.jumpbot_create(*(roi or (0, 0, 0, 0)))
        if not self.handle:
            raise RuntimeError("Could not initialize the native detector.")
        self._result = _Detection()
        assets = Path(__file__).with_name("assets")
        try:
            for name in ("digits.npz", "digits_windows.npz", "ui_templates.npz"):
                with np.load(assets / name) as archive:
                    for key in archive.files:
                        glyph = np.ascontiguousarray(archive[key], dtype=np.uint8)
                        height, width = glyph.shape
                        self._check(
                            self.library.jumpbot_add_template(
                                self.handle, key.encode("ascii"), glyph.ctypes.data, width, height
                            )
                        )
        except Exception:
            self.close()
            raise

    def _check(self, status):
        if status < 0:
            error = self.library.jumpbot_error(self.handle).decode(errors="replace")
            raise (CaptureLost if status == -2 else RuntimeError)(error)
        return status > 0

    def _open_handle(self):
        if self.handle is None:
            raise RuntimeError("Native vision has been closed.")
        return self.handle

    def _sample(self):
        raw = self._result
        observation = None
        if raw.board_height:
            roi = raw.roi
            shapes = []
            for i in range(raw.shape_count):
                shape = raw.shapes[i]
                points = np.ctypeslib.as_array(shape.points, (shape.count * 2,)).reshape(-1, 2)
                shapes.append(
                    Shape(points.astype(float), shape.x, shape.y, shape.angle, shape.area)
                )
            observation = Observation(
                None,
                raw.player.rect() if raw.player.w else None,
                [raw.obstacles[i].rect() for i in range(raw.obstacle_count)],
                bool(raw.active),
                (roi.x, roi.y, roi.w, roi.h),
                shapes,
            )
        action = None
        if raw.ui:
            action = UIAction(raw.ui.decode("ascii"), raw.ui_x, raw.ui_y, raw.ui_confidence)
        return Sample(
            (raw.width, raw.height),
            raw.board_height,
            observation,
            raw.score if raw.score >= 0 else None,
            action,
            raw.presented_at,
            raw.capture_seconds,
            raw.detection_seconds,
        )

    def poll(self, region):
        handle = self._open_handle()
        if self._check(self.library.jumpbot_poll(handle, *region, C.byref(self._result))):
            return self._sample()
        return None

    def inspect(self, frame):
        """Offline parity testing on existing pixels."""
        if (
            frame.dtype != np.uint8
            or frame.ndim != 3
            or frame.shape[2] not in (3, 4)
            or not frame.size
        ):
            raise ValueError("Expected a nonempty uint8 BGR or BGRA frame.")
        frame = np.ascontiguousarray(frame)
        height, width, channels = frame.shape
        self._check(
            self.library.jumpbot_inspect(
                self._open_handle(),
                frame.ctypes.data,
                width,
                height,
                frame.strides[0],
                channels,
                C.byref(self._result),
            )
        )
        return self._sample()

    def save(self, path):
        self._check(self.library.jumpbot_save(self._open_handle(), str(path)))

    def close(self):
        if self.handle:
            self.library.jumpbot_destroy(self.handle)
            self.handle = None
