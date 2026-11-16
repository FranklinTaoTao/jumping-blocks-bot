import json
from collections import deque
from dataclasses import asdict, dataclass, field

import numpy as np

from .motion import ShapeTracker
from .vision import Observation, camera_shift


def project_camera(world_y, anchor, previous_peak=None):
    """Project a path through a camera that only follows new upward progress.

    Coordinates increase downwards. Below the anchor the character moves while
    the camera stays still. After reaching a new high point, that camera offset
    persists through falls and the below-anchor part of subsequent jumps.
    """
    y = np.asarray(world_y, dtype=float)
    peak = np.asarray(anchor if previous_peak is None else previous_peak)
    peaks = np.minimum.accumulate(np.minimum(y, peak[..., None]), axis=-1)
    return y + np.maximum(0.0, anchor - peaks), peaks


@dataclass
class Physics:
    speed: float = 150.0
    jump: float = 700.0
    gravity: float = 1400.0

    @classmethod
    def load(cls, path):
        with open(path) as stream:
            return cls(**json.load(stream)["physics"])


@dataclass
class State:
    x: float
    y: float
    vx: float
    vy: float
    w: float
    h: float
    age_since_click: float = 10.0
    ready: bool = False
    obstacles: np.ndarray = field(default_factory=lambda: np.empty((0, 6)))
    shapes: list = field(default_factory=list)


class Tracker:
    def __init__(self, physics):
        self.physics = physics
        self.previous = None
        self.previous_t = None
        self.scroll = 0.0
        self.history = deque(maxlen=8)
        self.vx = self.vy = 0.0
        self.last_click = -100.0
        self.last_direction = 0
        self.square_speeds = {}
        self.shape_tracker = ShapeTracker()

    def clicked(self, direction, t):
        self.last_click = t
        self.last_direction = direction
        self.vx = direction * self.physics.speed
        self.vy = -self.physics.jump

    def update(self, observation: Observation | None, t):
        if observation is None or not observation.active or observation.player is None:
            self.previous = None
            self.history.clear()
            self.square_speeds.clear()
            self.shape_tracker = ShapeTracker()
            self.last_direction = 0
            self.last_click = -100.0
            return None
        p = observation.player
        dt = t - self.previous_t if self.previous_t is not None else 0
        valid = self.previous is not None and 0.001 < dt < 0.15
        shift, camera_ok = (
            camera_shift(self.previous.obstacles, observation.obstacles) if valid else (0, False)
        )
        if valid:
            self.scroll += shift
            raw_vx = (p.cx - self.previous.player.cx) / dt
            raw_vy = (p.cy - self.previous.player.cy - shift) / dt + self.physics.gravity * dt / 2
            if abs(raw_vx) < 600 and abs(raw_vy) < 1800:
                # A velocity reset must not be averaged over an earlier arc.
                reset = raw_vy < self.vy + self.physics.gravity * dt - 180
                if reset and not self.last_direction:
                    # Unknown/manual click during replay: start a new fitted arc.
                    # Live clicks are timestamped explicitly. Frame quantization
                    # must not be mistaken for an extra click that disarms input.
                    self.history.clear()
                self.vx = raw_vx
                self.vy = raw_vy
            else:
                valid = False
                self.history.clear()
        else:
            self.history.clear()
            self.scroll = 0
            self.vx = self.vy = 0
        self.history.append((t, p.cx, p.cy - self.scroll, camera_ok))
        # Capture can still show a pre-click rendered frame after the X11 event
        # has been delivered. Do not fit a new arc through that old falling
        # position; use the known reset until three clean post-event samples.
        recent = np.array([a for a in self.history if a[0] >= self.last_click + 0.050])
        if len(recent) >= 3:
            ts = recent[:, 0] - t
            # Fit position with known gravity across several frames, instead of
            # differentiating one quantized frame interval.
            design = np.stack([ts, np.ones(len(ts))], axis=1)
            self.vx = float(np.linalg.lstsq(design, recent[:, 1], rcond=None)[0][0])
            self.vy = float(
                np.linalg.lstsq(
                    design, recent[:, 2] - 0.5 * self.physics.gravity * ts**2, rcond=None
                )[0][0]
            )
        elif self.last_direction:
            self.vy = -self.physics.jump + self.physics.gravity * max(
                0, t - self.last_click - 0.012
            )
        if self.last_direction:
            self.vx = self.last_direction * self.physics.speed
        if len(self.history) >= 4:
            last = np.array(list(self.history)[-4:])
            if t - self.last_click > 0.1 and np.ptp(last[:, 1]) < 1.5 and np.ptp(last[:, 2]) < 1.5:
                self.vx = self.vy = 0
        boxes = []
        next_speeds = {}
        for i, b in enumerate(observation.obstacles):
            vx = vy = 0.0
            if valid and camera_ok and b.w < 45:
                choices = [
                    (j, a)
                    for j, a in enumerate(self.previous.obstacles)
                    if abs(a.w - b.w) < 3
                    and abs(a.h - b.h) < 3
                    and abs(a.cx - b.cx) < 30
                    and abs(a.cy + shift - b.cy) < 50
                ]
                if choices:
                    j, a = min(
                        choices,
                        key=lambda pair: abs(pair[1].cx - b.cx) + abs(pair[1].cy + shift - b.cy),
                    )
                    old = self.square_speeds.get(j, (0.0, 0.0))
                    vx = 0.6 * (b.cx - a.cx) / dt + 0.4 * old[0]
                    vy = 0.6 * (b.cy - a.cy - shift) / dt + 0.4 * old[1]
                    vx, vy = np.clip([vx, vy], -500, 500)
            next_speeds[i] = (vx, vy)
            boxes.append((b.x, b.y, b.w, b.h, vx, vy))
        self.square_speeds = next_speeds
        self.previous, self.previous_t = observation, t
        shapes = self.shape_tracker.update(observation.shapes, t, self.scroll)
        return State(
            p.cx,
            p.cy,
            self.vx,
            self.vy,
            p.w,
            p.h,
            t - self.last_click,
            valid and camera_ok and len(self.history) >= 3,
            np.array(boxes, dtype=float).reshape(-1, 6),
            shapes,
        )


def calibrate(samples):
    """Robust ballistic fits to overlapping, uninterrupted ~0.27-second arcs.

    samples are (timestamp, x, camera-corrected y). Input events need not be
    recorded: windows containing velocity resets fail the residual check.
    """
    data = np.asarray(samples, dtype=float)
    fits = []
    if len(data) < 10:
        raise ValueError("Too few character observations to calibrate.")
    for i in range(len(data) - 8):
        part = data[i : i + 9]
        ts = part[:, 0] - part[0, 0]
        if not 0.15 < ts[-1] < 0.4 or np.max(np.diff(ts)) > 0.07 or np.min(np.diff(ts)) <= 0:
            continue
        py = np.polyfit(ts, part[:, 2], 2)
        px = np.polyfit(ts, part[:, 1], 1)
        ey = float(np.sqrt(np.mean((np.polyval(py, ts) - part[:, 2]) ** 2)))
        ex = float(np.sqrt(np.mean((np.polyval(px, ts) - part[:, 1]) ** 2)))
        g, speed = 2 * py[0], abs(px[0])
        if ey < 1.3 and ex < 1.3 and 500 < g < 3000 and 50 < speed < 350:
            fits.append((g, speed, py[1], part[0, 0]))
    if len(fits) < 12:
        raise ValueError(f"Only {len(fits)} clean ballistic windows; need at least 12.")
    fits = np.array(fits)
    gravity, speed = np.median(fits[:, :2], axis=0)
    # The fastest clean ascending windows begin just after a velocity reset.
    # Extrapolate at most half a capture frame to account for click phase.
    ascending = -fits[:, 2]
    jump = float(np.percentile(ascending, 96) + gravity / 60)
    if not 400 < jump < 1000:
        raise ValueError("Could not identify a plausible upward reset speed.")
    physics = Physics(float(speed), jump, float(gravity))
    return {
        "physics": asdict(physics),
        "clean_fit_windows": len(fits),
        "gravity_10_90": np.percentile(fits[:, 0], [10, 90]).tolist(),
        "units": "pixels at board width 420; time in seconds",
        "jump_uncertainty": "Inferred without click timestamps; validate live.",
    }
