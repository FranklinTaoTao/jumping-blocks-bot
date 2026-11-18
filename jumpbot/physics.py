import json
from collections import deque
from dataclasses import asdict, dataclass, field

import numpy as np

from .motion import BoxTracker, ShapeTracker
from .vision import Observation, Rect, camera_shift


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
    motion_rate: float = 1.0
    camera_speed: float = 0.0
    jump_gain: float = 1.0
    obstacle_bounds: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    gravity_gain: float = 1.0
    wall_dwell: float = 0.0


class Tracker:
    def __init__(
        self,
        physics,
        adaptive=False,
        adaptive_gravity=False,
        reset_phase=0.012,
        spatial_velocity=False,
        skip_repeated_frames=False,
        recover_frame_clock=False,
    ):
        self.physics = physics
        self.previous = None
        self.previous_t = None
        self.scroll = 0.0
        self.history = deque(maxlen=8)
        self.vx = self.vy = 0.0
        self.last_click = -100.0
        self.last_direction = 0
        self.box_tracker = BoxTracker()
        self.shape_tracker = ShapeTracker()
        self.adaptive = adaptive
        self.adaptive_gravity = adaptive_gravity
        self.reset_phase = reset_phase
        self.spatial_velocity = spatial_velocity
        self.skip_repeated_frames = skip_repeated_frames
        self.recover_frame_clock = recover_frame_clock
        self.clock_gap_at = None
        self.motion_rate = 1.0
        self.camera_speed = 0.0
        self.jump_gain = 1.0
        self.reset_estimates = deque(maxlen=30)
        self.reset_seen = False
        self.reset_observed_at = None
        self.body_widths = deque(maxlen=20)
        self.arc_history = deque(maxlen=64)
        self.curvature_fits = deque(maxlen=40)
        self.gravity_gain = 1.0
        self.wall_since = None

    def _finish_arc(self):
        if not self.adaptive or not self.adaptive_gravity or not self.last_direction:
            return
        data = np.array(
            [
                r
                for r in self.arc_history
                if self.last_click + 0.18 <= r[0] <= self.last_click + 0.60 and r[3]
            ]
        )
        if len(data) < 6 or np.ptp(data[:, 0]) < 0.15 or np.ptp(data[:, 1]) < 18:
            return
        if data[:, 1].min() < 10 or data[:, 1].max() > 410:
            return
        x = self.last_direction * (data[:, 1] - data[-1, 1])
        if np.any(np.diff(x) < -0.6):
            return
        design = np.column_stack([x, np.ones(len(x))])
        quadratic = x * x - design @ np.linalg.lstsq(design, x * x, rcond=None)[0]
        residual = data[:, 2] - design @ np.linalg.lstsq(design, data[:, 2], rcond=None)[0]
        weight = quadratic @ quadratic
        product = quadratic @ residual
        curvature = 2 * product / weight
        error = np.sqrt(np.mean((residual - curvature * quadratic / 2) ** 2))
        if 0.045 < curvature < 0.085 and error < 1.0:
            self.curvature_fits.append((weight, product))
        if len(self.curvature_fits) >= 8:
            fits = np.array(self.curvature_fits)
            measured = 2 * fits[:, 1].sum() / fits[:, 0].sum()
            ratio = measured / (self.physics.gravity / self.physics.speed**2)
            self.gravity_gain = 0.8 * self.gravity_gain + 0.2 * float(np.clip(ratio, 0.90, 1.10))

    def _adapt_motion(self, t):
        """Use the spatial parabola to avoid differentiating a variable game clock."""
        age = t - self.last_click
        samples = [row for row in self.history if row[0] >= self.last_click + 0.005]
        samples = np.array(samples[-(3 if age < 0.18 else 6) :])
        if len(samples) < 3 or age < 0.040:
            return False
        direction = self.last_direction
        distance = direction * (samples[:, 1] - samples[-1, 1])
        at_wall = samples[-1, 1] <= 1 or samples[-1, 1] >= 419
        if at_wall:
            # Horizontal displacement no longer reveals elapsed game time at
            # a wall. Fit vertical motion using the last observed game rate.
            clean = samples[samples[:, 0] >= self.last_click + 0.05]
            if len(clean) < 3:
                return False
            ts = clean[:, 0] - t
            gravity = self.physics.gravity * self.motion_rate**2 * self.gravity_gain
            design = np.column_stack([ts, np.ones(len(ts))])
            values = clean[:, 2] - 0.5 * gravity * ts**2
            velocity, intercept = np.linalg.lstsq(design, values, rcond=None)[0]
            if np.sqrt(np.mean((values - design @ [velocity, intercept]) ** 2)) > 2:
                return False
            if (
                not self.reset_seen
                and age < 0.12
                and velocity > -0.65 * self.physics.jump * self.motion_rate
            ):
                return False
            self.vx, self.vy = 0.0, float(velocity)
            if not self.reset_seen:
                self.reset_observed_at = clean[-2, 0]
            self.reset_seen = True
            return True
        if np.ptp(distance) < 3 or np.any(np.diff(distance) < -0.6):
            return False
        design = np.column_stack([distance, np.ones(len(distance))])
        curvature = self.physics.gravity * self.gravity_gain / self.physics.speed**2
        adjusted = samples[:, 2] - 0.5 * curvature * distance**2
        slope, intercept = np.linalg.lstsq(design, adjusted, rcond=None)[0]
        residual = adjusted - design @ [slope, intercept]
        if np.sqrt(np.mean(residual**2)) > 2.0:
            return False
        # Before the reset has appeared, the previous falling arc must not arm input.
        reset_slope = -self.physics.jump / self.physics.speed
        if not self.reset_seen and age < 0.12 and slope > reset_slope + 1.2:
            return False
        dt = samples[:, 0] - samples[-1, 0]
        speed = np.linalg.lstsq(np.column_stack([dt, np.ones(len(dt))]), samples[:, 1], rcond=None)[
            0
        ][0]
        measured_rate = abs(speed) / self.physics.speed
        if not 0.45 < measured_rate < 1.35:
            return False
        if not self.reset_seen:
            self.reset_observed_at = samples[-2, 0]
        self.reset_seen = True
        if np.ptp(dt) >= 0.10 and np.ptp(distance) >= 10:
            self.motion_rate = 0.65 * self.motion_rate + 0.35 * measured_rate
        self.vx = direction * self.physics.speed * self.motion_rate
        self.vy = slope * self.physics.speed * self.motion_rate
        if 0.09 <= age <= 0.22:
            effective_jump = (
                -self.vy
                + self.physics.gravity
                * self.motion_rate** 2
                * self.gravity_gain
                * (age - self.reset_phase)
            )
            estimate = effective_jump / (self.physics.jump * self.motion_rate)
            if 0.85 < estimate < 1.20:
                self.reset_estimates.append(float(estimate))
                self.jump_gain = 0.9 * self.jump_gain + 0.1 * float(np.median(self.reset_estimates))
        return True

    def clicked(self, direction, t):
        self._finish_arc()
        self.arc_history.clear()
        self.last_click = t
        self.reset_seen = False
        self.reset_observed_at = None
        self.clock_gap_at = None
        self.last_direction = direction
        self.vx = direction * self.physics.speed
        self.vy = -self.physics.jump

    def _spatial_vertical_velocity(self, arc, current_x, min_duration, min_step):
        # Horizontal travel measures game time on one interior arc.
        seconds = self.last_direction * (arc[:, 1] - current_x) / self.physics.speed
        if np.ptp(seconds) < min_duration or np.any(np.diff(seconds) < min_step):
            return None
        matrix = np.column_stack([seconds, np.ones(len(arc))])
        adjusted = arc[:, 2] - 0.5 * self.physics.gravity * seconds**2
        fit = np.linalg.lstsq(matrix, adjusted, rcond=None)[0]
        if np.sqrt(np.mean((adjusted - matrix @ fit) ** 2)) <= 2:
            return float(fit[0])
        return None

    def update(self, observation: Observation | None, t):
        if observation is None or not observation.active or observation.player is None:
            self.previous = None
            self.history.clear()
            self.arc_history.clear()
            self.box_tracker = BoxTracker()
            self.shape_tracker = ShapeTracker()
            self.last_direction = 0
            self.last_click = -100.0
            self.motion_rate = 1.0
            self.camera_speed = 0.0
            self.wall_since = None
            self.clock_gap_at = None
            return None
        previous = self.previous
        if (
            self.skip_repeated_frames
            and self.last_direction
            and previous
            and observation.player == previous.player
            and observation.obstacles == previous.obstacles
            and len(observation.shapes) == len(previous.shapes)
            and all(
                np.array_equal(a.points, b.points)
                for a, b in zip(observation.shapes, previous.shapes)
            )
        ):
            # A duplicate presentation may carry a newer capture timestamp.
            # Keep it out of the motion fit and protect input from stale poses.
            return None
        p = observation.player
        if p.x > 4 and p.x + p.w < 416 and p.w >= 24:
            self.body_widths.append(p.w)
        body_width = float(np.median(self.body_widths)) if self.body_widths else 30.0
        if p.w < body_width - 3:
            if p.x <= 1:
                p = Rect(p.x + p.w - body_width, p.y, body_width, p.h)
            elif p.x + p.w >= 419:
                p = Rect(p.x, p.y, body_width, p.h)
        if p.cx <= 1 or p.cx >= 419:
            if self.wall_since is None:
                self.wall_since = t
        else:
            self.wall_since = None
        wall_dwell = t - self.wall_since if self.wall_since is not None else 0.0
        dt = t - self.previous_t if self.previous_t is not None else 0
        valid = self.previous is not None and 0.001 < dt < 0.15
        shift, camera_ok = (
            camera_shift(self.previous.obstacles, observation.obstacles) if valid else (0, False)
        )
        if valid:
            self.scroll += shift
            self.camera_speed = 0.6 * self.camera_speed + 0.4 * max(0.0, shift / dt)
            previous_x = self.history[-1][1] if self.history else self.previous.player.cx
            raw_vx = (p.cx - previous_x) / dt
            if (
                self.recover_frame_clock
                and not self.adaptive
                and self.last_direction
                and 0.006 <= dt < 0.15
                and 6 < min(previous_x, p.cx) <= max(previous_x, p.cx) < 414
                and self.previous_t >= self.last_click + 0.033
                and (p.cx - previous_x) * self.last_direction > 1.2
                and not 0.65 * self.physics.speed
                <= raw_vx * self.last_direction
                <= 1.55 * self.physics.speed
            ):
                self.clock_gap_at = t
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
            self.camera_speed = 0.0
        self.history.append((t, p.cx, p.cy - self.scroll, camera_ok))
        self.arc_history.append((t, p.cx, p.cy - self.scroll, camera_ok))
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
            if self.spatial_velocity and self.last_direction and not self.adaptive:
                # Horizontal displacement measures game time within one arc,
                # even if image delivery repeats or skips a presentation frame.
                arc = recent[recent[:, 3] > 0]
                if len(arc) >= 3 and arc[:, 1].min() > 3 and arc[:, 1].max() < 417:
                    velocity = self._spatial_vertical_velocity(arc, p.cx, 0.05, -0.004)
                    if velocity is not None:
                        self.vy = velocity
        elif self.last_direction:
            self.vy = -self.physics.jump + self.physics.gravity * max(
                0, t - self.last_click - self.reset_phase
            )
        if self.clock_gap_at is not None and 0 <= t - self.clock_gap_at <= 0.12:
            # Presentation jitter can corrupt several timestamp-based fits.
            # Use clean spatial samples until the timing anomaly ages out.
            arc = np.array([a for a in self.history if a[0] >= self.last_click + 0.033 and a[3]])
            if len(arc) >= 3 and arc[:, 1].min() > 6 and arc[:, 1].max() < 414:
                velocity = self._spatial_vertical_velocity(arc, p.cx, 0.032, 0.004)
                if velocity is not None:
                    self.vy = velocity
        if self.last_direction:
            self.vx = self.last_direction * self.physics.speed
        if len(self.history) >= 4:
            last = np.array(list(self.history)[-4:])
            if t - self.last_click > 0.1 and np.ptp(last[:, 1]) < 1.5 and np.ptp(last[:, 2]) < 1.5:
                self.vx = self.vy = 0
        adaptive_ready = True
        if self.adaptive and self.last_direction:
            adaptive_ready = self._adapt_motion(t)
        # The last jump direction persists through a wall stop, but horizontal
        # velocity does not. Pixel clipping can put the fitted center at 1/419.
        if (p.cx <= 1 and self.vx < 0) or (p.cx >= 419 and self.vx > 0):
            self.vx = 0.0
        boxes, obstacle_bounds = self.box_tracker.update(
            observation.obstacles, t, self.scroll, valid and camera_ok
        )
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
            valid
            and camera_ok
            and len(self.history) >= 3
            and t >= self.last_click + self.reset_phase
            and adaptive_ready,
            boxes,
            shapes,
            self.motion_rate,
            self.camera_speed,
            self.jump_gain,
            obstacle_bounds,
            self.gravity_gain,
            wall_dwell,
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
