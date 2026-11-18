"""Rigid-shape tracking for rotating hammers and other nonrectangular hazards."""

from dataclasses import dataclass, replace

import numpy as np


def reflect_position(position, velocity, times, lower, upper):
    """Position of a constant-speed block bouncing between two measured limits."""
    span = upper - lower
    phase = (position + velocity * np.asarray(times) - lower) % (2 * span)
    return lower + np.minimum(phase, 2 * span - phase)


def horizontal_extent(x, width, velocity, times, lower, upper):
    """Project a solid block or the moving inner edge of a wall bar."""
    if upper <= lower:
        return x + velocity * times, width
    if x < 4:
        return x, reflect_position(x + width, velocity, times, lower, upper) - x
    stop = x + width
    start = reflect_position(x, velocity, times, lower, upper)
    return start, stop - start if stop > 416 else width


class BoxTracker:
    """Track rectangles of every size in camera-corrected coordinates."""

    def __init__(self):
        self.tracks = []
        self.gaps = []

    def _moving_gaps(self, boxes, t, scroll, reliable):
        bars = [
            (i, box)
            for i, box in enumerate(boxes)
            if 30 <= box.h <= 42 or (box.y <= 3 and 8 <= box.h < 30)
        ]
        pairs = []
        fresh = []
        for left_index, left in bars:
            if left.x >= 4:
                continue
            for right_index, right in bars:
                width = right.x - (left.x + left.w)
                if (
                    right.x + right.w <= 416
                    or not 164 <= width <= 176
                    or abs(left.y + left.h - right.y - right.h) > 4
                ):
                    continue
                inner = (left.x + left.w + right.x) / 2
                bottom = (left.y + left.h + right.y + right.h) / 2 - scroll
                matches = [
                    old
                    for old in self.gaps
                    if 0 < t - old[0][-1][0] < 0.15
                    and abs(bottom - old[0][-1][2]) < 12
                    and abs(inner - old[0][-1][1]) < 22
                ]
                history, velocity = (
                    min(matches, key=lambda old: abs(bottom - old[0][-1][2]))
                    if matches
                    else ([], 0.0)
                )
                previous_velocity = velocity
                if history and reliable:
                    displacement = inner - history[-1][1]
                    if abs(previous_velocity) > 40 and displacement * previous_velocity < -60:
                        history = history[-1:]
                        velocity = -previous_velocity
                history = history[-7:] + [(t, inner, bottom)]
                if reliable and len(history) >= 3:
                    samples = np.asarray(history)
                    positions = samples[:, 1]
                    if np.ptp(positions) >= 3.0:
                        times = samples[:, 0] - samples[:, 0].mean()
                        velocity = np.clip(times @ positions / (times @ times), -120, 120)
                    elif t - history[0][0] >= 0.075:
                        velocity = 0.0
                fresh.append((history, velocity))
                if abs(velocity) > 40:
                    pairs.append((left_index, right_index, float(velocity), width))
                break
        missed = [
            old
            for old in self.gaps
            if t - old[0][-1][0] < 0.12
            and all(abs(new[0][-1][2] - old[0][-1][2]) >= 12 for new in fresh)
        ]
        self.gaps = fresh + missed
        return pairs

    def update(self, boxes, t, scroll, reliable):
        # The incoming sliding block is a roughly 70px square. Its visible
        # height changes as it enters the board, but its bottom edge and width
        # still locate the whole square. Match its horizontal motion before
        # the top edge becomes visible.
        observed = boxes
        boxes = []
        for box in observed:
            slider = 64 <= box.w <= 76 and 58 <= box.x <= 295
            large_square = 90 <= box.w <= 104 and 30 <= box.x and box.x + box.w <= 390
            if (slider or large_square) and 0 <= box.y <= 3 and 6 <= box.h < box.w:
                box = replace(box, y=box.y + box.h - box.w, h=box.w)
            boxes.append(box)
        candidates = []
        for i, box in enumerate(boxes):
            if box.x < 4 or box.x + box.w > 416:
                continue
            for j, track in enumerate(self.tracks):
                last = track["history"][-1]
                mobile = track["mobile"]
                score_overlap = mobile and box.y <= 145
                preserved_span = track["span"] is not None and box.y <= 145
                if (
                    0 < t - last[0] < 0.15
                    and abs(box.w - track["w"]) < (11 if score_overlap or preserved_span else 4)
                    and abs(box.h - track["h"])
                    < (12 if score_overlap or preserved_span else 9 if mobile else 4)
                ):
                    distance = abs(box.cx - last[1]) + abs(box.y - scroll - last[2])
                    if distance < 50:
                        candidates.append((distance, i, j))
        matched, used = {}, set()
        for _, i, j in sorted(candidates):
            if i not in matched and j not in used:
                matched[i] = j
                used.add(j)
        result, bounds, next_tracks = [], [], []
        for i, box in enumerate(boxes):
            raw = observed[i]
            history = []
            previous_velocity = np.zeros(2)
            initial_vx = 0.0
            mobile = False
            span = None
            if i in matched:
                track = self.tracks[matched[i]]
                history = track["history"][-7:]
                previous_velocity = track["velocity"]
                mobile = track["mobile"]
                span = track["span"]
                if mobile and box.y <= 145:
                    initial_vx = previous_velocity[0]
            if span and raw.y <= 145:
                span_x, span_width = span
                shared_edge = (
                    min(abs(raw.x - span_x), abs(raw.x + raw.w - span_x - span_width)) <= 2
                )
                if shared_edge and 2 <= raw.w - span_width <= 10:
                    top = raw.y + raw.h - span_width if raw.y <= 3 and raw.h < span_width else raw.y
                    height = max(raw.h, span_width) if raw.y <= 3 else raw.h
                    box = replace(box, x=span_x, y=top, w=span_width, h=height)
                elif not shared_edge:
                    span = None
            if 0 <= raw.y <= 3 and 16 <= raw.h <= 50 and 90 <= raw.w <= 104:
                # A rectangular square's full horizontal span is already
                # visible before a score numeral can widen an inner edge.
                span = (raw.x, raw.w)
            point = (t, box.cx, box.y - scroll)
            if history and reliable:
                displacement = np.asarray(point[1:]) - np.asarray(history[-1][1:])
                # Do not average opposite directions across an observed bounce.
                turn = np.any((displacement * previous_velocity < -10) & (abs(displacement) >= 1.5))
                expected_bounce = False
                if mobile and box.y <= 145 and t - history[-1][0] <= 0.08:
                    old_left = history[-1][1] - track["w"] / 2
                    vx = previous_velocity[0]
                    projected = old_left + vx * (t - history[-1][0])
                    past_limit = projected >= 356 - track["w"] if vx > 0 else projected <= 65
                    expected_bounce = past_limit and displacement[0] * vx <= abs(vx) * 0.8
                if turn or expected_bounce:
                    if mobile:
                        initial_vx = previous_velocity[0]
                        if expected_bounce or (
                            abs(displacement[0]) >= 1.5 and displacement[0] * initial_vx < -10
                        ):
                            initial_vx = -initial_vx
                    history = history[-1:]
            history = history + [point]
            velocity = np.array([initial_vx, 0.0])
            if reliable and len(history) >= 3:
                data = np.array(history)
                dt = data[:, 0] - t
                design = np.column_stack([dt, np.ones(len(dt))])
                velocity = np.linalg.lstsq(design, data[:, 1:3], rcond=None)[0][0]
                # Pixel quantization of stationary blocks is not real motion.
                velocity[np.ptp(data[:, 1:3], axis=0) < 3] = 0
                velocity = np.clip(velocity, -500, 500)
                if mobile and box.y <= 145 and abs(velocity[0]) < 50:
                    velocity[0] = initial_vx
            if 90 <= box.w <= 104 and raw.y <= 3 and raw.h < box.h:
                # At entry, the fitted top is inferred from a noisy bottom
                # edge. It cannot establish vertical motion in world space.
                velocity[1] = 0.0
            limit = (0.0, 0.0)
            forecast_vx = velocity[0]
            if 60 <= box.w <= 80 and 0.8 < box.w / box.h < 1.2 and abs(velocity[0]) > 30:
                # Measured in the Windows run: left edge reverses near 65;
                # right edge reverses near 356, on the normalized 420px board.
                lower, upper = 65.0, 356.0 - box.w
                # Keep the detected outline at t=0 when quantization puts
                # its edge just beyond a fitted reversal point.
                offset = box.x - min(max(box.x, lower), upper)
                limit = (lower + offset, upper + offset)
                velocity[1] = 0.0
                mobile = True
                # Preserve the measured direction while regularizing noisy
                # speeds in the forecast. Keep raw velocity for association.
                speed = min(max(abs(forecast_vx), 142.0), 178.0)
                forecast_vx = np.copysign(speed, forecast_vx)
            elif mobile and box.y <= 145 and abs(forecast_vx) >= 70:
                # The score can temporarily change the detected size enough
                # that a reflected forecast is unavailable.
                speed = abs(forecast_vx)
                adjusted_speed = speed + 0.65 * (np.clip(speed, 135.0, 185.0) - speed)
                forecast_vx = np.copysign(adjusted_speed, forecast_vx)
            result.append((box.x, box.y, box.w, box.h, forecast_vx, velocity[1]))
            bounds.append(limit)
            next_tracks.append(
                {
                    "history": history,
                    "velocity": velocity,
                    "w": box.w,
                    "h": box.h,
                    "mobile": mobile,
                    "span": span,
                }
            )
        self.tracks = next_tracks + [
            track
            for j, track in enumerate(self.tracks)
            if j not in used
            and (track["span"] is not None or track["mobile"])
            and t - track["history"][-1][0] < 0.10
        ]
        result = np.array(result, dtype=float).reshape(-1, 6)
        bounds = np.array(bounds).reshape(-1, 2)
        for left, right, speed, width in self._moving_gaps(boxes, t, scroll, reliable):
            result[left, 4] = result[right, 4] = speed
            # The 170px opening reverses near x=45 on Windows, with a
            # symmetric upper bound. Bounds refer to each inner edge.
            bounds[left] = (45, 420 - width - 45)
            bounds[right] = bounds[left] + width
        return result, bounds


@dataclass
class Shape:
    points: np.ndarray
    cx: float
    cy: float
    angle: float
    area: float


@dataclass
class MovingShape:
    points: np.ndarray
    pivot: tuple[float, float]
    velocity: tuple[float, float]
    omega: float
    uncertainty: float
    angle_limits: tuple[float, float] | None = None
    swing_angle: float = 0.0

    def predict(self, times):
        t = np.asarray(times)
        pivot = np.asarray(self.pivot)
        offsets = self.points - pivot
        angle = self.omega * t
        if self.angle_limits is not None:
            lower, upper = self.angle_limits
            # A noisy angle can sit just past an end of the swing. Start
            # the motion at that limit without rotating the observed contour
            # before any time has elapsed.
            start = np.clip(self.swing_angle, lower, upper)
            angle = reflect_position(start, self.omega, t, lower, upper) - start
        c, s = np.cos(angle), np.sin(angle)
        x = c[:, None] * offsets[None, :, 0] - s[:, None] * offsets[None, :, 1]
        y = s[:, None] * offsets[None, :, 0] + c[:, None] * offsets[None, :, 1]
        result = np.stack([x, y], axis=2) + pivot[None, None]
        result += t[:, None, None] * np.array(self.velocity)[None, None]
        return result


def centered_hammer_pin(shape):
    """Estimate a visible center-mounted hammer's pin and swing angle."""
    x, y = shape.points.T
    top = y.min()
    if shape.area <= 2500 or top < 1 or y.max() - top < 100:
        return None
    if abs(x[y <= top + 2].mean() - 210) > 19:
        return None
    # The top of the pin is about seven normalized pixels above its center.
    pin_y = top + 7
    angle = float(np.arctan2(shape.cy - pin_y, shape.cx - 210))
    if 0.4 < angle < 2.75:
        return pin_y, angle
    return None


def hammer_pin_rate(pins):
    """Estimate the swing rate through a reversal near the measured limits."""
    history = np.asarray(pins)
    ages = history[-1, 0] - history[:, 0]
    angles = history[:, 2]
    angle = angles[-1]
    observed = (angle - angles[0]) / ages[0]
    lower, upper = np.pi / 6, 5 * np.pi / 6
    if len(history) < 3 or not lower <= angle <= upper:
        return observed
    # Fit both directions against the recent pin measurements. The reflected
    # model distinguishes a recent reversal from a slow or stationary swing.
    # The measured Windows swing speed also regularizes noisy short fits.
    rates = np.array([-1.05, 1.05])
    previous = reflect_position(angle, -rates[:, None], ages[None, :], lower, upper)
    errors = np.sqrt(np.mean((previous - angles) ** 2, axis=1))
    best = int(errors.argmin())
    if errors[best] < 0.045 and errors[1 - best] - errors[best] > 0.015:
        return rates[best]
    return observed


class ShapeTracker:
    def __init__(self):
        self.tracks = []

    def update(self, shapes, t, scroll):
        next_tracks = []
        result = []
        used = set()
        for shape in shapes:
            center = np.array([shape.cx, shape.cy - scroll])
            pin = centered_hammer_pin(shape)
            pins = []
            candidates = [
                (i, track)
                for i, track in enumerate(self.tracks)
                if i not in used
                and t - track["t"] < 0.15
                and 0.65 < shape.area / track["area"] < 1.5
                and np.linalg.norm(center - track["center"]) < 70
            ]
            if candidates:
                i, track = min(candidates, key=lambda a: np.linalg.norm(center - a[1]["center"]))
                used.add(i)
                pins = track["pins"]
                # Principal-axis angles are equivalent modulo pi.
                delta = (shape.angle - track["raw_angle"] + np.pi / 2) % np.pi - np.pi / 2
                angle = track["angle"] + delta
                history = track["history"][-19:] + [(t, *center, angle)]
                if not 0.92 < shape.area / track["area"] < 1.08:
                    history = [(t, *center, angle)]
            else:
                angle = shape.angle
                history = [(t, *center, angle)]
            omega = 0.0
            angle_limits = None
            swing_angle = 0.0
            velocity = np.zeros(2)
            pivot_world = center.copy()
            uncertainty = 20.0
            if shape.points[:, 1].min() < 110:
                # Score text and the top edge can distort an incoming contour.
                history = history[-3:]
            if len(history) >= 3:
                data = np.array(history)
                ts = data[:, 0] - t
                design = np.stack([ts, np.ones(len(ts))], axis=1)
                omega = float(np.clip(np.linalg.lstsq(design, data[:, 3], rcond=None)[0][0], -8, 8))
                if shape.area > 2500:
                    rates = np.diff(data[-6:, 3]) / np.diff(data[-6:, 0])
                    omega = float(np.clip(np.sign(rates[-1]) * np.median(abs(rates)), -2.5, 2.5))
                velocity = np.linalg.lstsq(design, data[:, 1:3], rcond=None)[0][0]
                velocity = np.clip(velocity, -400, 400)
                uncertainty = 20.0
                # A rotating centroid follows a circle around a fixed pivot.
                # Fit that rigid transform once enough angular motion is seen.
                if len(history) >= 7 and np.ptp(data[:, 3]) > 0.20:
                    a = data[:, 3]
                    c, s = np.cos(a), np.sin(a)
                    design = np.zeros((2 * len(data), 4))
                    design[0::2] = np.stack([np.ones(len(a)), np.zeros(len(a)), c, -s], axis=1)
                    design[1::2] = np.stack([np.zeros(len(a)), np.ones(len(a)), s, c], axis=1)
                    values = data[:, 1:3].reshape(-1)
                    fit = np.linalg.lstsq(design, values, rcond=None)[0]
                    residual = np.sqrt(np.mean((design @ fit - values) ** 2))
                    radius = np.linalg.norm(fit[2:])
                    if residual < 1.6 and radius < 180:
                        pivot_world = fit[:2]
                        velocity[:] = 0
                        uncertainty = 2.0 + residual
                        theta = float(np.arctan2(*(center - pivot_world)[::-1]))
                        if shape.area > 2500 and 0.4 < theta < 2.75:
                            angle_limits = (np.pi / 6, 5 * np.pi / 6)
                            swing_angle = theta
            pivot = pivot_world + np.array([0, scroll])
            if pin is not None:
                pin_y, pin_angle = pin
                world_y = pin_y - scroll
                if not pins:
                    # The rolling score changes the apparent contour area as
                    # a hammer enters. Match its stationary pin independently.
                    nearby = [
                        track["pins"]
                        for track in self.tracks
                        if track["pins"]
                        and 0 < t - track["pins"][-1][0] <= 0.08
                        and abs(world_y - track["pins"][-1][1]) <= 5
                    ]
                    if nearby:
                        pins = min(nearby, key=lambda history: abs(world_y - history[-1][1]))
                pins = [
                    (ts, y, angle)
                    for ts, y, angle in pins
                    if t - ts <= 0.16 and abs(world_y - y) <= 10
                ]
                pins.append((t, world_y, pin_angle))
                pin_age = t - pins[0][0]
                if pin_age >= 0.055 and (uncertainty == 20.0 or pin_y < 145):
                    pin_rate = hammer_pin_rate(pins)
                    # A stable, directly visible pin constrains the pivot
                    # better than fitting a circle to a clipped centroid.
                    if 0.25 <= abs(pin_rate) <= 2.0 and (
                        uncertainty == 20.0 or abs(pin_rate) == 1.05
                    ):
                        pivot = (210.0, pin_y)
                        velocity[:] = 0.0
                        omega = pin_rate
                        uncertainty = min(uncertainty, 12.0)
                        angle_limits = (np.pi / 6, 5 * np.pi / 6)
                        swing_angle = pin_angle
            result.append(
                MovingShape(
                    shape.points,
                    tuple(pivot),
                    tuple(velocity),
                    omega,
                    uncertainty,
                    angle_limits,
                    swing_angle,
                )
            )
            next_tracks.append(
                {
                    "t": t,
                    "center": center,
                    "area": shape.area,
                    "raw_angle": shape.angle,
                    "angle": angle,
                    "history": history,
                    "pins": pins,
                }
            )
        self.tracks = next_tracks
        return result


def collides(px, py, shapes, times, half_width, half_height, margin):
    """Vectorized separating-axis test against predicted convex hulls.

    px/py have shape (candidate, time). Shape predictions share a time axis.
    Convex hulls conservatively include the T-shaped hammer's inner notches.
    """
    collision = np.zeros(px.shape, dtype=bool)
    clearance = np.full(px.shape[0], 100.0)
    for shape in shapes:
        polygon = shape.predict(times)
        mx = margin + np.minimum(12.0, shape.uncertainty * np.asarray(times))
        minx, maxx = polygon[:, :, 0].min(1), polygon[:, :, 0].max(1)
        miny, maxy = polygon[:, :, 1].min(1), polygon[:, :, 1].max(1)
        overlap = (
            (px + half_width > minx - mx)
            & (px - half_width < maxx + mx)
            & (py + half_height > miny - mx)
            & (py - half_height < maxy + mx)
        )
        edges = np.roll(polygon, -1, axis=1) - polygon
        normals = np.stack([-edges[:, :, 1], edges[:, :, 0]], axis=2)
        norms = np.linalg.norm(normals, axis=2)
        normals /= np.maximum(norms[:, :, None], 1e-9)
        projected = np.einsum("kvc,kec->kev", polygon, normals)
        lo, hi = projected.min(2), projected.max(2)
        center = px[:, :, None] * normals[None, :, :, 0] + py[:, :, None] * normals[None, :, :, 1]
        radius = (
            half_width * abs(normals[:, :, 0]) + half_height * abs(normals[:, :, 1]) + mx[:, None]
        )
        separated = (center + radius < lo[None]) | (center - radius > hi[None])
        collision |= overlap & ~separated.any(2)
        dx = np.maximum(np.maximum(minx - (px + half_width), (px - half_width) - maxx), 0)
        dy = np.maximum(np.maximum(miny - (py + half_height), (py - half_height) - maxy), 0)
        clearance = np.minimum(clearance, np.sqrt(dx * dx + dy * dy).min(1))
    return collision.any(1), clearance
