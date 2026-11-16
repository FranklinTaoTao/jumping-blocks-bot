"""Rigid-shape tracking for rotating hammers and other nonrectangular hazards."""

from dataclasses import dataclass

import numpy as np


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

    def predict(self, times):
        t = np.asarray(times)
        pivot = np.asarray(self.pivot)
        offsets = self.points - pivot
        angle = self.omega * t
        c, s = np.cos(angle), np.sin(angle)
        x = c[:, None] * offsets[None, :, 0] - s[:, None] * offsets[None, :, 1]
        y = s[:, None] * offsets[None, :, 0] + c[:, None] * offsets[None, :, 1]
        result = np.stack([x, y], axis=2) + pivot[None, None]
        result += t[:, None, None] * np.array(self.velocity)[None, None]
        return result


class ShapeTracker:
    def __init__(self):
        self.tracks = []

    def update(self, shapes, t, scroll):
        next_tracks = []
        result = []
        used = set()
        for shape in shapes:
            center = np.array([shape.cx, shape.cy - scroll])
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
                # Principal-axis angles are equivalent modulo pi.
                delta = (shape.angle - track["raw_angle"] + np.pi / 2) % np.pi - np.pi / 2
                angle = track["angle"] + delta
                history = track["history"][-19:] + [(t, *center, angle)]
            else:
                angle = shape.angle
                history = [(t, *center, angle)]
            omega = 0.0
            velocity = np.zeros(2)
            pivot_world = center.copy()
            uncertainty = 10.0
            if len(history) >= 3:
                data = np.array(history)
                ts = data[:, 0] - t
                design = np.stack([ts, np.ones(len(ts))], axis=1)
                omega = float(np.clip(np.linalg.lstsq(design, data[:, 3], rcond=None)[0][0], -8, 8))
                velocity = np.linalg.lstsq(design, data[:, 1:3], rcond=None)[0][0]
                velocity = np.clip(velocity, -400, 400)
                uncertainty = 4.0
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
            pivot = pivot_world + np.array([0, scroll])
            result.append(
                MovingShape(shape.points, tuple(pivot), tuple(velocity), omega, uncertainty)
            )
            next_tracks.append(
                {
                    "t": t,
                    "center": center,
                    "area": shape.area,
                    "raw_angle": shape.angle,
                    "angle": angle,
                    "history": history,
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
