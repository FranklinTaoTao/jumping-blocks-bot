from dataclasses import dataclass

import numpy as np

from .motion import collides
from .navigation import Navigation
from .physics import Physics, State, project_camera


@dataclass
class Decision:
    action: int  # -1 left, 0 wait, +1 right
    path: np.ndarray
    reason: str
    score: float = -float("inf")
    predicted_horizon: float = 0.0


class Planner:
    """Receding-horizon beam search over wait/left/right velocity resets.

    Every candidate is sampled at <=15ms, with velocity-dependent clearance
    inflation to cover motion between samples. No game internals or ML.
    """

    def __init__(
        self,
        physics: Physics,
        horizon=1.8,
        step=0.06,
        beam=150,
        margin=4.0,
        cooldown=0.12,
        click_cost=12.0,
    ):
        self.p = physics
        self.horizon, self.step, self.beam = horizon, step, beam
        self.margin, self.cooldown = margin, cooldown
        self.click_cost = click_cost

    def plan(self, s: State, board_height, latency=0.02):
        if not s.ready:
            return Decision(0, np.empty((0, 2)), "waiting for reliable tracking")
        dt = self.step
        count = max(1, round(self.horizon / dt))
        # Advance the observation to estimated input delivery time.
        x = np.array([s.x + s.vx * latency])
        y = np.array([s.y + s.vy * latency + 0.5 * self.p.gravity * latency**2])
        vx, vy = np.array([s.vx]), np.array([s.vy + self.p.gravity * latency])
        age = np.array([s.age_since_click + latency])
        first = np.array([0], dtype=int)
        cost = np.array([0.0])
        terminal = np.array([False])
        progress_area = np.array([0.0])
        paths = np.empty((1, 0, 2))
        initial_y = y[0]
        anchor = min(board_height / 2, s.y)
        peak = np.array([min(anchor, y[0])])
        navigation = Navigation(s, margin=self.margin + 1)
        # Extra samples keep even thin moving squares from tunnelling.
        sample_t = np.linspace(0, dt, max(5, int(np.ceil(dt / 0.015)) + 1))[None, :]
        best = None
        for depth in range(count):
            n = len(x)
            parent = np.repeat(np.arange(n), 3)
            action = np.tile([0, -1, 1], n)
            allowed = (action == 0) | ((age[parent] >= self.cooldown) & ~terminal[parent])
            parent, action = parent[allowed], action[allowed]
            xx, yy = x[parent], y[parent]
            vxx = np.where(action != 0, action * self.p.speed, vx[parent])
            vyy = np.where(action != 0, -self.p.jump, vy[parent])
            ages = np.where(action != 0, 0.0, age[parent]) + dt
            frontier = 8 + s.h / 2
            discriminant = vyy * vyy - 2 * self.p.gravity * (yy - frontier)
            crossing = np.where(
                (vyy < 0) & (discriminant >= 0),
                (-vyy - np.sqrt(np.maximum(0, discriminant))) / self.p.gravity,
                np.inf,
            )
            crossing = np.maximum(0, crossing)
            crossing = np.where(terminal[parent], 0.0, crossing)
            local_t = np.minimum(sample_t, crossing[:, None])
            px = xx[:, None] + vxx[:, None] * local_t
            py = yy[:, None] + vyy[:, None] * local_t + 0.5 * self.p.gravity * local_t**2
            screen_y, peaks = project_camera(py, anchor, peak[parent])
            time = latency + depth * dt + local_t
            safe = (
                (px > s.w / 2 + self.margin)
                & (px < 420 - s.w / 2 - self.margin)
                & (screen_y < board_height - s.h / 2 - self.margin)
                & (py > s.h / 2 + self.margin)
            ).all(1)
            # Reaching the visible frontier is a completed partial plan, not a
            # failure. Freeze it there instead of inventing unseen obstacles or
            # making the controller perpetually postpone its first jump.
            terminals = terminal[parent] | (crossing <= dt)
            clearance = np.full(len(parent), 100.0)
            for bx, by, bw, bh, bvx, bvy in s.obstacles:
                ox, oy = bx + bvx * time, by + bvy * time
                # Bound relative displacement between adjacent samples.
                sample_gap = dt / (sample_t.shape[1] - 1)
                # Future turns cannot rely on sub-frame-perfect delivery.
                # Grow a 25ms control-timing envelope over the first 150ms.
                jitter = 0.025 * np.minimum(1.0, (depth * dt + local_t) / 0.15)
                mx = self.margin + abs(vxx - bvx)[:, None] * (sample_gap / 2 + jitter)
                my = self.margin + (abs(vyy[:, None] + self.p.gravity * local_t - bvy)) * (
                    sample_gap / 2 + jitter
                )
                overlap_x = (px + s.w / 2 > ox - mx) & (px - s.w / 2 < ox + bw + mx)
                overlap_y = (py + s.h / 2 > oy - my) & (py - s.h / 2 < oy + bh + my)
                safe &= ~(overlap_x & overlap_y).any(1)
                dx = np.maximum(np.maximum(ox - (px + s.w / 2), (px - s.w / 2) - (ox + bw)), 0)
                dy = np.maximum(np.maximum(oy - (py + s.h / 2), (py - s.h / 2) - (oy + bh)), 0)
                clearance = np.minimum(clearance, np.sqrt(dx * dx + dy * dy).min(1))
            if s.shapes:
                moving_hit, moving_clearance = collides(
                    px,
                    py,
                    s.shapes,
                    (latency + depth * dt + sample_t).reshape(-1),
                    s.w / 2,
                    s.h / 2,
                    self.margin + 5.0,
                )
                safe &= ~moving_hit | terminal[parent]
                clearance = np.minimum(clearance, moving_clearance)
            if not safe.any():
                # Prefer the action surviving longest if all full-horizon plans
                # fail. Replanning next frame may expose a new escape route.
                return best or Decision(0, np.empty((0, 2)), "no collision-free immediate action")
            xx = px[:, -1]
            yy = py[:, -1]
            costs = cost[parent] + (action != 0) * self.click_cost + 1.2 / (clearance + 1)
            areas = progress_area[parent] + (initial_y - yy) * dt
            # Reward ascent, and approach the next opening before reaching it.
            guidance = navigation.cost(xx, yy)
            scores = (
                -0.9 * guidance
                + 0.2 * (initial_y - yy)
                + 0.05 * areas
                - costs
                + 0.12 * np.minimum(clearance, 40)
            )
            valid = np.flatnonzero(safe)
            # Preserve alternatives for each immediate action, preventing early
            # pruning of a necessary wait while an immediate jump rises faster.
            firsts = action if depth == 0 else first[parent]
            selected = []
            for a in [0, -1, 1]:
                group = valid[firsts[valid] == a]
                # Duplicate action histories can converge to the same state.
                # Keep distinct positions/velocities, rather than spending the
                # whole beam on near-identical greedy upward trajectories.
                ordered = group[np.argsort(scores[group])[::-1]]
                keys = np.stack(
                    [
                        np.round(xx[ordered] / 10),
                        np.round(yy[ordered] / 12),
                        np.sign(vxx[ordered]),
                        np.round((vyy[ordered] + self.p.gravity * dt) / 90),
                        np.minimum(3, np.round(ages[ordered] / 0.06)),
                        terminals[ordered],
                        np.round(peaks[ordered, -1] / 10),
                    ],
                    axis=1,
                )
                _, unique = np.unique(keys, axis=0, return_index=True)
                retained = ordered[np.sort(unique)[: max(1, self.beam // 3)]]
                selected.extend(retained.tolist())
            idx = np.asarray(selected, dtype=int)
            paths = np.concatenate(
                [paths[parent[idx]], np.stack([xx[idx], yy[idx]], axis=1)[:, None]], axis=1
            )
            terminal = terminals[idx]
            x, y = xx[idx], yy[idx]
            vx = np.where(terminal, 0.0, vxx[idx])
            vy = np.where(terminal, 0.0, vyy[idx] + self.p.gravity * dt)
            progress_area = areas[idx]
            peak = peaks[idx, -1]
            age, first, cost = ages[idx], firsts[idx], costs[idx]
            k = int(np.argmax(scores[idx]))
            best = Decision(
                int(first[k]),
                paths[k].copy(),
                f"predicted {(depth + 1) * dt:.2f}s",
                float(scores[idx[k]]),
                (depth + 1) * dt,
            )
        return best
