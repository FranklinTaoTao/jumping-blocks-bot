"""Geometric lookahead around inflated rectangles, used as a search heuristic.

The visibility graph suggests routes; the ballistic planner still checks actual
reachable trajectories and collisions. No straight-line mouse steering is used.
"""

import heapq

import numpy as np


class Navigation:
    def __init__(self, state, margin=6.0):
        hw, hh = state.w / 2 + margin, state.h / 2 + margin
        self.bounds = hw, 420 - hw
        self.frontier = hh + 3
        obs = state.obstacles
        if state.shapes:
            boxes = []
            for shape in state.shapes:
                lo, hi = shape.points.min(0), shape.points.max(0)
                boxes.append([*lo, *(hi - lo), 0.0, 0.0])
            obs = np.concatenate([obs, np.array(boxes)], axis=0)
        self.rects = (
            np.stack(
                [
                    obs[:, 0] - hw,
                    obs[:, 1] - hh,
                    obs[:, 0] + obs[:, 2] + hw,
                    obs[:, 1] + obs[:, 3] + hh,
                ],
                axis=1,
            )
            if len(obs)
            else np.empty((0, 4))
        )
        goals = [(x, self.frontier) for x in np.arange(hw + 2, 420 - hw, 16)]
        corners = []
        for left, top, right, bottom in self.rects:
            corners.extend(
                [
                    (left - 1, top - 1),
                    (right + 1, top - 1),
                    (left - 1, bottom + 1),
                    (right + 1, bottom + 1),
                ]
            )
        goals = [p for p in goals if self.free(p)]
        corners = [
            p
            for p in corners
            if hw < p[0] < 420 - hw and self.frontier <= p[1] < state.y + 260 and self.free(p)
        ]
        self.nodes = np.array(goals + corners, dtype=float).reshape(-1, 2)
        self.distance = np.full(len(self.nodes), np.inf)
        if len(self.nodes) and goals:
            visible = self.visible(self.nodes)
            lengths = np.linalg.norm(self.nodes[:, None] - self.nodes[None, :], axis=2)
            lengths[~visible] = np.inf
            # Prefer a centered exit into unseen space, rather than racing up
            # along a screen edge where the next opening may be unreachable.
            target = 210.0
            edge_bars = obs[
                ((obs[:, 0] < 4) | (obs[:, 0] + obs[:, 2] > 416))
                & (obs[:, 1] < 130)
                & (obs[:, 3] >= 6)
                & (obs[:, 3] <= 42)
            ]
            if len(edge_bars):
                top = edge_bars[:, 1].min()
                row = edge_bars[abs(edge_bars[:, 1] - top) < 5]
                left = max((b[0] + b[2] for b in row if b[0] < 4), default=0)
                right = min((b[0] for b in row if b[0] + b[2] > 416), default=420)
                target = (left + right) / 2
            goal_costs = [0.35 * abs(point[0] - target) for point in goals]
            pending = [(float(d), i) for i, d in enumerate(goal_costs)]
            heapq.heapify(pending)
            self.distance[: len(goals)] = goal_costs
            while pending:
                d, i = heapq.heappop(pending)
                if d > self.distance[i]:
                    continue
                for j in np.flatnonzero(np.isfinite(lengths[i])):
                    nd = d + lengths[i, j]
                    if nd < self.distance[j]:
                        self.distance[j] = nd
                        heapq.heappush(pending, (nd, j))

    def free(self, point):
        x, y = point
        return not np.any(
            (self.rects[:, 0] <= x)
            & (x <= self.rects[:, 2])
            & (self.rects[:, 1] <= y)
            & (y <= self.rects[:, 3])
        )

    def visible(self, points):
        origin = np.asarray(points)[:, None, :]
        delta = self.nodes[None, :, :] - origin
        result = np.ones(delta.shape[:2], dtype=bool)
        for left, top, right, bottom in self.rects:
            enter = np.zeros(delta.shape[:2])
            leave = np.ones(delta.shape[:2])
            for axis, lo, hi in [(0, left, right), (1, top, bottom)]:
                step = delta[:, :, axis]
                start = origin[:, :, axis]
                parallel = abs(step) < 1e-9
                denom = np.where(parallel, 1.0, step)
                a, z = (lo - start) / denom, (hi - start) / denom
                lower, upper = np.minimum(a, z), np.maximum(a, z)
                inside = (start >= lo) & (start <= hi)
                lower = np.where(parallel, np.where(inside, -np.inf, np.inf), lower)
                upper = np.where(parallel, np.where(inside, np.inf, -np.inf), upper)
                enter = np.maximum(enter, lower)
                leave = np.minimum(leave, upper)
            result &= enter > leave
        return result

    def cost(self, x, y):
        if not len(self.nodes) or not np.isfinite(self.distance).any():
            return y - self.frontier
        points = np.stack([x, y], axis=1)
        lengths = np.linalg.norm(points[:, None, :] - self.nodes[None, :, :], axis=2)
        lengths += self.distance[None, :]
        lengths[~self.visible(points)] = np.inf
        best = lengths.min(1)
        # Near a safety-margin boundary, the geometric graph may have no edge.
        return np.where(np.isfinite(best), best, 1000 + abs(y - self.frontier))
