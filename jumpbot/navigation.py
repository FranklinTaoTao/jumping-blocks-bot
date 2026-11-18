"""Geometric lookahead around inflated rectangles, used as a search heuristic.

The visibility graph suggests routes; the ballistic planner still checks actual
reachable trajectories and collisions. No straight-line mouse steering is used.
"""

import heapq

import numpy as np
from numba import njit


def bar_gaps(obstacles, *, include_moving=False):
    """Return visible matched rows as (top, bottom, opening_left, opening_right)."""
    bars = [box for box in obstacles if 30 <= box[3] <= 42 and (include_moving or abs(box[4]) < 15)]
    gaps = []
    for left in bars:
        if left[0] >= 4:
            continue
        for right in bars:
            start, end = left[0] + left[2], right[0]
            if right[0] + right[2] > 416 and abs(right[1] - left[1]) <= 5 and start < end:
                gaps.append(
                    (
                        min(left[1], right[1]),
                        max(left[1] + left[3], right[1] + right[3]),
                        start,
                        end,
                    )
                )
    return sorted(gaps, reverse=True)


def square_gaps(obstacles):
    """Return narrow openings between paired large squares."""
    squares = [box for box in obstacles if 85 <= box[2] <= 110 and 75 <= box[3] <= 110]
    return [
        (
            min(left[1], right[1]),
            max(left[1] + left[3], right[1] + right[3]),
            left[0] + left[2],
            right[0],
        )
        for left in squares
        for right in squares
        if 60 <= right[0] - left[0] - left[2] <= 100 and abs(left[1] - right[1]) <= 6
    ]


def clipped_bar_gaps(obstacles):
    """Recognize the two edges of an incoming bar before it is fully visible."""
    clipped = [b for b in obstacles if 0 <= b[1] <= 3 and 8 <= b[3] <= 42]
    gaps = []
    for left in clipped:
        if left[0] >= 4:
            continue
        for right in clipped:
            start, end = left[0] + left[2], right[0]
            bottom = max(left[1] + left[3], right[1] + right[3])
            if (
                right[0] + right[2] > 416
                and min(left[3], right[3]) < 30
                and abs(left[1] + left[3] - right[1] - right[3]) <= 3
                and 150 <= end - start <= 190
            ):
                gaps.append((bottom - 38, bottom, start, end))
    return gaps


def opening_guidance(state, xx, yy, margin, speed, jump, cooldown):
    """Cost of an exit pose for leftward, stopped and rightward trajectories."""
    half_width = state.w / 2 + margin + 4
    half_height = state.h / 2 + margin
    stable = np.max(np.abs(state.obstacles[:, 4:]), axis=1) <= 10
    if len(state.obstacle_bounds) == len(state.obstacles):
        stable &= state.obstacle_bounds[:, 1] <= state.obstacle_bounds[:, 0]
    static = state.obstacles[stable]
    candidates = bar_gaps(static) + square_gaps(static) + clipped_bar_gaps(static)
    openings = [
        (top - half_height, bottom + half_height, left + half_width, right - half_width)
        for top, bottom, left, right in candidates
        if right - left > 2 * half_width and top - half_height < state.y + 65
    ]
    openings.sort(reverse=True)

    # Keep a broad, nearly flat preference in the middle. When an opening
    # supplies a destination, its horizontal geometry takes priority.
    center = 1 - np.exp(-0.5 * ((xx - 210) / 100) ** 2)
    unknown = np.clip((openings[-1][0] - yy + 12) / 35, 0.12, 1) if openings else 1.0
    costs = np.broadcast_to(36 * center * unknown, (3, *xx.shape)).copy()
    drift = speed * min(0.24, cooldown + 0.04)
    projected_x = np.clip(xx[None] + drift * np.array([-1, 0, 1])[:, None, None], 0, 420)

    def distance(x, left, right):
        return np.maximum(np.maximum(left - x, x - right), 0)

    # The topmost visible exit has room to keep options open for unseen terrain.
    if openings:
        _, _, left, right = openings[-1]
        goal = np.clip((165.0, 255.0), left, right)
    for i in range(len(openings) - 2, -1, -1):
        current, upper = openings[i], openings[i + 1]
        exit_y, enter_y, left, right = current
        _, upper_entry, upper_left, upper_right = upper
        height = exit_y - upper_entry
        if height <= 0:
            goal = np.clip(goal, left, right)
            continue
        # Back-propagate the upper entry aperture using the horizontal travel
        # available at the fastest ascent. Forward search still checks the arc
        # and can delay ascent when the openings have less overlap.
        travel = speed * height / max(1.0, jump)
        feasible = np.clip((upper_left - travel, upper_right + travel), left, right)
        target = np.clip(goal, *feasible)
        stage = np.clip((enter_y + 60 - yy) / (enter_y + 60 - exit_y), 0, 1)
        stage *= np.clip((yy - upper_entry) / min(55, height), 0, 1)
        approach = distance(xx, *feasible) * 0.35 + distance(xx, *target) * 0.19
        wrong_way = np.maximum(
            distance(projected_x, upper_left, upper_right) - distance(xx, upper_left, upper_right),
            0,
        )
        costs += stage * (approach + 0.45 * wrong_way)
        goal = target
    return costs


@njit(cache=True)
def route_cost(points, nodes, distances, rects, frontier):
    """Evaluate visibility without allocating a grid-by-node matrix per wall."""
    result = np.empty(len(points))
    usable = np.any(np.isfinite(distances))
    for i in range(len(points)):
        x, y = points[i]
        if not len(nodes) or not usable:
            result[i] = y - frontier
            continue
        best = np.inf
        for j in range(len(nodes)):
            if not np.isfinite(distances[j]):
                continue
            dx, dy = nodes[j, 0] - x, nodes[j, 1] - y
            cost = np.sqrt(dx * dx + dy * dy) + distances[j]
            if cost >= best:
                continue
            clear = True
            for r in range(len(rects)):
                enter, leave = 0.0, 1.0
                for axis in range(2):
                    start = x if axis == 0 else y
                    step = dx if axis == 0 else dy
                    lo, hi = rects[r, axis], rects[r, axis + 2]
                    if abs(step) < 1e-9:
                        if start < lo or start > hi:
                            enter, leave = 1.0, 0.0
                            break
                    else:
                        a, b = (lo - start) / step, (hi - start) / step
                        enter = max(enter, min(a, b))
                        leave = min(leave, max(a, b))
                if enter <= leave:
                    clear = False
                    break
            if clear:
                best = cost
        result[i] = best if np.isfinite(best) else 1000 + abs(y - frontier)
    return result


class Navigation:
    def __init__(self, state, margin=6.0, jump_clearance=0.0):
        hw, hh = state.w / 2 + margin, state.h / 2 + margin
        self.bounds = 0.0, 420.0
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
        self.jump_shadows = []
        if jump_clearance:
            for i, b in enumerate(obs):
                if (b[0] < 4 or b[0] + b[2] > 416) and 30 <= b[3] <= 42:
                    # Evaluate future positions under every bar, including bars
                    # the character has not yet entered beneath. A turn also
                    # jumps, so these pockets need room for the upward impulse.
                    # Native collision checks still use the actual rectangle.
                    self.jump_shadows.append(
                        (self.rects[i, 0], self.rects[i, 2], self.rects[i, 3], jump_clearance)
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
        points = np.stack([x, y], axis=1)
        costs = route_cost(points, self.nodes, self.distance, self.rects, self.frontier)
        for left, right, bottom, rise in self.jump_shadows:
            sideways = np.minimum(np.maximum(0, x - left), np.maximum(0, right - x))
            shortfall = np.maximum(0, bottom + rise - y)
            below = y >= bottom - 3
            costs += 2 * shortfall * np.minimum(1, sideways / 120) * below
        return costs
