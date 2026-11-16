"""Compiled deterministic beam search; no learned model or training."""

import math

import numpy as np
from numba import njit


@njit(cache=True)
def lookup(grid, x, y, spacing):
    gx = min(max(x / spacing, 0.0), grid.shape[1] - 1.001)
    gy = min(max(y / spacing, 0.0), grid.shape[0] - 1.001)
    ix, iy = int(gx), int(gy)
    ax, ay = gx - ix, gy - iy
    return (1 - ay) * ((1 - ax) * grid[iy, ix] + ax * grid[iy, ix + 1]) + ay * (
        (1 - ax) * grid[iy + 1, ix] + ax * grid[iy + 1, ix + 1]
    )


@njit(cache=True)
def search(
    initial,
    obstacles,
    grid,
    shape_bounds,
    shape_axes,
    shape_limits,
    edge_counts,
    speed,
    jump,
    gravity,
    latency,
    step,
    levels,
    beam,
    margin,
    cooldown,
    click_cost,
    height,
    substeps=5,
):
    # Fields: x,y,vx,vy,age,cost,progress integral,first action,terminal,peak y.
    states = np.zeros((levels + 1, beam, 10))
    parents = np.zeros((levels + 1, beam), np.int64)
    actions = np.zeros((levels + 1, beam), np.int64)
    x0 = initial[0] + initial[2] * latency
    y0 = initial[1] + initial[3] * latency + 0.5 * gravity * latency * latency
    states[0, 0, :5] = np.array(
        [x0, y0, initial[2], initial[3] + gravity * latency, initial[6] + latency]
    )
    anchor = min(height / 2, initial[1])
    states[0, 0, 9] = min(anchor, y0)
    count = 1
    hw, hh = initial[4] / 2, initial[5] / 2
    frontier = hh + 8
    best_depth = 0
    best_index = 0
    best_score = -1e30
    for depth in range(levels):
        candidates = np.zeros((count * 3, 10))
        scores = np.full(count * 3, -1e30)
        candidate_parent = np.zeros(count * 3, np.int64)
        candidate_action = np.zeros(count * 3, np.int64)
        total = 0
        for i in range(count):
            prev = states[depth, i]
            for choice in range(3):
                action = 0 if choice == 0 else -1 if choice == 1 else 1
                if action != 0 and (prev[4] + 1e-6 < cooldown or prev[8] > 0.5):
                    continue
                vx = action * speed if action else prev[2]
                vy = -jump if action else prev[3]
                crossing = 1e20
                disc = vy * vy - 2 * gravity * (prev[1] - frontier)
                if vy < 0 and disc >= 0:
                    crossing = max(0.0, (-vy - math.sqrt(disc)) / gravity)
                if prev[8] > 0.5:
                    crossing = 0.0
                terminal = prev[8] > 0.5 or crossing <= step
                safe = True
                clearance = 100.0
                ex, ey = prev[0], prev[1]
                low_y = prev[9]
                for sample in range(substeps + 1):
                    local = min(step * sample / substeps, crossing)
                    x = prev[0] + vx * local
                    y = prev[1] + vy * local + 0.5 * gravity * local * local
                    ex, ey = x, y
                    low_y = min(low_y, y)
                    screen_y = y + max(0.0, anchor - low_y)
                    if (
                        x <= hw + margin
                        or x >= 420 - hw - margin
                        or screen_y >= height - hh - margin
                        or y <= hh + margin
                    ):
                        safe = False
                        break
                    clearance = min(clearance, x - hw, 420 - x - hw)
                    world_t = latency + depth * step + local
                    jitter = 0.025 * min(1.0, (depth * step + local) / 0.15)
                    for o in range(len(obstacles)):
                        b = obstacles[o]
                        bx, by = b[0] + b[4] * world_t, b[1] + b[5] * world_t
                        mx = margin + abs(vx - b[4]) * (step / substeps / 2 + jitter)
                        my = margin + abs(vy + gravity * local - b[5]) * (
                            step / substeps / 2 + jitter
                        )
                        if (
                            x + hw > bx - mx
                            and x - hw < bx + b[2] + mx
                            and y + hh > by - my
                            and y - hh < by + b[3] + my
                        ):
                            safe = False
                            break
                        dx = max(bx - x - hw, x - hw - bx - b[2], 0.0)
                        dy = max(by - y - hh, y - hh - by - b[3], 0.0)
                        clearance = min(clearance, math.sqrt(dx * dx + dy * dy))
                    if not safe:
                        break
                    if prev[8] < 0.5:
                        tick = depth * (substeps + 1) + sample
                        for h in range(shape_bounds.shape[1]):
                            bounds = shape_bounds[tick, h]
                            dm = margin + 5.0 + min(12.0, bounds[4] * world_t)
                            if not (
                                x + hw > bounds[0] - dm
                                and x - hw < bounds[2] + dm
                                and y + hh > bounds[1] - dm
                                and y - hh < bounds[3] + dm
                            ):
                                continue
                            separate = False
                            for edge in range(edge_counts[h]):
                                nx, ny = shape_axes[tick, h, edge]
                                projection = x * nx + y * ny
                                radius = hw * abs(nx) + hh * abs(ny) + dm
                                lo, hi = shape_limits[tick, h, edge]
                                if projection + radius < lo or projection - radius > hi:
                                    separate = True
                                    break
                            if not separate:
                                safe = False
                                break
                    if not safe:
                        break
                if not safe:
                    continue
                edge_cost = 0.5 * max(0.0, 60.0 - ex, ex - 360.0) * step
                cost = prev[5] + (click_cost if action else 0.0) + 1.2 / (clearance + 1) + edge_cost
                area = prev[6] + (y0 - ey) * step
                first = action if depth == 0 else int(prev[7])
                nav = lookup(grid, ex, ey, 10.0)
                score = (
                    -0.9 * nav + 0.2 * (y0 - ey) + 0.05 * area - cost + 0.12 * min(clearance, 40.0)
                )
                candidates[total] = np.array(
                    [
                        ex,
                        ey,
                        0.0 if terminal else vx,
                        0.0 if terminal else vy + gravity * step,
                        (0.0 if action else prev[4]) + step,
                        cost,
                        area,
                        first,
                        1.0 if terminal else 0.0,
                        low_y,
                    ]
                )
                scores[total] = score
                candidate_parent[total] = i
                candidate_action[total] = action
                total += 1
        if total == 0:
            break
        order = np.argsort(scores[:total])[::-1]
        group_counts = np.zeros(3, np.int64)
        seen = set()
        count = 0
        for j in order:
            a = candidates[j]
            group = int(a[7]) + 1
            if group_counts[group] >= beam // 3:
                continue
            xb = int(round(a[0] / 8))
            yb = int(round(a[1] / 10))
            vb = int(round(a[3] / 70)) + 32
            sign = 0 if a[2] < -1 else 2 if a[2] > 1 else 1
            age = min(3, int(round(a[4] / step)))
            key = (
                ((((xb * 256 + yb) * 3 + sign) * 128 + vb) * 4 + age) * 2 + int(a[8])
            ) * 3 + group
            key = key * 128 + int(round(a[9] / 8))
            if key in seen:
                continue
            seen.add(key)
            states[depth + 1, count] = a
            parents[depth + 1, count] = candidate_parent[j]
            actions[depth + 1, count] = candidate_action[j]
            if count == 0:
                best_index = 0
                best_score = scores[j]
            group_counts[group] += 1
            count += 1
            if count >= beam:
                break
        best_depth = depth + 1
    if best_depth == 0:
        return 0, np.empty((0, 2)), 0, best_score
    path = np.empty((best_depth, 2))
    first_action = 0
    index = best_index
    for d in range(best_depth, 0, -1):
        path[d - 1] = states[d, index, :2]
        first_action = actions[d, index]
        index = parents[d, index]
    return first_action, path, best_depth, best_score


def shape_arrays(shapes, times):
    counts = np.array([len(h.points) for h in shapes], dtype=np.int64)
    edges = max(counts, default=0)
    bounds = np.zeros((len(times), len(shapes), 5))
    axes = np.zeros((len(times), len(shapes), edges, 2))
    limits = np.zeros_like(axes)
    for i, h in enumerate(shapes):
        polygon = h.predict(times)
        bounds[:, i, 0:2] = polygon.min(1)
        bounds[:, i, 2:4] = polygon.max(1)
        bounds[:, i, 4] = h.uncertainty
        edge = np.roll(polygon, -1, axis=1) - polygon
        normal = np.stack([-edge[:, :, 1], edge[:, :, 0]], axis=2)
        normal /= np.maximum(1e-9, np.linalg.norm(normal, axis=2)[:, :, None])
        projected = np.einsum("kvc,kec->kev", polygon, normal)
        axes[:, i, : len(h.points)] = normal
        limits[:, i, : len(h.points), 0] = projected.min(2)
        limits[:, i, : len(h.points), 1] = projected.max(2)
    return bounds, axes, limits, counts
