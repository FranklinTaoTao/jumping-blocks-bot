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
def advance_scene(scene, before, after, dt, tau, decay=-1.0):
    """Exact first-order camera response to a linearly changing target offset."""
    if tau <= 0:
        return after
    if dt <= 1e-9:
        return scene
    speed = (after - before) / dt
    if decay < 0:
        decay = math.exp(-dt / tau)
    return after - tau * speed + (scene - before + tau * speed) * decay


@njit(cache=True)
def reflect_position(position, velocity, t, lower, upper):
    span = upper - lower
    phase = (position + velocity * t - lower) % (2 * span)
    return lower + min(phase, 2 * span - phase)


@njit(inline="always")
def reflected_extent(x, width, velocity, t, lower, upper):
    if x < 4:
        return x, reflect_position(x + width, velocity, t, lower, upper) - x
    right = x + width
    start = reflect_position(x, velocity, t, lower, upper)
    return start, right - start if right > 416 else width


@njit(cache=True)
def reset_position_error(before_velocity, current_velocity, gravity, timing_error):
    """Bound the displacement caused by an uncertain time of velocity reset."""
    return abs(before_velocity - current_velocity) * timing_error + gravity * timing_error**2


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
    control_interval=0.05,
    camera_tau=0.0,
    camera_speed=0.0,
    vertical_uncertainty=0.0,
    obstacle_bounds=None,
    wall_penalty=0.05,
    timing_uncertainty=0.025,
    planned_actions=None,
    vertical_error_floor=0.0,
    reset_timing_uncertainty=0.0,
    first_action=2,
    root_quality=None,
    pocket_penalty=0.0,
    blocked_action=2,
    root_mask=7,
    single_input=False,
    route_values=None,
    underside_margin=0.0,
    opening_goal=None,
):
    if root_quality is not None:
        root_quality[:, 0] = 0
        root_quality[:, 1] = -1e30
    # Fields: x,y,vx,vy,age,cost,progress integral,first action,terminal,peak y,scene.
    fields = 12 if reset_timing_uncertainty else 11
    states = np.zeros((levels + 1, beam, fields))
    if reset_timing_uncertainty:
        states[0, 0, 11] = np.nan
    parents = np.zeros((levels + 1, beam), np.int64)
    actions = np.zeros((levels + 1, beam), np.int64)
    x0 = min(420.0, max(0.0, initial[0] + initial[2] * latency))
    anchor = min(height / 2, initial[1])
    lag = max(0.0, camera_speed) * camera_tau
    physical_y = initial[1] - lag
    physical_vy = initial[3]
    if camera_speed > 1 and initial[1] <= height / 2 + 3:
        physical_vy += camera_tau * gravity
    y0 = physical_y + physical_vy * latency + 0.5 * gravity * latency * latency
    states[0, 0, :5] = np.array(
        [x0, y0, initial[2], physical_vy + gravity * latency, initial[6] + latency]
    )
    initial_peak = min(anchor - lag, y0)
    states[0, 0, 9] = initial_peak
    scene = 0.0
    peak = min(anchor - lag, physical_y)
    for k in range(1, 5):
        tick = latency * k / 4
        previous_target = max(0.0, anchor - peak)
        peak = min(peak, physical_y + physical_vy * tick + 0.5 * gravity * tick * tick)
        scene = advance_scene(
            scene, previous_target, max(0.0, anchor - peak), latency / 4, camera_tau
        )
    states[0, 0, 10] = scene
    count = 1
    hw, hh = initial[4] / 2, initial[5] / 2
    edge_bars = np.empty((len(obstacles), 3))
    bar_count = 0
    if pocket_penalty:
        for bx, by, bw, bh, vx, vy in obstacles:
            if 30 <= bh <= 42 and vx == 0 and vy == 0 and (bx < 4 or bx + bw > 416):
                edge_bars[bar_count, 0] = bx - hw - margin
                edge_bars[bar_count, 1] = bx + bw + hw + margin
                edge_bars[bar_count, 2] = by + bh + hh + margin
                bar_count += 1
    rise = jump * jump / (2 * gravity)
    frontier = hh + 8
    best_depth = 0
    best_index = 0
    best_score = -1e30
    x_bin, y_bin, velocity_bin = (4, 4, 25) if opening_goal is not None else (8, 10, 70)
    sample_dt = step / substeps
    sample_decay = math.exp(-sample_dt / camera_tau) if camera_tau > 0 else 0.0
    # Each level consumes only the prefix populated at that level.
    candidates = np.empty((beam * 3, fields))
    scores = np.empty(beam * 3)
    candidate_parent = np.empty(beam * 3, np.int64)
    candidate_action = np.empty(beam * 3, np.int64)
    for depth in range(levels):
        total = 0
        for i in range(count):
            prev = states[depth, i]
            for choice in range(3):
                action = 0 if choice == 0 else -1 if choice == 1 else 1
                if depth == 0 and (
                    action == blocked_action
                    or (first_action != 2 and action != first_action)
                    or not ((1 << (action + 1)) & root_mask)
                ):
                    continue
                if single_input and depth > 0 and action:
                    continue
                if (
                    opening_goal is not None
                    and prev[1] + max(0.0, anchor - prev[9]) - prev[10] > opening_goal[0]
                ):
                    if (action < 0 and prev[0] < opening_goal[1]) or (
                        action > 0 and prev[0] > opening_goal[2]
                    ):
                        continue
                if (
                    action != 0
                    and depth > 0
                    and int(prev[7]) == 0
                    and depth * step + 1e-6 < control_interval
                ):
                    continue
                if action != 0 and (prev[4] + 1e-6 < cooldown or prev[8] > 0.5):
                    continue
                vx = action * speed if action else prev[2]
                vy = -jump if action else prev[3]
                launch_velocity = np.nan
                if reset_timing_uncertainty:
                    launch_velocity = prev[3] if action else prev[11]
                crossing = 1e20
                disc = vy * vy - 2 * gravity * (prev[1] - frontier)
                if vy < 0 and disc >= 0:
                    crossing = max(0.0, (-vy - math.sqrt(disc)) / gravity)
                if prev[8] > 0.5:
                    crossing = 0.0
                terminal = prev[8] > 0.5 or crossing <= step
                safe = True
                clearance_squared = 10000.0
                ex, ey = prev[0], prev[1]
                low_y = prev[9]
                scene = prev[10]
                previous_target = max(0.0, anchor - low_y)
                previous_local = 0.0
                # A continuous wait starts at the preceding checked endpoint.
                # Resets still recheck it with their new velocity envelope.
                first_sample = 1 if depth > 0 and action == 0 else 0
                for sample in range(first_sample, substeps + 1):
                    local = min(step * sample / substeps, crossing)
                    x = min(420.0, max(0.0, prev[0] + vx * local))
                    y = prev[1] + vy * local + 0.5 * gravity * local * local
                    ex, ey = x, y
                    low_y = min(low_y, y)
                    screen_y = y + max(0.0, anchor - low_y)
                    target = max(0.0, anchor - low_y)
                    scene = advance_scene(
                        scene,
                        previous_target,
                        target,
                        local - previous_local,
                        camera_tau,
                        sample_decay if abs(local - previous_local - sample_dt) < 1e-9 else -1.0,
                    )
                    previous_target, previous_local = target, local
                    relative_y = screen_y - scene
                    if screen_y >= height - hh - margin or relative_y <= hh + margin:
                        safe = False
                        break
                    world_t = latency + depth * step + local
                    jitter = timing_uncertainty * min(1.0, (latency + depth * step + local) / 0.15)
                    position_error = vertical_error_floor * min(1.0, world_t / 0.12)
                    for o in range(len(obstacles)):
                        b = obstacles[o]
                        bx, by = b[0] + b[4] * world_t, b[1] + b[5] * world_t
                        bw = b[2]
                        relative_speed = abs(vx - b[4])
                        if obstacle_bounds is not None and len(obstacle_bounds) > o:
                            lower, upper = obstacle_bounds[o]
                            if upper > lower:
                                bx, bw = reflected_extent(b[0], bw, b[4], world_t, lower, upper)
                                relative_speed = abs(vx) + abs(b[4])
                        mx = margin + relative_speed * (step / substeps / 2 + jitter)
                        my = (
                            margin
                            + vertical_uncertainty
                            + abs(vy + gravity * local - b[5]) * (step / substeps / 2 + jitter)
                        )
                        if reset_timing_uncertainty and not np.isnan(launch_velocity):
                            # For a reset at L observed at fixed time T:
                            # dY(T)/dL = velocity_before_reset - velocity_at_T.
                            # The launch-height error therefore survives at the
                            # apex, even though velocity_at_T is zero.
                            impulse_error = reset_position_error(
                                launch_velocity,
                                vy + gravity * local,
                                gravity,
                                reset_timing_uncertainty,
                            )
                            my = (
                                margin
                                + vertical_uncertainty
                                + impulse_error
                                + abs(vy + gravity * local - b[5]) * step / substeps / 2
                            )
                        my = max(my, margin + vertical_uncertainty + position_error)
                        if (
                            underside_margin
                            and 30 <= b[3] <= 42
                            and (b[0] < 4 or b[0] + b[2] > 416)
                            and relative_y >= by + b[3]
                            and vy + gravity * local < b[5]
                        ):
                            my += underside_margin
                        if (
                            x + hw > bx - mx
                            and x - hw < bx + bw + mx
                            and relative_y + hh > by - my
                            and relative_y - hh < by + b[3] + my
                        ):
                            safe = False
                            break
                        dx = max(bx - x - hw, x - hw - bx - bw, 0.0)
                        dy = max(by - relative_y - hh, relative_y - hh - by - b[3], 0.0)
                        clearance_squared = min(clearance_squared, dx * dx + dy * dy)
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
                                and relative_y + hh + vertical_uncertainty + position_error
                                > bounds[1] - dm
                                and relative_y - hh - vertical_uncertainty - position_error
                                < bounds[3] + dm
                            ):
                                continue
                            separate = False
                            for edge in range(edge_counts[h]):
                                nx, ny = shape_axes[tick, h, edge]
                                projection = x * nx + relative_y * ny
                                radius = (
                                    hw * abs(nx)
                                    + (hh + vertical_uncertainty + position_error) * abs(ny)
                                    + dm
                                )
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
                clearance = math.sqrt(clearance_squared)
                effective_y = ey + max(0.0, anchor - low_y) - scene
                wall_cost = wall_penalty * max(0.0, 60.0 - ex, ex - 360.0) * step
                # A horizontal turn also jumps. Spending time deep under a bar
                # leaves fewer phases in which a turn toward the opening is safe.
                if bar_count:
                    for k in range(bar_count):
                        left, right, bottom = edge_bars[k]
                        below = effective_y - bottom
                        sideways = min(ex - left, right - ex)
                        if 0 <= below < rise and sideways > 0:
                            if left < 0:
                                sideways = right - ex
                            elif right > 420:
                                sideways = ex - left
                            wall_cost += pocket_penalty * sideways * (1 - below / rise) * step
                cost = prev[5] + (click_cost if action else 0.0) + 1.2 / (clearance + 1) + wall_cost
                area = prev[6] + (initial[1] - effective_y) * step
                first = action if depth == 0 else int(prev[7])
                if route_values is None:
                    nav = lookup(grid, ex, effective_y, 10.0)
                else:
                    heading = 0 if vx < -1 else 2 if vx > 1 else 1
                    nav = lookup(route_values[heading], ex, effective_y, 10.0)
                score = (
                    -0.9 * nav
                    + 0.2 * (initial[1] - effective_y)
                    + 0.05 * area
                    - cost
                    + 0.12 * min(clearance, 40.0)
                )
                if opening_goal is not None:
                    if effective_y > opening_goal[0]:
                        distance = max(0.0, opening_goal[1] - ex, ex - opening_goal[2])
                        score -= 1.4 * distance
                    else:
                        score += 240.0
                candidate = candidates[total]
                candidate[0] = ex
                candidate[1] = ey
                candidate[2] = 0.0 if terminal else vx
                candidate[3] = 0.0 if terminal else vy + gravity * step
                candidate[4] = (0.0 if action else prev[4]) + step
                candidate[5] = cost
                candidate[6] = area
                candidate[7] = first
                candidate[8] = 1.0 if terminal else 0.0
                candidate[9] = low_y
                candidate[10] = scene
                if reset_timing_uncertainty:
                    candidates[total, 11] = launch_velocity
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
            xb = int(round(a[0] / x_bin))
            yb = int(round(a[1] / y_bin))
            vb = int(round(a[3] / velocity_bin)) + 64
            sign = 0 if a[2] < -1 else 2 if a[2] > 1 else 1
            age_bins = int(math.ceil(cooldown / step))
            age = min(age_bins, int(a[4] / step + 1e-6))
            key = (
                ((((xb * 256 + yb) * 3 + sign) * 256 + vb) * (age_bins + 1) + age) * 2 + int(a[8])
            ) * 3 + group
            key = key * 128 + int(round(a[9] / 8))
            key = key * 16 + min(15, int(round(max(0.0, anchor - a[9] - a[10]) / 2)))
            if reset_timing_uncertainty:
                launch_bin = 0 if np.isnan(a[11]) else 1 + min(63, max(0, int(a[11] / 100) + 32))
                key = key * 65 + launch_bin
            if key in seen:
                continue
            seen.add(key)
            states[depth + 1, count] = a
            parents[depth + 1, count] = candidate_parent[j]
            actions[depth + 1, count] = candidate_action[j]
            if root_quality is not None and group_counts[group] == 0:
                root_quality[group, 0] = depth + 1
                root_quality[group, 1] = scores[j]
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
        path[d - 1, 0] = states[d, index, 0]
        path[d - 1, 1] = (
            states[d, index, 1] + max(0.0, anchor - states[d, index, 9]) - states[d, index, 10]
        )
        first_action = actions[d, index]
        if planned_actions is not None:
            planned_actions[d - 1] = first_action
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
