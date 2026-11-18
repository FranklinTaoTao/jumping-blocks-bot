import numpy as np

from .native_search import search, shape_arrays
from .navigation import Navigation, bar_gaps, opening_guidance, square_gaps
from .physics import State
from .planner import Decision


def _shape_samples(shapes, latency, step, levels, substeps):
    if not shapes:
        return shape_arrays(shapes, ())
    starts = latency + np.arange(levels) * step
    offsets = np.linspace(0, step, substeps + 1)
    return shape_arrays(shapes, (starts[:, None] + offsets).ravel())


class FastPlanner:
    def __init__(
        self,
        physics,
        horizon=3.0,
        step=0.05,
        beam=900,
        margin=4.0,
        cooldown=0.12,
        click_cost=12.0,
        camera_tau=0.0,
        vertical_uncertainty=0.0,
        jump_guidance=False,
        timing_uncertainty=0.025,
        emergency=False,
        vertical_error_floor=0.0,
        reset_timing_uncertainty=0.0,
        pocket_penalty=0.0,
        narrow_gap_cooldown=None,
        wall_approach=False,
        coast_recovery=False,
        timed_gap_turns=False,
        route_rules=False,
    ):
        self.p = physics
        self.horizon, self.step, self.beam = horizon, step, beam
        self.margin, self.cooldown, self.click_cost = margin, cooldown, click_cost
        self.camera_tau = camera_tau
        self.vertical_uncertainty = vertical_uncertainty
        self.jump_guidance = jump_guidance
        self.timing_uncertainty = timing_uncertainty
        self.emergency = emergency
        self.vertical_error_floor = vertical_error_floor
        self.reset_timing_uncertainty = reset_timing_uncertainty
        self.pocket_penalty = pocket_penalty
        self.narrow_gap_cooldown = narrow_gap_cooldown
        self.coast_recovery = coast_recovery
        self.route_rules = route_rules
        self.underside_margin = 2.0 if route_rules else 0.0
        self.turn_probe = None
        if timed_gap_turns:
            self.turn_probe = FastPlanner(
                physics,
                horizon=max(0.75, step),
                step=step,
                beam=min(300, beam),
                margin=margin,
                cooldown=cooldown,
                click_cost=click_cost,
                camera_tau=camera_tau,
                vertical_uncertainty=vertical_uncertainty,
                jump_guidance=jump_guidance,
                timing_uncertainty=timing_uncertainty,
                vertical_error_floor=vertical_error_floor,
                reset_timing_uncertainty=reset_timing_uncertainty,
                pocket_penalty=pocket_penalty,
            )
            self.turn_probe.underside_margin = self.underside_margin
        self.approach_planner = None
        if wall_approach:
            self.approach_planner = FastPlanner(
                physics,
                horizon=horizon,
                step=min(step, 0.02),
                beam=max(beam, 900),
                margin=margin,
                cooldown=cooldown,
                click_cost=click_cost,
                camera_tau=camera_tau,
                vertical_uncertainty=vertical_uncertainty,
                jump_guidance=jump_guidance,
                timing_uncertainty=timing_uncertainty,
                vertical_error_floor=vertical_error_floor,
                reset_timing_uncertainty=reset_timing_uncertainty,
                pocket_penalty=pocket_penalty,
                route_rules=route_rules,
            )

    def gap_cooldown(self, state):
        if self.narrow_gap_cooldown is None:
            return self.cooldown
        for top, bottom, left, right in square_gaps(state.obstacles):
            if not left - 8 <= state.x <= right + 8:
                continue
            if top - 20 <= state.y <= bottom + 160:
                return min(self.cooldown, self.narrow_gap_cooldown)
            if 20 < top - state.y <= 80:
                # A short reset sequence can continue above the squares when
                # coasting will carry the player into the next visible bar.
                half = state.w / 2 + self.margin
                projected_x = state.x + state.vx * self.cooldown
                for _, bar_bottom, bar_left, bar_right in bar_gaps(state.obstacles):
                    if not (0 < state.y - bar_bottom <= 175 and bar_left <= state.x <= bar_right):
                        continue
                    if (state.vx > 0 and projected_x > bar_right - half) or (
                        state.vx < 0 and projected_x < bar_left + half
                    ):
                        return min(self.cooldown, self.narrow_gap_cooldown)
        return self.cooldown

    def upper_gap_reversal(self, state, latency, beside_square_gap):
        """Direction toward the next opening if coasting would leave it."""
        half_width = state.w / 2 + self.margin
        # Check that coasting keeps the player inside the opening over the
        # interval normally needed between successive turns.
        projected_x = state.x + state.vx * (latency + self.cooldown)
        squares = square_gaps(state.obstacles)
        openings = sorted(bar_gaps(state.obstacles) + squares, reverse=True)
        for opening in openings:
            _, bottom, left, right = opening
            if not 0 < state.y - bottom <= 235 or right - left <= 2 * half_width:
                continue
            approaching_squares = opening in squares and left - 8 <= state.x <= right + 8
            if not (beside_square_gap or approaching_squares):
                continue
            if state.vx > 0 and projected_x > right - half_width:
                return -1
            if state.vx < 0 and projected_x < left + half_width:
                return 1
            break
        return 0

    def approach_opening(self, state, decision, rise, board_height, latency, control_interval):
        """Check for a route through the upper opening before jumping wallward."""
        if (
            120 < state.x < 300
            or decision.predicted_horizon < 0.5
            or decision.action * state.vx <= 0
            or not state.w / 2 < state.x < 420 - state.w / 2
        ):
            return decision
        openings = bar_gaps(state.obstacles)
        opening = next((g for g in openings if 0 < state.y - g[1] <= 2 * rise), None)
        if opening is None:
            return decision
        top, bottom, left, right = opening
        outward = (
            1
            if state.x >= 300 and state.x > right
            else (-1 if state.x <= 120 and state.x < left else 0)
        )
        clear_y = top - state.h / 2 - self.margin
        if (
            decision.action != outward
            or clear_y <= state.h / 2 + 8
            or np.min(decision.path[:, 1]) <= clear_y
        ):
            return decision
        approach = self.approach_planner.plan(
            state, board_height, latency, control_interval, _blocked_action=outward
        )
        if not approach.planned_inputs or approach.planned_inputs[0][1] != -outward:
            return decision
        reaches_opening = np.min(approach.path[:, 1]) <= clear_y
        if not reaches_opening and approach.predicted_horizon >= decision.predicted_horizon:
            # A useful staging position fits horizontally and leaves room to
            # start the next ascent. Prefer getting there earlier.
            hw, hh = state.w / 2 + self.margin, state.h / 2 + self.margin

            def staging_time(path, step):
                x, y = path.T
                inside = np.flatnonzero(
                    (left + hw <= x)
                    & (x <= right - hw)
                    & (bottom + hh <= y)
                    & (y <= bottom + hh + rise)
                )
                return step * (inside[0] + 1) if len(inside) else np.inf

            reaches_opening = staging_time(approach.path, self.approach_planner.step) + 0.2 < (
                staging_time(decision.path, self.step)
            )
        if reaches_opening:
            approach.reason = "approaching upper opening: " + approach.reason
            return approach
        return decision

    def coast_safe(self, state, board_height, duration):
        """Validate the whole wait before a scheduled reset, including hazards."""
        return (
            self._coast_path(state, board_height, duration, self.margin, self.timing_uncertainty)
            is not None
        )

    def time_gap_turn(
        self, state, decision, board_height, latency, control_interval, *, cooldown=None
    ):
        action = decision.action
        if (
            not action
            or action * state.vx >= 0
            or decision.predicted_horizon < 1.5
            or state.shapes
            or not decision.reason.startswith(("predicted", "upper gap", "narrow gap"))
        ):
            return decision
        if not any(
            left - 8 <= state.x <= right + 8 and 8 < state.y - bottom < 130
            for _, bottom, left, right in square_gaps(state.obstacles)
        ):
            return decision
        # Move a turn away from the early collision boundary if its later
        # window still leaves room to coast and complete the same route.
        early = self.turn_probe.plan(
            state,
            board_height,
            max(0.002, latency - 0.024),
            control_interval,
            first_action=action,
            _cooldown=cooldown,
        )
        if early.predicted_horizon >= 0.70:
            return decision
        late = self.turn_probe.plan(
            state,
            board_height,
            latency + 0.040,
            control_interval,
            first_action=action,
            _cooldown=cooldown,
        )
        delayed = latency + 0.024
        if late.predicted_horizon < 0.70 or not self.coast_safe(
            state, board_height, latency + 0.040
        ):
            return decision
        turn = self.plan(
            state, board_height, delayed, control_interval, first_action=action, _cooldown=cooldown
        )
        if turn.action != action or turn.predicted_horizon < decision.predicted_horizon:
            return decision
        turn.reason = "timing the turn through the square gap"
        return turn

    def wait_for_hazard_window(self, state, decision, board_height, latency, interval):
        if (
            not decision.action
            or state.shapes
            or decision.predicted_horizon < 1.5
            or not decision.reason.startswith("predicted")
        ):
            return decision
        followup_x = state.x + state.vx * (latency + self.cooldown)
        first_two = decision.planned_inputs[:2]
        if (
            len(first_two) == 2
            and decision.action * state.vx > 0
            and first_two[1][1] == -decision.action
            and first_two[1][0] <= latency + self.cooldown + 0.060
        ):
            # A quick reversal may have to take effect before the preceding
            # jump carries the player out of the opening.
            followup_x = state.x + state.vx * (first_two[1][0] + 0.025)
        half = state.w / 2 + self.margin
        leaving_wall_band = state.vy > 250 and (
            (state.x <= 35 and decision.action == 1) or (state.x >= 385 and decision.action == -1)
        )
        near_bar = any(
            20 < state.y - bottom <= 300
            and (
                min(abs(state.x - left), abs(state.x - right)) < 80
                or (leaving_wall_band and not left <= state.x <= right)
            )
            and (decision.action * state.vx < 0 or not left + half < followup_x < right - half)
            for _, bottom, left, right in bar_gaps(state.obstacles, include_moving=True)
        )
        near_slider = any(
            60 <= w <= 80
            and 60 <= h <= 80
            and abs(vx) > 50
            and 0 < state.y - (y + h) <= 180
            and x - 85 <= state.x <= x + w + 85
            for x, y, w, h, vx, _ in state.obstacles
        )
        medium_blocks = [
            b
            for b in state.obstacles
            if 40 <= b[2] <= 52
            and 40 <= b[3] <= 52
            and b[0] >= 4
            and b[0] + b[2] <= 416
            and abs(b[4]) < 15
            and abs(b[5]) < 15
        ]
        near_square_row = any(
            20 < state.y - (y + h) <= 230
            and x - 45 < state.x < x + w + 45
            and sum(abs(b[1] - y) <= 4 for b in medium_blocks) >= 3
            for x, y, w, h, _, _ in medium_blocks
        )
        if not (near_bar or near_slider or near_square_row):
            return decision
        hazard = "bar" if near_bar else "sliding block" if near_slider else "row of squares"
        early, late = max(0.002, latency - 0.020), latency + 0.020
        action = decision.action
        early_jump = self.turn_probe.plan(state, board_height, early, interval, first_action=action)
        late_jump = self.turn_probe.plan(state, board_height, late, interval, first_action=action)
        if min(early_jump.predicted_horizon, late_jump.predicted_horizon) >= 0.55:
            if (
                near_bar
                and state.vy < 0
                and action * state.vx < 0
                and decision.root_mask & 2
                and len(first_two) == 2
                and first_two[0][1] == first_two[1][1] == action
            ):
                spacing = first_two[1][0] - first_two[0][0]
                if spacing < self.cooldown + 0.060:
                    wait = self.plan(state, board_height, latency, interval, first_action=0)
                    if (
                        wait.predicted_horizon >= decision.predicted_horizon
                        and len(wait.planned_inputs) >= 2
                        and wait.planned_inputs[0][1] == action
                        and wait.planned_inputs[1][0] - wait.planned_inputs[0][0] >= spacing + 0.060
                        and self.coast_safe(state, board_height, wait.planned_inputs[0][0] + 0.020)
                    ):
                        wait.reason = "coasting to give the next jump more time"
                        return wait
            return decision
        if (decision.root_mask & 2) and all(
            self.turn_probe.plan(
                state, board_height, timing, interval, first_action=0
            ).predicted_horizon
            >= 0.70
            for timing in (early, late)
        ):
            wait = self.plan(state, board_height, latency, interval, first_action=0)
            if wait.predicted_horizon >= 1.0 and self.coast_safe(
                state, board_height, latency + interval
            ):
                wait.reason = f"waiting for a wider timing window under the {hazard}"
                return wait
        if early_jump.predicted_horizon < 0.55 <= late_jump.predicted_horizon:
            # Waiting for a new observation can be too late while falling.
            # Preserve the jump but move it into the later safe window.
            for offset, reserve in ((0.024, 0.016), (0.012, 0.012)):
                delayed = latency + offset
                latest = delayed + reserve
                if not self.coast_safe(state, board_height, latest):
                    continue
                if any(
                    self.turn_probe.plan(
                        state, board_height, timing, interval, first_action=action
                    ).predicted_horizon
                    < 0.55
                    for timing in (delayed - reserve, latest)
                ):
                    continue
                route = self.plan(state, board_height, delayed, interval, first_action=action)
                if route.action == action and route.predicted_horizon >= decision.predicted_horizon:
                    route.reason = f"timing the jump beneath the {hazard}"
                    return route
        return decision

    def time_wall_departure(self, state, decision, board_height, latency, interval):
        inward = 1 if state.x < 5 else -1 if state.x > 415 else 0
        if (
            not inward
            or decision.action != inward
            or state.vy < 500
            or state.shapes
            or decision.predicted_horizon < 1.5
        ):
            return decision
        openings = bar_gaps(state.obstacles, include_moving=True)
        if not any(0 < state.y - bottom < 340 for _, bottom, _, _ in openings):
            return decision
        early = max(0.0, latency - 0.024)
        if (
            self.turn_probe.plan(
                state, board_height, early, interval, first_action=inward
            ).predicted_horizon
            >= 0.70
        ):
            return decision
        delayed = latency + 0.028
        late = delayed + 0.014
        if not self.coast_safe(state, board_height, late):
            return decision
        # At high falling speed even a few early milliseconds take away the
        # height needed to travel sideways before the next rise under the bar.
        for timing in (delayed - 0.024, late):
            check = self.turn_probe.plan(state, board_height, timing, interval, first_action=inward)
            if check.predicted_horizon < 0.70:
                return decision
        route = self.plan(state, board_height, delayed, interval, first_action=inward)
        if route.action == inward and route.predicted_horizon >= decision.predicted_horizon:
            route.reason = "timing the departure from the wall"
            return route
        return decision

    def time_pocket_jump(self, state, decision, board_height, latency, interval, *, opening=None):
        """Avoid a narrow reset between an overhead bar and a block below."""
        if not decision.action or not decision.planned_inputs or state.vy < 80:
            return decision
        when, action = decision.planned_inputs[0]
        if action != decision.action:
            return decision
        vy = state.vy
        gravity = self.p.gravity * state.motion_rate**2 * state.gravity_gain
        rise = (self.p.jump * state.jump_gain) ** 2 / (2 * self.p.gravity * state.gravity_gain)
        reach = 0.48 * self.p.speed * state.motion_rate + state.w / 2
        latest = when + 0.090
        below = state.y + vy * latest + 0.5 * gravity * latest**2
        if not any(
            26 <= h <= 44
            and (x < 4 or x + w > 416)
            and state.y - rise - state.h <= y + h < state.y
            and x - reach <= state.x <= x + w + reach
            for x, y, w, h, _, _ in state.obstacles
        ) or not any(
            x >= 4
            and x + w <= 416
            and min(w, h) >= 25
            and state.y - state.h / 2 <= y <= below + state.h / 2 + self.margin
            and x - reach <= state.x <= x + w + reach
            for x, y, w, h, _, _ in state.obstacles
        ):
            return decision

        # Hold the same physical input across probes. Independent beam searches
        # can hide a narrow window by finding different follow-up jumps.
        hold = (
            min(0.75, decision.planned_inputs[1][0] - when)
            if len(decision.planned_inputs) > 1
            else 0.75
        )
        cache = {}

        def safe(t, direction):
            t = max(0.0, round(t, 6))
            if t not in cache:
                if self.coast_safe(state, board_height, t):
                    cache[t] = self._immediate_probe(
                        state, board_height, t, hold, margin=self.margin
                    )[0]
                else:
                    cache[t] = np.zeros(3)
            return cache[t][direction + 1] + 1e-7 >= hold

        reserve = 0.020
        if all(safe(when + offset, action) for offset in (-reserve, 0, reserve)):
            return decision

        step = 0.005
        # A future reset may be placed after the minimum delivery latency.
        # Earlier samples are only used to measure room for delivery jitter.
        phases = np.arange(max(0.0, latency - 0.060), max(when, latency) + 0.065, step)
        candidates = []
        current_slack = 0.0
        for direction in (action, -action):
            valid = np.array([safe(t, direction) for t in phases])
            edges = np.flatnonzero(np.diff(np.r_[False, valid, False]))
            for start, end in zip(edges[::2], edges[1::2]):
                earliest, latest = phases[start], phases[end - 1]
                center = max(latency, (earliest + latest) / 2)
                if center > latest:
                    continue
                slack = min(center - earliest, latest - center)
                if direction == action and earliest <= when <= latest:
                    current_slack = min(when - earliest, latest - when)
                if slack >= step / 2 - 1e-7:
                    candidates.append((slack, direction, center, earliest, latest))

        candidates = [item for item in candidates if item[0] >= current_slack + step / 2]
        minimum_horizon = max(1.0, decision.predicted_horizon - 0.25)
        if not candidates or max(item[0] for item in candidates) < reserve:
            wait = self.plan(
                state, board_height, latency, interval, first_action=0, _opening=opening
            )
            if (
                not wait.action
                and wait.planned_inputs
                and wait.predicted_horizon >= minimum_horizon
            ):
                next_time, next_direction = wait.planned_inputs[0]
                if next_time >= latency + interval and all(
                    safe(next_time + offset, next_direction)
                    for offset in np.arange(-reserve, reserve + step / 2, step)
                ):
                    wait.reason = "coasting to a wider jump window"
                    wait.reset_window = (next_time - reserve, next_time + reserve)
                    return wait

        for slack, direction, center, earliest, latest in sorted(
            candidates, key=lambda item: -item[0]
        ):
            if not all(safe(center + offset, direction) for offset in (-slack, 0, slack)):
                continue
            route = self.plan(
                state, board_height, center, interval, first_action=direction, _opening=opening
            )
            if route.action == direction and route.predicted_horizon >= minimum_horizon:
                route.reason = (
                    "taking the wider jump window"
                    if direction != action
                    else "centering the jump between bar and block"
                )
                route.reset_window = (float(earliest), float(latest))
                return route
        return decision

    def _immediate_probe(
        self, state, board_height, latency, duration, first_action=2, root_mask=7, *, margin=None
    ):
        levels = max(1, int(np.ceil(duration / 0.0125)))
        step = duration / levels
        bounds, axes, limits, counts = _shape_samples(state.shapes, latency, step, levels, 2)
        initial = np.array(
            [state.x, state.y, state.vx, state.vy, state.w, state.h, state.age_since_click],
            dtype=np.float64,
        )
        quality = np.empty((3, 2))
        action, path, depth, _ = search(
            initial,
            state.obstacles,
            np.zeros((2, 2)),
            bounds,
            axes,
            limits,
            counts,
            self.p.speed * state.motion_rate,
            self.p.jump * state.motion_rate * state.jump_gain,
            self.p.gravity * state.motion_rate**2 * state.gravity_gain,
            latency,
            step,
            levels,
            3,
            max(2.0, self.margin * 0.5) if margin is None else margin,
            0.0,
            0.0,
            board_height,
            2,
            0.0,
            float(self.camera_tau),
            float(state.camera_speed),
            0.0,
            state.obstacle_bounds,
            0.0,
            0.0,
            first_action=first_action,
            root_quality=quality,
            root_mask=root_mask,
            single_input=True,
            underside_margin=self.underside_margin,
        )
        return quality[:, 0] * step, action, path, depth * step

    def immediate_mask(self, state, board_height, latency, control_interval, cooldown):
        """Admit safe root actions through the next usable decision time."""
        if state.age_since_click + latency < cooldown:
            return 2  # A jump is currently ineligible under this cooldown.
        jitter = 0.016
        next_opportunity = max(latency + control_interval, cooldown - state.age_since_click)
        waiting = (
            self._immediate_probe(
                state, board_height, 0.0, next_opportunity + jitter, first_action=0
            )[0][1]
            + 1e-7
        )
        wait_score = 2 if waiting >= next_opportunity else 0
        robust = 2 if waiting >= next_opportunity + jitter else 0
        nominal = wait_score
        jump_hold = max(cooldown, control_interval)
        for index, phase in enumerate((latency, max(0.0, latency - jitter), latency + jitter)):
            if waiting >= phase:
                survives, _, _, _ = self._immediate_probe(state, board_height, phase, jump_hold)
                safe = (1 if survives[0] + 1e-7 >= jump_hold else 0) | (
                    4 if survives[2] + 1e-7 >= jump_hold else 0
                )
            else:
                safe = 0
            if index == 0:
                robust |= safe
                nominal |= safe
            else:
                robust &= safe | 2
        # When timing samples leave no robust root, use a nominally safe
        # root if there is one. If none survives, preserve the planner's
        # ability to find the longest remaining trajectory.
        return robust or nominal or 7

    def _coast_path(self, state, board_height, duration, margin, timing_uncertainty):
        if duration <= 1e-6:
            return np.empty((0, 2))
        levels = max(1, int(np.ceil(duration / 0.01)))
        step = duration / levels
        bounds, axes, limits, counts = _shape_samples(state.shapes, 0.0, step, levels, 2)
        initial = np.array(
            [state.x, state.y, state.vx, state.vy, state.w, state.h, state.age_since_click],
            dtype=np.float64,
        )
        _, path, depth, _ = search(
            initial,
            state.obstacles,
            np.zeros((2, 2)),
            bounds,
            axes,
            limits,
            counts,
            self.p.speed * state.motion_rate,
            self.p.jump * state.motion_rate * state.jump_gain,
            self.p.gravity * state.motion_rate**2 * state.gravity_gain,
            0.0,
            step,
            levels,
            3,
            margin,
            1e6,
            0.0,
            board_height,
            2,
            0.04,
            float(self.camera_tau),
            float(state.camera_speed),
            float(self.vertical_uncertainty),
            state.obstacle_bounds,
            0.0,
            float(timing_uncertainty),
            np.zeros(levels, np.int64),
            float(self.vertical_error_floor),
            float(self.reset_timing_uncertainty),
            0,
            underside_margin=self.underside_margin,
        )
        if depth == levels and len(path) and path[-1, 1] > state.h / 2 + 9:
            return path
        return None

    def coast_for_clipped_hammer(self, state, board_height, decision, latency, interval):
        if (
            not decision.action
            or state.vy > 180
            or state.y > board_height * 0.62
            or not (decision.root_mask & 2)
        ):
            return decision
        nearby_x = state.x + decision.action * self.p.speed * 0.20
        for shape in state.shapes:
            if shape.uncertainty < 20 or shape.angle_limits is not None:
                continue
            left, top = shape.points.min(axis=0)
            right, bottom = shape.points.max(axis=0)
            if (
                top > 35
                or right - left < 65
                or bottom - top < 65
                or not 0 < state.y - bottom < 285
                or not left - 52 <= nearby_x <= right + 52
            ):
                continue
            # A full pin can appear within a few frames of the hammer clearing
            # the top and score. Avoid a reset into the untracked swing.
            duration = max(0.13, latency + interval + 0.03)
            path = self._coast_path(state, board_height, duration, 3.0, self.timing_uncertainty)
            if path is not None:
                return Decision(
                    0,
                    path,
                    "coasting for incoming hammer tracking",
                    decision.score,
                    duration,
                    root_mask=decision.root_mask,
                )
        return decision

    def plan(
        self,
        state,
        board_height,
        latency=0.02,
        control_interval=0.05,
        first_action=None,
        root_quality=None,
        *,
        _cooldown=None,
        _blocked_action=2,
        _opening=None,
    ):
        if root_quality is not None:
            root_quality[:, 0] = 0
            root_quality[:, 1] = -1e30
        if not state.ready:
            return Decision(0, np.empty((0, 2)), "waiting for reliable tracking")
        rise = (self.p.jump * state.jump_gain) ** 2 / (2 * self.p.gravity * state.gravity_gain)
        substeps = max(2, int(np.ceil(self.step / 0.01)))
        nav = Navigation(state, self.margin + 1, jump_clearance=rise if self.jump_guidance else 0)
        # Guidance is a smooth potential field; exact collision geometry is
        # checked independently in native search at every intermediate sample.
        yy, xx = np.mgrid[0 : board_height + 11 : 10, 0:431:10]
        values = nav.cost(xx.reshape(-1), yy.reshape(-1)).reshape(xx.shape)
        cooldown = self.cooldown if _cooldown is None else _cooldown
        root_mask = 7
        route_values = None
        if self.route_rules:
            root_mask = self.immediate_mask(
                state, board_height, latency, control_interval, cooldown
            )
            route_values = values + opening_guidance(
                state,
                xx,
                yy,
                self.margin,
                self.p.speed * state.motion_rate,
                self.p.jump * state.motion_rate * state.jump_gain,
                cooldown,
            )
        levels = round(self.horizon / self.step)
        planned_actions = np.zeros(levels, dtype=np.int64)
        bounds, axes, limits, counts = _shape_samples(
            state.shapes, latency, self.step, levels, substeps
        )
        initial = np.array(
            [state.x, state.y, state.vx, state.vy, state.w, state.h, state.age_since_click],
            dtype=np.float64,
        )
        action, path, depth, score = search(
            initial,
            state.obstacles,
            values,
            bounds,
            axes,
            limits,
            counts,
            self.p.speed * state.motion_rate,
            self.p.jump * state.motion_rate * state.jump_gain,
            self.p.gravity * state.motion_rate**2 * state.gravity_gain,
            float(latency),
            self.step,
            levels,
            self.beam,
            self.margin,
            cooldown,
            self.click_cost,
            board_height,
            substeps,
            float(control_interval),
            float(self.camera_tau),
            float(state.camera_speed),
            float(self.vertical_uncertainty),
            state.obstacle_bounds,
            0.05 * (1 + min(9.0, state.wall_dwell / 2)),
            float(self.timing_uncertainty),
            planned_actions,
            float(self.vertical_error_floor),
            float(self.reset_timing_uncertainty),
            2 if first_action is None else int(first_action),
            root_quality,
            self.pocket_penalty,
            _blocked_action,
            root_mask=root_mask,
            route_values=route_values,
            underside_margin=self.underside_margin,
            opening_goal=_opening,
        )
        reason = (
            f"predicted {depth * self.step:.2f}s (native)"
            if depth
            else "no collision-free immediate action"
        )
        decision = Decision(action, path, reason, score, depth * self.step)
        if self.route_rules:
            decision.root_mask = root_mask
        decision.planned_inputs = [
            (float(latency + i * self.step), int(planned_actions[i]))
            for i in range(depth)
            if planned_actions[i]
        ]
        if depth == 0 and first_action is None and not (root_mask & 2):
            # The longer search uses larger margins. If it rejects every
            # route, use the jump that still passed the immediate safety
            # probe when a wait would collide before the next decision.
            duration = max(cooldown, control_interval)
            mask = root_mask if _blocked_action == 2 else root_mask & ~(1 << (_blocked_action + 1))
            _, rescue, route, survives = self._immediate_probe(
                state,
                board_height,
                latency,
                duration,
                root_mask=mask,
            )
            if rescue and survives + 1e-7 >= duration:
                return Decision(
                    rescue,
                    route,
                    "immediate safety jump",
                    predicted_horizon=survives,
                    planned_inputs=[(latency, rescue)],
                    root_mask=root_mask,
                )
        if (
            _cooldown is None
            and first_action is None
            and root_quality is None
            and self.narrow_gap_cooldown is not None
            and decision.predicted_horizon < 1.0
        ):
            gap_cooldown = self.gap_cooldown(state)
            beside_square_gap = gap_cooldown < self.cooldown
            if beside_square_gap and decision.predicted_horizon <= 0.10:
                rescue = self.plan(
                    state, board_height, latency, control_interval, _cooldown=gap_cooldown
                )
                if rescue.action and rescue.predicted_horizon >= 0.5:
                    rescue.reason = "narrow gap: " + rescue.reason
                    if self.route_rules and self.turn_probe is not None:
                        rescue = self.time_gap_turn(
                            state,
                            rescue,
                            board_height,
                            latency,
                            control_interval,
                            cooldown=gap_cooldown,
                        )
                    return rescue
            reversal_cooldown = min(self.cooldown, self.narrow_gap_cooldown)
            if not decision.action and state.age_since_click + latency >= reversal_cooldown:
                direction = self.upper_gap_reversal(state, latency, beside_square_gap)
                if direction:
                    rescue = self.plan(
                        state,
                        board_height,
                        latency,
                        control_interval,
                        first_action=direction,
                        _cooldown=reversal_cooldown,
                    )
                    if rescue.action == direction and rescue.predicted_horizon >= 1.5:
                        rescue.reason = "upper gap reversal: " + rescue.reason
                        if self.route_rules and self.turn_probe is not None:
                            rescue = self.time_gap_turn(
                                state,
                                rescue,
                                board_height,
                                latency,
                                control_interval,
                                cooldown=reversal_cooldown,
                            )
                        return rescue
        if self.emergency and first_action is None and decision.predicted_horizon < 0.15:
            fallback = FastPlanner(
                self.p,
                horizon=0.6,
                step=self.step,
                beam=min(self.beam, 300),
                margin=1.0,
                cooldown=min(self.cooldown, 0.12),
                click_cost=self.click_cost,
                camera_tau=self.camera_tau,
                timing_uncertainty=0.008,
                emergency=False,
                vertical_error_floor=self.vertical_error_floor,
                reset_timing_uncertainty=self.reset_timing_uncertainty,
                route_rules=self.route_rules,
            )
            rescue = fallback.plan(state, board_height, latency, control_interval)
            if rescue.action and rescue.predicted_horizon >= decision.predicted_horizon + 0.2:
                rescue.reason = "emergency nominal: " + rescue.reason
                return rescue
        if (
            self.coast_recovery
            and first_action is None
            and root_quality is None
            and _cooldown is None
            and _blocked_action == 2
            and decision.action
            and root_mask & 2
            and (
                decision.predicted_horizon <= 0.20
                or (
                    decision.predicted_horizon <= 0.40
                    and decision.action * state.vx < 0
                    and any(
                        0 < state.y - bottom <= 250
                        and (state.x <= left if decision.action < 0 else state.x >= right)
                        for _, bottom, left, right in bar_gaps(state.obstacles)
                    )
                )
            )
            and state.vy < -120
            and not state.shapes
        ):
            # With no new reset, the current arc carries less timing error than
            # a launch. Check whether continuing it can clear the obstacle.
            coast_time = 0.32
            path = self._coast_path(
                state, board_height, coast_time, margin=2.0, timing_uncertainty=0.0
            )
            if path is not None:
                return Decision(
                    0,
                    path,
                    "coasting past immediate obstacle",
                    decision.score,
                    coast_time,
                    root_mask=decision.root_mask,
                )
        if (
            self.approach_planner is not None
            and first_action is None
            and root_quality is None
            and _blocked_action == 2
        ):
            decision = self.approach_opening(
                state, decision, rise, board_height, latency, control_interval
            )
        if (
            self.turn_probe is not None
            and first_action is None
            and root_quality is None
            and _cooldown is None
            and _blocked_action == 2
        ):
            decision = self.time_gap_turn(state, decision, board_height, latency, control_interval)
            if self.route_rules:
                decision = self.time_wall_departure(
                    state, decision, board_height, latency, control_interval
                )
                centered = self.time_pocket_jump(
                    state, decision, board_height, latency, control_interval
                )
                decision = (
                    self.wait_for_hazard_window(
                        state, decision, board_height, latency, control_interval
                    )
                    if centered is decision
                    else centered
                )
        if self.route_rules and state.shapes and first_action is None and root_quality is None:
            decision = self.coast_for_clipped_hammer(
                state, board_height, decision, latency, control_interval
            )
        return decision

    def warmup(self):
        self.plan(State(210, 500, 0, 0, 30, 32, ready=True), 745, latency=0.0)
        if self.approach_planner is not None:
            self.approach_planner.warmup()
        if self.turn_probe is not None:
            self.turn_probe.warmup()
