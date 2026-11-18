"""Keep the next planned reset deadline while validating it on fresh frames."""

from collections import deque
from copy import copy
from dataclasses import dataclass, replace
from math import sqrt
from statistics import median, pstdev

import numpy as np

from .navigation import bar_gaps
from .physics import State
from .planner import Decision

MAX_INPUT_FRAME_AGE = 0.090
INPUT_DEADLINE_RESERVE = 0.002


@dataclass(frozen=True)
class ScheduledInput:
    effect_at: float
    direction: int


@dataclass
class Command:
    decision: Decision
    effect_at: float | None = None
    committed: bool = False


class WallStall:
    """Let a stalled round end naturally and resume on the next restart."""

    def __init__(self, timeout=10.0):
        self.timeout = timeout
        self.reset()

    def update(self, state, captured, confirmed_score):
        if self.ending or state is None:
            return self.ending

        if confirmed_score != self.score:
            self.score = confirmed_score
            self.wall_time = 0.0
            self.last_contact = None
        if self.last_contact is not None and captured - self.last_contact >= 3.0:
            self.wall_time = 0.0
        if state.wall_dwell > 0:
            # A short detour can return to the same wall without gaining height.
            # Count time observed at the wall, excluding the detour itself.
            if confirmed_score is not None and self.last_capture is not None:
                interval = captured - self.last_capture
                if 0 < interval < 0.15:
                    self.wall_time += min(interval, state.wall_dwell)
            self.last_contact = captured
        self.last_capture = captured
        self.ending = max(state.wall_dwell, self.wall_time) >= self.timeout
        return self.ending

    def reset(self):
        self.ending = False
        self.score = None
        self.last_contact = None
        self.last_capture = None
        self.wall_time = 0.0


class OpeningApproach:
    """Keep a checked route through an opening when direct ascent leads to a pocket."""

    def __init__(self, planner):
        self.planner = planner
        self.fine = planner.approach_planner
        # A narrow window can fall between the fine planner's time steps.
        self.refined = copy(self.fine)
        self.refined.step = min(self.fine.step, 0.0125)
        self.refined.beam = min(self.fine.beam, 450)
        self.pocket_search = copy(self.refined)
        self.pocket_search.step = min(self.refined.step, 0.010)
        self.pocket_search.beam = max(self.refined.beam, 540)
        self.pocket_search.horizon = min(self.refined.horizon, 2.10)
        self.pocket_search.timing_uncertainty = 0.006
        self.reset()

    def warmup(self):
        state = State(50, 500, 0, 200, 30, 30, ready=True)
        self.pocket_search.plan(state, 730, _opening=(280.0, 200.0, 350.0))

    def reset(self):
        self.pending = deque()
        self.proposed = False
        self.last_search = -float("inf")
        self.pocket_edges = None
        self.bar_deadline = None
        self.bar_return = None
        self.short_bar_return = None
        self.wall_departure = None
        self.gap_followup = None
        self.next_gap_followup = None

    def confirm_sent(self):
        if self.proposed and self.pending:
            self.pending.popleft()
            if self.next_gap_followup is not None:
                self.gap_followup = self.next_gap_followup
                self.pending.append(self.gap_followup)
        self.next_gap_followup = None
        self.proposed = False

    def _remember(self, route, captured, limit=2):
        for when, direction in route.planned_inputs[:limit]:
            self.pending.append(ScheduledInput(captured + when, direction))
        self.proposed = bool(route.action)

    def _crosses_opening(self, route, state, opening):
        top, _, left, right = opening
        if not len(route.path):
            return False
        x, y = route.path.T
        half = state.w / 2 + self.planner.margin
        clear_y = max(state.h / 2 + 15, top - state.h / 2 - self.planner.margin)
        return np.any((y <= clear_y) & (left + half <= x) & (x <= right - half))

    def _leave_wall(self, normal, state, height, captured, latency, interval, opening):
        _, _, left, right = opening
        inward = 1 if state.x <= 1 and left >= 45 else -1 if state.x >= 419 and right <= 375 else 0
        if not inward or state.vx != 0 or state.wall_dwell < 0.4:
            return None

        route = normal
        if (
            not route.planned_inputs
            or route.planned_inputs[0][1] != inward
            or not self._crosses_opening(route, state, opening)
        ):
            if normal.action and not self.planner.coast_safe(state, height, latency + 0.035):
                return None
            self.last_search = captured
            route = self.fine.plan(state, height, latency, interval)
        if (
            route.predicted_horizon < 1.5
            or not route.planned_inputs
            or route.planned_inputs[0][1] != inward
            or not self._crosses_opening(route, state, opening)
        ):
            return self._late_wall_departure(
                state, height, captured, latency, interval, opening, inward
            )
        offset = route.planned_inputs[0][0]
        if not self.planner.coast_safe(state, height, offset + 0.010):
            # Try just before the late coast boundary instead of selecting a
            # departure that can run into the floor before it takes effect.
            offset -= 0.006
            if offset < latency or not self.planner.coast_safe(state, height, offset + 0.010):
                return self._late_wall_departure(
                    state, height, captured, latency, interval, opening, inward
                )
            route = self.fine.plan(state, height, offset, interval, first_action=inward)
            if route.action != inward or not self._crosses_opening(route, state, opening):
                return self._late_wall_departure(
                    state, height, captured, latency, interval, opening, inward
                )
            if offset > latency + 0.035:
                route = replace(route, action=0)
        self._remember(route, captured)
        return replace(route, reason="leaving the wall through the opening")

    def _late_wall_departure(self, state, height, captured, latency, interval, opening, inward):
        top, bottom, left, right = opening
        edge = left if inward > 0 else right
        if state.vy < 450 or not any(
            85 <= w <= 110
            and 75 <= h <= 110
            and abs(vx) < 10
            and abs(vy) < 10
            and 70 <= y - bottom <= 230
            and x - 12 <= edge <= x + w + 12
            and (
                i >= len(state.obstacle_bounds)
                or state.obstacle_bounds[i, 0] >= state.obstacle_bounds[i, 1]
            )
            for i, (x, y, w, h, vx, vy) in enumerate(state.obstacles)
        ):
            return None
        if not self.planner.coast_safe(state, height, latency + 0.055):
            return None

        # Try the lower part of the arc without letting the falling player
        # reach the lower bar before the jump can take effect.
        safe, limit = latency, latency + 0.320
        for _ in range(8):
            mid = (safe + limit) / 2
            if self.planner.coast_safe(state, height, mid):
                safe = mid
            else:
                limit = mid
        half_w = state.w / 2 + self.planner.margin
        half_h = state.h / 2 + self.planner.margin
        goal = (top - half_h, left + half_w, right - half_w)
        for reserve in (0.026, 0.041):
            offset = safe - reserve
            if offset < latency + 0.035 or not self.planner.coast_safe(
                state, height, offset + 0.022
            ):
                continue
            route = self.pocket_search.plan(
                state, height, offset, interval, first_action=inward, _opening=goal
            )
            if len(route.planned_inputs) >= 2 and self._crosses_opening(route, state, opening):
                route = replace(
                    route, action=0, reason="waiting to leave the wall around the large block"
                )
                self._stage_pocket(route, captured, opening)
                return route
        return None

    def _keep_early_turn(self, normal, state, height, captured, latency, interval, opening):
        _, bottom, left, right = opening
        inward = 1 if state.x < 120 else -1 if state.x > 300 else 0
        half = state.w / 2 + self.planner.margin
        side_opening = left + half > 210 or right - half < 210
        if (
            side_opening
            and normal.action == -inward
            and inward * state.vx > 0
            and left < state.x < right
        ):
            if 20 < state.y - bottom < 100 and normal.planned_inputs[0][1] == -inward:
                wait = self.fine.plan(state, height, latency, interval, first_action=0)
                if (
                    wait.predicted_horizon >= normal.predicted_horizon
                    and wait.planned_inputs
                    and wait.planned_inputs[0][1] == inward
                    and self._crosses_opening(wait, state, opening)
                    and self.planner.coast_safe(state, height, latency + interval)
                ):
                    self._remember(wait, captured)
                    return replace(wait, reason="coasting to the opening approach")
        outward = -1 if state.x < left else 1 if state.x > right else 0
        if (
            outward
            and normal.action == outward
            and len(normal.planned_inputs) >= 2
            and normal.planned_inputs[1][1] == -outward
            and normal.planned_inputs[1][0] - latency <= 0.40
            and self._crosses_opening(normal, state, opening)
        ):
            self._remember(normal, captured)
            return replace(normal, reason="starting the opening approach")
        return None

    def _follow(self, state, height, captured, latency, interval):
        target = self.pending[0]
        offset = target.effect_at - captured
        if offset < latency - 0.025:
            return None
        offset = max(latency, offset)
        if offset > latency + 0.035:
            # While still below the opening, a fresh pose can give us a safe
            # inward reset sooner than the planned wall departure.
            at_wall = (state.x <= 1 and target.direction == 1) or (
                state.x >= 419 and target.direction == -1
            )
            if at_wall and state.vx == 0 and state.vy > 500 and offset <= latency + 0.16:
                earlier = self.planner.plan(state, height, latency, interval)
                if (
                    earlier.action == target.direction
                    and earlier.predicted_horizon >= 1.5
                    and earlier.planned_inputs
                    and earlier.planned_inputs[0][0] < offset - 0.02
                ):
                    opening = next(
                        (g for g in bar_gaps(state.obstacles) if 20 < state.y - g[1] < 410), None
                    )
                    if opening is not None and self._crosses_opening(earlier, state, opening):
                        self.pending.clear()
                        return earlier
            # A bar deadline already has a checked latest coast time.
            # While waiting, keep it through that time and recheck with the
            # full release margin on the frame that will send it.
            reserve = 0.0 if target is self.bar_deadline else 0.010
            if self.planner.coast_safe(state, height, offset + reserve):
                return Decision(0, np.empty((0, 2)), "coasting to the opening approach")
            return None
        phases = ((self.fine, 0.0), (self.fine, 0.014), (self.fine, 0.026), (self.refined, 0.014))
        cooldown = self.planner.gap_cooldown(state) if target is self.gap_followup else None
        for planner, adjustment in phases:
            timing = offset + adjustment
            if not self.planner.coast_safe(state, height, timing + 0.010):
                continue
            route = planner.plan(
                state, height, timing, interval, first_action=target.direction, _cooldown=cooldown
            )
            if route.action == target.direction and route.predicted_horizon >= 1.5:
                if target is self.short_bar_return:
                    # A nominal crossing may only work at one delivery
                    # time. Check the same turn with a later reset before
                    # preferring it to a fresh route.
                    later = planner.plan(
                        state, height, timing + 0.012, interval, first_action=target.direction
                    )
                    if later.action != target.direction or later.predicted_horizon < 1.5:
                        return None
                    route = replace(route, reset_window=(timing, timing + 0.012))
                if (
                    cooldown is not None
                    and cooldown < self.planner.cooldown
                    and len(self.pending) == 1
                ):
                    self._continue_gap_after_send(route, captured)
                self.proposed = True
                return replace(route, reason="following the opening approach")
        return None

    def _continue_gap_after_send(self, route, captured):
        if len(route.planned_inputs) >= 2:
            (first, direction), (second, reversal) = route.planned_inputs[:2]
            if reversal == -direction and second - first < self.planner.cooldown:
                self.next_gap_followup = ScheduledInput(captured + second, reversal)

    def _through_pocket(self, state, height, captured, latency, interval, opening, offset):
        top, _, left, right = opening
        margin = self.planner.margin
        half = state.w / 2 + margin
        goal = (top - state.h / 2 - margin, left + half, right - half)
        route = self.pocket_search.plan(state, height, offset, interval, _opening=goal)
        if not route.planned_inputs or not self._crosses_opening(route, state, opening):
            return None
        if route.action:
            adjusted = self.pocket_search.time_pocket_jump(
                state, route, height, latency, interval, opening=goal
            )
            if self._crosses_opening(adjusted, state, opening):
                route = adjusted
        first, _ = route.planned_inputs[0]
        if not self.planner.coast_safe(state, height, first + 0.022):
            return None
        if first > latency + 0.032:
            route = replace(route, action=0)
        self._stage_pocket(route, captured, opening)
        return replace(
            route,
            reason="routing through the opening"
            if route.action
            else "coasting to the opening route",
        )

    def _stage_pocket(self, route, captured, opening):
        self.pocket_edges = opening[2:]
        self.pending.clear()
        self._remember(route, captured, limit=len(route.planned_inputs))

    def _low_wall_departure(self, state, height, captured, latency, interval, opening, inward):
        top, bottom, left, right = opening
        edge = left if inward > 0 else right
        block = next(
            (
                b
                for b in state.obstacles
                if 28 <= b[2] <= 42
                and 35 <= b[1] - bottom <= 150
                and b[0] - 9 <= edge <= b[0] + b[2] + 9
            ),
            None,
        )
        if block is None:
            return None
        x, y, width, _, _, _ = block
        half_w = state.w / 2 + self.planner.margin
        half_h = state.h / 2 + self.planner.margin
        speed = self.planner.p.speed * state.motion_rate
        gravity = self.planner.p.gravity * state.motion_rate**2 * state.gravity_gain
        jump = self.planner.p.jump * state.motion_rate * state.jump_gain

        # Center the second jump's apex vertically between the bar and the
        # block, accounting for the height lost at each end of the block.
        ceiling = bottom + half_h + self.planner.underside_margin
        floor = y - half_h - 0.5 * gravity * ((width / 2 + half_w) / speed) ** 2
        distance = x + width / 2 if inward > 0 else 420 - x - width / 2
        flight = distance / speed - jump / gravity
        if ceiling >= floor or not self.planner.cooldown <= flight <= 0.94:
            return None
        launch_y = (
            (ceiling + floor) / 2
            + jump**2 / (2 * gravity)
            + jump * flight
            - 0.5 * gravity * flight**2
        )
        if launch_y <= state.y:
            return None
        offset = (sqrt(state.vy**2 + 2 * gravity * (launch_y - state.y)) - state.vy) / gravity
        if not latency + 0.050 <= offset <= latency + 0.320:
            return None
        if not self.pocket_search.coast_safe(state, height, offset + 0.010):
            return None
        goal = (top - half_h, left + half_w, right - half_w)
        route = self.pocket_search.plan(
            state, height, offset, interval, first_action=inward, _opening=goal
        )
        if len(route.planned_inputs) < 2 or not self._crosses_opening(route, state, opening):
            return None
        route = replace(route, action=0, reason="waiting for the low wall departure")
        self._stage_pocket(route, captured, opening)
        self.wall_departure = self.pending[0]
        return route

    def _pocket_coast_planner(self):
        if self.pending and self.pending[0] is self.wall_departure:
            return self.pocket_search
        return self.planner

    def _continue_pocket(self, state, height, captured, latency):
        """Check the next scheduled arc on a fresh frame before running another beam search."""
        if len(self.pending) < 2 or state.shapes:
            return None
        target = self.pending[0]
        coast_planner = self._pocket_coast_planner()
        offset = target.effect_at - captured
        following = self.pending[1].effect_at - captured
        max_offset = following - self.planner.cooldown - 0.010
        center = max(latency, offset)
        if offset < latency - 0.020 or center > max_offset:
            return None
        timings = np.linspace(max(0.0, center - 0.020), min(max_offset, center + 0.020), 9)
        valid = []
        for when in timings:
            if state.age_since_click + when < self.planner.cooldown:
                valid.append(False)
                continue
            duration = following - when + 0.010
            scores, _, _, _ = self.pocket_search._immediate_probe(
                state,
                height,
                when,
                duration,
                first_action=target.direction,
                margin=self.planner.margin,
            )
            valid.append(
                scores[target.direction + 1] + 1e-7 >= duration
                and coast_planner.coast_safe(state, height, when)
            )
        edges = np.flatnonzero(np.diff(np.r_[False, valid, False]))
        best = None
        for start, end in zip(edges[::2], edges[1::2]):
            left, right = timings[start], timings[end - 1]
            when = max(latency, (left + right) / 2)
            reserve = min(when - left, right - when)
            if reserve >= 0.008 and (best is None or reserve > best[0]):
                best = reserve, when, left, right
        if best is None:
            return None
        _, when, left, right = best
        duration = following - when
        _, action, path, survived = self.pocket_search._immediate_probe(
            state,
            height,
            when,
            duration,
            first_action=target.direction,
            margin=self.planner.margin,
        )
        if action != target.direction or survived + 1e-7 < duration:
            return None
        self.proposed = True
        inputs = [(item.effect_at - captured, item.direction) for item in self.pending]
        inputs[0] = (float(when), action)
        return Decision(
            action,
            path,
            "following the opening route",
            predicted_horizon=duration,
            planned_inputs=inputs,
            reset_window=(float(left), float(right)),
        )

    def plan(self, state, height, captured, latency, interval, *, latest_effect=None):
        route = self._plan_with_followups(state, height, captured, latency, interval, latest_effect)
        if (
            self.pending
            or state.shapes
            or not route.planned_inputs
            or abs(state.vx) < 100
            or (latest_effect is not None and latency >= latest_effect)
        ):
            return route
        if route.action:
            revised = self._check_short_bar_route(route, state, height, captured, latency, interval)
            if revised is None:
                revised = self._avoid_tight_apex(route, state, height, latency, interval)
        else:
            revised = self._avoid_narrow_bar_exit(route, state, height, latency, interval)
        return revised if revised is not None else route

    def _avoid_tight_apex(self, route, state, height, latency, interval):
        if route.predicted_horizon < 1.5 or state.vy >= -100 or route.reset_window is not None:
            return None
        start, direction = route.planned_inputs[0]
        if direction != route.action:
            return None
        physics = self.planner.p
        gravity = physics.gravity * state.motion_rate**2 * state.gravity_gain
        jump = physics.jump * state.motion_rate * state.jump_gain
        apex = jump / gravity
        if len(route.planned_inputs) > 1 and route.planned_inputs[1][0] - start < apex:
            return None

        # A jump that just misses an underside near its apex can hit it
        # if a later reset starts higher on the current rising arc.
        speed = physics.speed * state.motion_rate
        x_at_reset = min(max(state.x + state.vx * start, 0), 420)
        x = min(max(x_at_reset + direction * speed * apex, 0), 420)
        y = state.y + state.vy * start + 0.5 * gravity * start**2 - jump * apex / 2 - state.h / 2
        delay = 0.012
        reserve = self.planner.margin + max(0, -state.vy - gravity * start) * delay + 1
        half = state.w / 2 + 10
        if not any(
            0 <= y - top - h <= reserve
            and left - half <= x <= left + w + half
            and abs(vx) + abs(vy) < 10
            for left, top, w, h, vx, vy in state.obstacles
        ):
            return None
        _, _, _, survived = self.planner._immediate_probe(
            state,
            height,
            start + delay,
            apex,
            first_action=direction,
            margin=self.planner.margin,
        )
        if survived + 1e-7 >= apex or not self.planner.coast_safe(
            state, height, latency + interval + delay
        ):
            return None
        wait = self.planner.plan(state, height, latency, interval, first_action=0)
        if wait.predicted_horizon >= 1.5:
            return replace(wait, reason="waiting before a timing-sensitive apex")
        return None

    def _check_short_bar_route(self, route, state, height, captured, latency, interval):
        direction = route.action
        if not 0.65 <= route.predicted_horizon < 1.6 or direction * state.vx < 0:
            return None
        for gap in bar_gaps(state.obstacles):
            _, bottom, left, right = gap
            near_edge = (
                left - 10 <= state.x <= left + 45
                if direction > 0
                else right - 45 <= state.x <= right + 10
            )
            if not near_edge or not 0 < state.y - bottom < 78:
                continue
            # The coarse plan gets into the gap but may miss a quick return
            # that keeps the player clear of the next obstacle.
            checked = self.fine.plan(state, height, latency, interval, first_action=direction)
            if (
                checked.action != direction
                or checked.predicted_horizon < 2.5
                or len(checked.planned_inputs) < 2
                or not self._crosses_opening(checked, state, gap)
            ):
                continue
            first, second = checked.planned_inputs[:2]
            if second[1] == -direction and second[0] - first[0] <= self.planner.cooldown + 0.040:
                self._remember(checked, captured)
                self.short_bar_return = self.pending[1]
                return replace(checked, reason="checking the return through the nearby bar")
        return None

    def _avoid_narrow_bar_exit(self, route, state, height, latency, interval):
        when, direction = route.planned_inputs[0]
        if (
            not 0.23 <= when <= 0.45
            or direction * state.vx <= 0
            or route.predicted_horizon < 2
            or state.vy >= 0
        ):
            return None
        half = state.w / 2 + self.planner.margin
        x = state.x + state.vx * when
        for gap in bar_gaps(state.obstacles):
            top, bottom, left, right = gap
            if not (14 < state.y - bottom < 120 and left + half < state.x < right - half):
                continue
            # A square above the bar can leave only one player's width on
            # the side the current coast is approaching.
            squeezed = any(
                26 <= w <= 45
                and 25 <= h <= 45
                and abs(vx) < 10
                and abs(vy) < 10
                and 35 <= top - y - h <= 165
                and (
                    i >= len(state.obstacle_bounds)
                    or state.obstacle_bounds[i, 0] >= state.obstacle_bounds[i, 1]
                )
                and (
                    (half < right - (box_x + w) <= 2 * half + 13 and box_x < x < right)
                    if direction > 0
                    else (half < box_x - left <= 2 * half + 13 and left < x < box_x + w)
                )
                for i, (box_x, y, w, h, vx, vy) in enumerate(state.obstacles)
            )
            if not squeezed:
                continue
            earlier = self.fine.plan(state, height, latency, interval, first_action=-direction)
            if (
                earlier.action != -direction
                or earlier.predicted_horizon < 2.5
                or len(earlier.planned_inputs) < 2
                or not self._crosses_opening(earlier, state, gap)
            ):
                continue
            duration = earlier.planned_inputs[1][0] - earlier.planned_inputs[0][0]
            _, action, _, survived = self.planner._immediate_probe(
                state,
                height,
                latency + 0.010,
                duration,
                first_action=-direction,
                margin=self.planner.margin,
            )
            if action == -direction and survived + 1e-7 >= duration:
                return replace(
                    earlier,
                    reason="reversing before a narrow exit above the bar",
                    reset_window=(latency, latency + 0.010),
                )
        return None

    def _plan_with_followups(self, state, height, captured, latency, interval, latest_effect):
        route = self._plan(state, height, captured, latency, interval, latest_effect)
        if self.pending or state.shapes or not route.planned_inputs:
            return route
        if (
            route.action
            and len(route.planned_inputs) >= 2
            and route.predicted_horizon >= 1.5
            and route.reason.startswith(
                ("narrow gap:", "upper gap reversal:", "timing the turn through the square gap")
            )
        ):
            first, second = route.planned_inputs[:2]
            if (
                second[1] == -first[1]
                and second[0] - first[0] < self.planner.cooldown
                and self.planner.gap_cooldown(state) < self.planner.cooldown
            ):
                self._remember(route, captured)
                self.gap_followup = self.pending[1]
                return replace(route, reason="keeping the follow-up through the square gap")
        if route.predicted_horizon < 2.0:
            return route
        if not route.action:
            return self._keep_bar_deadline(route, state, height, captured, latency, interval)
        if route.action * state.vx >= 0 or len(route.planned_inputs) < 2:
            return route
        first, second = route.planned_inputs[:2]
        spacing = second[0] - first[0]
        if spacing > self.planner.cooldown + 0.070:
            return route
        blocks = [
            (x, x + w)
            for i, (x, y, w, h, vx, vy) in enumerate(state.obstacles)
            if 85 <= w <= 110
            and 75 <= h <= 110
            and abs(vx) < 10
            and abs(vy) < 10
            and 175 < state.y - y - h < 385
            and x - 65 < state.x < x + w + 65
            and (
                i >= len(state.obstacle_bounds)
                or state.obstacle_bounds[i, 0] >= state.obstacle_bounds[i, 1]
            )
        ]
        if not blocks:
            if second[1] == -route.action and self._returns_through_bar(
                route, state, first[0], spacing
            ):
                self._remember(route, captured)
                self.bar_return = self.pending[1]
                return replace(route, reason="keeping the return turn before the upper bar")
            return route

        # An early launch can commit us to a rapid second reset to stay on
        # the planned side of an incoming block. If already heading clear,
        # prefer a safe coast that gives the next reset more room.
        moving_clear = any(
            (state.x < left and state.vx < 0) or (state.x > right and state.vx > 0)
            for left, right in blocks
        )
        if moving_clear and self.planner.coast_safe(state, height, latency + 0.025):
            wait = self.planner.plan(state, height, latency, interval, first_action=0)
            if wait.predicted_horizon >= route.predicted_horizon and len(wait.planned_inputs) >= 2:
                start, following = wait.planned_inputs[:2]
                if (
                    start[0] <= latency + 0.180
                    and following[0] - start[0] >= spacing + 0.020
                    and self.planner.coast_safe(state, height, start[0] + 0.014)
                ):
                    self._remember(wait, captured)
                    return replace(wait, reason="waiting for the wider route around a large block")
        self._remember(route, captured)
        return replace(route, reason="keeping the follow-up around a large block")

    def _returns_through_bar(self, route, state, first_effect, spacing):
        # The launch heads across the opening, so it needs the second reset
        # to turn back before reaching the edge of the gap.
        half = state.w / 2 + self.planner.margin
        x = (
            state.x
            + state.vx * first_effect
            + route.action * self.planner.p.speed * state.motion_rate * spacing
        )
        return any(
            220 < state.y - gap[1] < 315
            and gap[2] <= state.x <= gap[3]
            and (x >= gap[3] - half if route.action > 0 else x <= gap[2] + half)
            and self._crosses_opening(route, state, gap)
            for gap in bar_gaps(state.obstacles)
        )

    def _can_replace_bar_return(self, route, state, remaining, latency, interval):
        if route.predicted_horizon < 2.0 or not route.planned_inputs:
            return False
        if not route.action:
            # Retain the queued reset until its send window. Replace it with
            # a coast only if the next observation leaves time to send the
            # alternative reset, including a small scheduling reserve.
            first = route.planned_inputs[0][0]
            if remaining > latency + 0.035 or first < latency + interval + 0.004:
                return False
        opening = next((g for g in bar_gaps(state.obstacles) if 0 < state.y - g[1] < 315), None)
        return opening is not None and self._crosses_opening(route, state, opening)

    def _keep_bar_deadline(self, route, state, height, captured, latency, interval):
        when, _ = route.planned_inputs[0]
        # The projected contact must be with this bar's edge. A different
        # nearby obstacle can also limit the coast, but needs its own route.
        half = state.w / 2 + self.planner.margin + 4
        projected = state.x + state.vx * (when + interval)
        if not 0.030 - 1e-7 <= when - latency <= 0.150 + 1e-7 or not any(
            left <= state.x <= right
            # If the pose is changing quickly relative to the bar itself,
            # keep replanning until the timing stabilizes.
            and abs(state.vy) * interval <= 0.8 * (bottom - top)
            and (
                (state.vx < 0 and projected <= left + half)
                or (state.vx > 0 and projected >= right - half)
            )
            and (
                (0 < state.y - bottom <= 85 and state.vy < 0)
                or (top - 115 <= state.y <= bottom and state.vy > 0)
            )
            for top, bottom, left, right in bar_gaps(state.obstacles)
        ):
            return route
        # If another control interval would close the route, preserve the
        # planned jump through the wait and check it on the next fresh pose.
        if self.planner.coast_safe(state, height, when) and not self.planner.coast_safe(
            state, height, when + interval
        ):
            self._remember(route, captured, limit=1)
            self.bar_deadline = self.pending[0]
            return replace(route, reason="keeping the jump before the bar closes the route")
        return route

    def _plan(self, state, height, captured, latency, interval, latest_effect):
        self.proposed = False
        self.next_gap_followup = None
        if self.pocket_edges is not None and state.ready:
            left, right = self.pocket_edges
            opening = next(
                (
                    g
                    for g in bar_gaps(state.obstacles)
                    if abs(g[2] - left) <= 4
                    and abs(g[3] - right) <= 4
                    and state.y > g[0] - state.h / 2
                ),
                None,
            )
            if opening is not None and self.pending:
                offset = max(latency, self.pending[0].effect_at - captured)
                if (
                    latest_effect is not None
                    and self.pending[0] is self.wall_departure
                    and latency + 0.032 < offset <= latency + interval + 0.022
                ):
                    # A dropped frame can miss the low departure entirely.
                    # Use this frame if its checked window fits the sender's
                    # deadline.
                    early = self._continue_pocket(state, height, captured, latency)
                    if early is not None and early.planned_inputs[0][0] < latest_effect:
                        return early
                    self.proposed = False
                if offset > latency + 0.032 and self._pocket_coast_planner().coast_safe(
                    state, height, offset + 0.010
                ):
                    return Decision(0, np.empty((0, 2)), "coasting to the opening route")
                route = self._continue_pocket(state, height, captured, latency)
                if route is not None:
                    return route
                route = self._through_pocket(
                    state, height, captured, latency, interval, opening, offset
                )
                if route is not None:
                    return route
            self.pending.clear()
            self.pocket_edges = None
        normal = None
        if self.pending and state.ready:
            target = self.pending[0]
            if target is self.bar_deadline and target.effect_at - captured > latency + 0.035:
                normal = self.planner.plan(state, height, latency, interval)
                if normal.action and normal.predicted_horizon >= 1.5:
                    self.pending.clear()
                    return normal
            elif target is self.bar_return:
                normal = self.planner.plan(state, height, latency, interval)
                if self._can_replace_bar_return(
                    normal, state, target.effect_at - captured, latency, interval
                ):
                    self.pending.clear()
                    return normal
            decision = self._follow(state, height, captured, latency, interval)
            if decision is not None:
                return decision
            self.pending.clear()

        if normal is None:
            normal = self.planner.plan(state, height, latency, interval)
        if (
            normal.predicted_horizon < 0.7
            or captured - self.last_search < 0.065
            or any(abs(box[4]) > 30 for box in state.obstacles)
        ):
            return normal

        opening = next((g for g in bar_gaps(state.obstacles) if 20 < state.y - g[1] < 410), None)
        # A still polygon well above the exit does not rule out staging
        # below it. The trajectory search still checks its geometry.
        if opening is None or any(
            shape.angle_limits is not None
            or abs(shape.omega) > 0.03
            or np.max(np.abs(shape.velocity)) > 10
            or np.max(shape.points[:, 1]) + shape.uncertainty >= opening[0] - state.h
            for shape in state.shapes
        ):
            return normal
        recovery = self._leave_wall(normal, state, height, captured, latency, interval, opening)
        if recovery is not None:
            return recovery
        if normal.action and normal.planned_inputs:
            early = self._keep_early_turn(
                normal, state, height, captured, latency, interval, opening
            )
            if early is not None:
                return early

        top, bottom, left, right = opening
        if state.x < left - 50:
            edge, inward = left, 1
        elif state.x > right + 50:
            edge, inward = right, -1
        else:
            return normal
        if state.camera_speed > 15 or not any(
            28 <= w <= 42
            and 27 <= h <= 44
            and 35 <= y - bottom <= 150
            and x - 9 <= edge <= x + w + 9
            for x, y, w, h, _, _ in state.obstacles
        ):
            return normal

        clear_y = top - state.h / 2 - self.planner.margin
        if np.min(normal.path[:, 1]) <= clear_y:
            return normal
        if normal.action and not self.planner.coast_safe(state, height, latency + 0.035):
            return normal

        # Search laterally from lower on the current arc. A reset at full
        # height can hit the bar or the small block just below its edge.
        self.last_search = captured
        if any(
            25 <= w <= 44
            and 25 <= h <= 44
            and 165 <= y - bottom <= 310
            and abs(x + w / 2 - edge) <= 65
            for x, y, w, h, _, _ in state.obstacles
        ):
            focused = self._through_pocket(
                state, height, captured, latency, interval, opening, latency
            )
            if focused is not None:
                return focused
            # The two jumps through this corridor can fall between search
            # ticks. Try a shifted grid while there is time to coast.
            near_wall = state.x <= 6 if inward > 0 else state.x >= 414
            if (
                near_wall
                and not normal.action
                and state.vy > 200
                and self.planner.coast_safe(state, height, latency + 0.065)
            ):
                offset = latency + 0.7 * self.pocket_search.step
                focused = self._through_pocket(
                    state, height, captured, latency, interval, opening, offset
                )
                if focused is not None:
                    return focused
            at_wall = state.x <= 2 if inward > 0 else state.x >= 418
            if at_wall and state.vx == 0 and state.vy > 150:
                departure = self._low_wall_departure(
                    state, height, captured, latency, interval, opening, inward
                )
                if departure is not None:
                    return departure
        gravity = self.planner.p.gravity * state.motion_rate**2 * state.gravity_gain
        last_offset = -1.0
        for clearance in (65, 95, 42, 27):
            y = height - state.h / 2 - clearance
            discriminant = state.vy**2 + 2 * gravity * (y - state.y)
            if discriminant < 0:
                continue
            offset = max(latency, (sqrt(discriminant) - state.vy) / gravity)
            if offset > 1.05 or abs(offset - last_offset) < 0.004:
                continue
            last_offset = offset
            if not self.planner.coast_safe(state, height, offset + 0.012):
                continue
            route = self.fine.plan(state, height, offset, interval, first_action=inward)
            if (
                route.action != inward
                or route.predicted_horizon < 2.0
                or np.min(route.path[:, 1]) > clear_y
            ):
                continue
            timings = (max(latency, offset - 0.012), offset + 0.012)
            if any(
                self.planner.turn_probe.plan(
                    state, height, delay, interval, first_action=inward
                ).predicted_horizon
                < 0.72
                for delay in timings
            ):
                continue
            if offset > latency + 0.025:
                route = replace(route, action=0, reason="coasting to the opening approach")
            else:
                route = replace(route, reason="starting the opening approach")
            self._remember(route, captured)
            return route
        return normal


@dataclass
class Reversal:
    captured: float
    pressed: float
    velocity: float
    previous_intercept: float
    new_positions: list


class ResponseTiming:
    """Estimate visible response from the meeting point of two horizontal arcs."""

    def __init__(self, speed, initial_delay, delay_range=None):
        self.speed = speed
        self.delay = initial_delay
        self.minimum_delay, self.maximum_delay = delay_range or (0.006, 0.075)
        self.samples = deque(maxlen=15 if delay_range else 9)
        self.positions = deque(maxlen=12)
        self.pending = None
        self.last_press = -100.0

    def observe(self, state, captured):
        measured = None
        if self.pending is not None and captured >= self.pending.pressed + 0.18:
            measured = self._finish()
        if state is None or not (6 < state.x < 414 and 25 <= state.w <= 35):
            return measured
        self.positions.append((captured, state.x))
        reversal = self.pending
        if reversal is not None and captured >= reversal.pressed + 0.092:
            intercept = state.x - reversal.velocity * (captured - reversal.captured)
            reversal.new_positions.append(intercept)
        return measured

    def pressed(self, state, captured, pressed, direction):
        measured = self._finish()
        velocity = self.speed * direction
        if 20 < state.x < 400 and velocity * state.vx <= -0.90 * self.speed**2:
            start = max(captured - 0.100, self.last_press + 0.08)
            intercepts = [
                x + velocity * (t - captured) for t, x in self.positions if start <= t <= captured
            ]
            if len(intercepts) >= 3 and pstdev(intercepts) <= 1.65:
                self.pending = Reversal(captured, pressed, velocity, median(intercepts), [])
        self.last_press = pressed
        return measured

    def _finish(self):
        reversal, self.pending = self.pending, None
        if reversal is None or len(reversal.new_positions) < 3:
            return None
        if pstdev(reversal.new_positions) > 1.65:
            return None
        delta = (reversal.previous_intercept - median(reversal.new_positions)) / (
            2 * reversal.velocity
        )
        measured = reversal.captured + delta - reversal.pressed
        if not -0.008 < measured < 0.112:
            return None
        self.samples.append(measured)
        if len(self.samples) >= 7:
            self.delay = min(self.maximum_delay, max(self.minimum_delay, median(self.samples)))
        return measured

    def clear_observations(self):
        self.pending = None
        self.positions.clear()


class TimedController:
    def __init__(self, planner, input_lag=0.035, minimum_horizon=0.5, max_press_age=0.070):
        self.planner = planner
        self.input_lag = input_lag
        self.minimum_horizon = minimum_horizon
        self.max_press_age = max_press_age
        self.pending = None
        self._proposal = None
        self.stats = {"verified": 0, "invalidated": 0, "waits": 0}

    def reset(self):
        self.pending = None
        self._proposal = None

    @staticmethod
    def _next(decision, captured, after):
        return next(
            (
                ScheduledInput(captured + offset, direction)
                for offset, direction in decision.planned_inputs
                if captured + offset > after + 1e-6
            ),
            None,
        )

    def confirm_sent(self, actual_effect):
        if self._proposal is None:
            return
        effect, following = self._proposal
        self.pending = (
            ScheduledInput(following.effect_at + actual_effect - effect, following.direction)
            if following
            else None
        )
        self._proposal = None

    def plan(self, state, height, captured, latency, control_interval):
        self._proposal = None
        if not state.ready:
            return Command(self.planner.plan(state, height, latency, control_interval))
        earliest = captured + latency
        if self.pending is not None:
            effect = max(self.pending.effect_at, earliest)
            direction = self.pending.direction
            if self.planner.coast_safe(state, height, effect - captured):
                verified = self.planner.plan(
                    state, height, effect - captured, control_interval, first_action=direction
                )
                if (
                    verified.action == direction
                    and verified.predicted_horizon >= self.minimum_horizon
                ):
                    self.stats["verified"] += 1
                    due = (
                        self.pending.effect_at <= earliest + control_interval
                        and effect - self.input_lag <= captured + self.max_press_age
                    )
                    if due:
                        self._proposal = (effect, self._next(verified, captured, effect))
                        verified.reason = "verified committed input: " + verified.reason
                        return Command(verified, effect, True)
                    verified.action = 0
                    self.stats["waits"] += 1
                    verified.reason = "coasting to verified committed input"
                    return Command(verified, None, True)
            self.stats["invalidated"] += 1
            self.pending = None
        decision = self.planner.plan(state, height, latency, control_interval)
        if decision.action:
            self._proposal = (earliest, self._next(decision, captured, earliest))
            return Command(decision, earliest)
        self.pending = self._next(decision, captured, captured - 1e-6)
        return Command(decision)
