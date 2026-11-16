import numpy as np

from .native_search import search, shape_arrays
from .navigation import Navigation
from .physics import State
from .planner import Decision


class FastPlanner:
    def __init__(
        self, physics, horizon=3.0, step=0.05, beam=900, margin=4.0, cooldown=0.12, click_cost=12.0
    ):
        self.p = physics
        self.horizon, self.step, self.beam = horizon, step, beam
        self.margin, self.cooldown, self.click_cost = margin, cooldown, click_cost

    def plan(self, state, board_height, latency=0.02):
        if not state.ready:
            return Decision(0, np.empty((0, 2)), "waiting for reliable tracking")
        nav = Navigation(state, self.margin + 1)
        # Guidance is a smooth potential field; exact collision geometry is
        # checked independently in native search at every intermediate sample.
        yy, xx = np.mgrid[0 : board_height + 11 : 10, 0:431:10]
        values = nav.cost(xx.reshape(-1), yy.reshape(-1)).reshape(xx.shape)
        levels = round(self.horizon / self.step)
        substeps = 5
        times = np.concatenate(
            [
                latency + d * self.step + np.linspace(0, self.step, substeps + 1)
                for d in range(levels)
            ]
        )
        bounds, axes, limits, counts = shape_arrays(state.shapes, times)
        initial = np.array(
            [state.x, state.y, state.vx, state.vy, state.w, state.h, state.age_since_click]
        )
        action, path, depth, score = search(
            initial,
            state.obstacles,
            values,
            bounds,
            axes,
            limits,
            counts,
            self.p.speed,
            self.p.jump,
            self.p.gravity,
            latency,
            self.step,
            levels,
            self.beam,
            self.margin,
            self.cooldown,
            self.click_cost,
            board_height,
            substeps,
        )
        reason = (
            f"predicted {depth * self.step:.2f}s (native)"
            if depth
            else "no collision-free immediate action"
        )
        return Decision(action, path, reason, score, depth * self.step)

    def warmup(self):
        self.plan(State(210, 500, 0, 0, 30, 32, ready=True), 745, latency=0)
