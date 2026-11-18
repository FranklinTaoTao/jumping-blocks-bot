import unittest
from unittest.mock import Mock

import numpy as np

from jumpbot.controller import ResponseTiming, TimedController, WallStall
from jumpbot.fast_planner import FastPlanner
from jumpbot.physics import Physics, State
from jumpbot.planner import Decision


class WallStallTests(unittest.TestCase):
    def test_timeout_stays_latched_until_restart(self):
        stall = WallStall()
        wall = State(0, 500, 0, 0, 30, 30, wall_dwell=9.99)
        self.assertFalse(stall.update(wall, 9.99, 28))
        wall.wall_dwell = 10.0
        self.assertTrue(stall.update(wall, 10.0, 28))
        self.assertTrue(stall.update(None, 10.01, 28))
        self.assertTrue(stall.update(State(210, 500, 0, 0, 30, 30), 11.0, 28))
        stall.reset()
        self.assertFalse(stall.update(State(210, 500, 0, 0, 30, 30), 12.0, 0))

    def test_repeated_returns_only_accumulate_wall_time_without_progress(self):
        stall = WallStall()
        moving = State(120, 500, 150, 0, 30, 30)
        wall = State(0, 500, 0, 0, 30, 30)
        for i in range(200):
            in_contact = i % 50
            wall.wall_dwell = max(0, in_contact - 20) / 10
            state = wall if wall.wall_dwell else moving
            ended = stall.update(state, i / 10, 28)
            if ended:
                break
        self.assertTrue(ended)
        self.assertGreater(i / 10, 10)
        self.assertGreaterEqual(stall.wall_time, 10)

        for advance_score in (True, False):
            stall.reset()
            for i in range(250):
                interval = 50 if advance_score else 70
                in_contact = i % interval
                wall.wall_dwell = max(0, in_contact - (interval - 30)) / 10
                state = wall if wall.wall_dwell else moving
                score = 28 + (i // interval if advance_score else 0)
                self.assertFalse(stall.update(state, i / 10, score))


class TimingTests(unittest.TestCase):
    def setUp(self):
        self.state = State(210, 500, 0, 0, 30, 30, ready=True)

    def test_response_timing_adapts_to_a_sustained_change(self):
        for limits, expected in ((None, 0.011), ((0.020, 0.050), 0.020)):
            with self.subTest(limits=limits):
                timing = ResponseTiming(150, 0.035, limits)
                for index, actual_delay in enumerate([0.041] * 8 + [0.011] * 9):
                    captured = 1.0 + index * 0.5
                    pressed = captured + 0.02
                    effect = pressed + actual_delay
                    for offset in (-0.085, -0.068, -0.051, -0.034, -0.017, 0):
                        self.state.x = 210 + 150 * offset
                        self.state.vx = 150
                        timing.observe(self.state, captured + offset)
                    timing.pressed(self.state, captured, pressed, -1)
                    for offset in (0.10, 0.12, 0.14, 0.16, 0.19):
                        t = pressed + offset
                        self.state.x = 210 + 150 * (effect - captured) - 150 * (t - effect)
                        self.state.vx = -150
                        measured = timing.observe(self.state, t)
                    self.assertAlmostEqual(measured, actual_delay)
                    if index == 7:
                        self.assertAlmostEqual(timing.delay, 0.041)
                self.assertAlmostEqual(timing.delay, expected)

    def test_waiting_does_not_postpone_the_planned_input(self):
        planner = Mock()
        planner.coast_safe.return_value = True

        def plan(state, height, latency, interval, first_action=None):
            if first_action is None:
                return Decision(
                    0,
                    np.empty((0, 2)),
                    "free",
                    predicted_horizon=3.0,
                    planned_inputs=[(0.2, -1), (0.4, 1)],
                )
            return Decision(
                first_action,
                np.empty((0, 2)),
                "forced",
                predicted_horizon=3.0,
                planned_inputs=[(latency, first_action), (latency + 0.2, 1)],
            )

        planner.plan.side_effect = plan
        controller = TimedController(planner)
        self.assertEqual(controller.plan(self.state, 736, 0.0, 0.06, 0.04).decision.action, 0)
        for captured in [0.04, 0.08]:
            self.assertEqual(
                controller.plan(self.state, 736, captured, 0.06, 0.04).decision.action, 0
            )
            self.assertAlmostEqual(controller.pending.effect_at, 0.2)
        command = controller.plan(self.state, 736, 0.12, 0.06, 0.04)
        self.assertEqual(command.decision.action, -1)
        self.assertAlmostEqual(command.effect_at, 0.2)
        # Merely planning a click must not advance the schedule.
        self.assertAlmostEqual(controller.pending.effect_at, 0.2)
        controller.confirm_sent(0.205)
        self.assertAlmostEqual(controller.pending.effect_at, 0.405)
        self.assertEqual(controller.pending.direction, 1)

    def test_new_obstacle_invalidates_the_old_promise(self):
        planner = Mock()
        planner.coast_safe.return_value = False
        planner.plan.side_effect = [
            Decision(
                0, np.empty((0, 2)), "wait", predicted_horizon=3.0, planned_inputs=[(0.2, -1)]
            ),
            Decision(
                1, np.empty((0, 2)), "escape", predicted_horizon=3.0, planned_inputs=[(0.06, 1)]
            ),
        ]
        controller = TimedController(planner)
        controller.plan(self.state, 736, 0.0, 0.06, 0.04)
        command = controller.plan(self.state, 736, 0.1, 0.06, 0.04)
        self.assertEqual(command.decision.action, 1)
        self.assertFalse(command.committed)
        self.assertEqual(controller.stats["invalidated"], 1)

    def test_coasting_checks_the_path_before_input_not_just_its_endpoint(self):
        planner = FastPlanner(Physics(), horizon=0.4, step=0.025, beam=60)
        state = State(
            210,
            500,
            0,
            -700,
            30,
            30,
            ready=True,
            obstacles=np.array([[0, 460, 420, 5, 0, 0]], float),
        )
        self.assertFalse(planner.coast_safe(state, 745, 0.15))
        state = State(420, 500, 150, 0, 30, 30, ready=True)
        self.assertTrue(planner.coast_safe(state, 745, 0.12))

    def test_validation_honors_each_requested_first_action(self):
        planner = FastPlanner(Physics(), horizon=0.4, step=0.025, beam=60)
        for direction in [-1, 0, 1]:
            decision = planner.plan(self.state, 745, latency=0.04, first_action=direction)
            self.assertEqual(decision.action, direction)
