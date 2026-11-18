import unittest

import numpy as np

from jumpbot.controller import OpeningApproach, ScheduledInput
from jumpbot.fast_planner import FastPlanner
from jumpbot.motion import BoxTracker, hammer_pin_rate, horizontal_extent
from jumpbot.native_search import reflected_extent
from jumpbot.navigation import clipped_bar_gaps, opening_guidance, square_gaps
from jumpbot.physics import Physics, State
from jumpbot.vision import Rect, camera_shift


class RouteRuleTests(unittest.TestCase):
    def setUp(self):
        self.planner = FastPlanner(Physics(150.4, 688.55, 1415), route_rules=True, cooldown=0.18)

    def test_immediate_safety_waits_under_a_bar(self):
        state = State(
            350,
            375,
            0,
            10,
            30,
            30,
            ready=True,
            obstacles=np.array([[260, 300, 159, 38, 0, 0]], float),
        )
        self.assertEqual(self.planner.immediate_mask(state, 745, 0.030, 0.040, 0.18), 2)
        self.assertEqual(self.planner.plan(state, 745, 0.030, 0.040).action, 0)

    def test_upward_bar_check_reserves_two_extra_pixels(self):
        boxes = np.array([[260, 300, 159, 38, 0, 0]], float)
        close = State(350, 370.5, 0, -200, 30, 30, ready=True, obstacles=boxes)
        clear = State(350, 372, 0, -200, 30, 30, ready=True, obstacles=boxes)
        base = FastPlanner(self.planner.p)
        for planner, state, expected in (
            (base, close, 0.2),
            (self.planner, close, 0.1),
            (self.planner, clear, 0.2),
        ):
            coast = planner._immediate_probe(state, 745, 0.0, 0.2, first_action=0)[0][1]
            self.assertAlmostEqual(coast, expected)

    def test_immediate_safety_jumps_when_waiting_would_hit_floor(self):
        state = State(310, 680, 0, 470, 30, 30, ready=True)
        self.assertEqual(self.planner.immediate_mask(state, 745, 0.045, 0.085, 0.18), 5)
        self.assertIn(self.planner.plan(state, 745, 0.045, 0.085).action, (-1, 1))

    def test_immediate_safety_jumps_when_the_long_search_margin_rules_it_out(self):
        state = State(310, 723.3, 0, 100, 30, 30, ready=True)
        decision = self.planner.plan(state, 745, 0.030, 0.040)
        self.assertEqual(decision.reason, "immediate safety jump")
        self.assertIn(decision.action, (-1, 1))
        self.assertEqual(decision.planned_inputs, [(0.030, decision.action)])

    def test_unknown_space_favors_a_broad_center_symmetrically(self):
        state = State(210, 410, 0, 0, 30, 30, ready=True)
        xx, yy = np.meshgrid([40, 165, 210, 255, 380], [350])
        costs = opening_guidance(state, xx, yy, 4, 150.4, 688.55, 0.18)
        np.testing.assert_allclose(costs[0], costs[2])
        np.testing.assert_allclose(costs[0, 0], costs[0, 0, ::-1])
        self.assertLess(costs[0, 0, 1], 0.1 * 36)
        self.assertGreater(costs[0, 0, 0], 0.7 * 36)

    def test_exit_heading_points_toward_the_offset_upper_opening(self):
        lower = [[1, 380, 120, 38, 0, 0], [291, 380, 128, 38, 0, 0]]
        upper = [[1, 130, 49, 38, 0, 0], [220, 130, 199, 38, 0, 0]]
        xx, yy = np.meshgrid([165, 255], [350])
        state = State(235, 415, 0, 0, 30, 30, ready=True, obstacles=np.array(lower + upper, float))
        costs = opening_guidance(state, xx, yy, 4, 150.4, 688.55, 0.18)
        self.assertLess(costs[0, 0, 1], costs[2, 0, 1])
        self.assertLess(costs[0, 0, 0], costs[0, 0, 1])

    def test_matches_the_two_edges_of_a_partly_visible_upper_bar(self):
        left = [1, 0, 49, 14, 0, 0]
        right = [220, 0, 199, 14, 0, 0]
        self.assertEqual(clipped_bar_gaps([left, right]), [(-24, 14, 50, 220)])
        self.assertEqual(clipped_bar_gaps([left, right[:3] + [22, 0, 0]]), [])

    def test_moving_squares_do_not_form_a_static_opening(self):
        xx, yy = np.meshgrid([165, 255], [350])
        boxes = np.array([[51, 180, 95, 95, 140, 0], [230, 180, 95, 95, -140, 0]], float)
        empty = State(210, 410, 0, 0, 30, 30, ready=True)
        moving = State(210, 410, 0, 0, 30, 30, ready=True, obstacles=boxes)
        np.testing.assert_allclose(
            opening_guidance(empty, xx, yy, 4, 150.4, 688.55, 0.18),
            opening_guidance(moving, xx, yy, 4, 150.4, 688.55, 0.18),
        )

    def test_moving_block_remains_moving_across_bounce_and_height_change(self):
        tracker = BoxTracker()
        for index, position in enumerate([277, 280, 283, 286]):
            tracker.update([Rect(position, 248, 70, 69)], index * 0.02, 0, True)
        boxes, bounds = tracker.update([Rect(283, 248, 70, 75)], 0.08, 0, True)
        self.assertAlmostEqual(boxes[0, 4], -150)
        np.testing.assert_allclose(bounds[0], [65, 286])

    def test_slider_forecast_preserves_the_detected_edge_at_a_reversal(self):
        tracker = BoxTracker()
        for i, x in enumerate((280, 284, 288)):
            boxes, bounds = tracker.update([Rect(x, 160, 70, 70)], 0.02 * i, 0, True)
        x, _, width, _, speed, _ = boxes[0]
        lower, upper = bounds[0]
        self.assertEqual(speed, 178)
        for forecast in (horizontal_extent, reflected_extent):
            self.assertEqual(forecast(x, width, speed, 0.0, lower, upper), (288, 70))
            start, _ = forecast(x, width, speed, 0.1, lower, upper)
            self.assertAlmostEqual(start, 288 - 17.8)

    def test_tracks_sliding_block_before_its_top_edge_is_visible(self):
        tracker = BoxTracker()
        for index, (position, visible_height) in enumerate([(280, 13), (282, 21), (285, 27)]):
            boxes, bounds = tracker.update(
                [Rect(position, 0, 69, visible_height)], index / 60, visible_height - 13, True
            )
        self.assertEqual(boxes[0, 1], -42)
        self.assertEqual(boxes[0, 3], 69)
        self.assertAlmostEqual(boxes[0, 4], 150)
        np.testing.assert_allclose(bounds[0], [65, 287])
        boxes, _ = tracker.update([Rect(282, 0, 74, 34)], 3 / 60, 21, True)
        self.assertAlmostEqual(boxes[0, 4], -150)

    def test_slider_track_survives_a_missing_detection(self):
        tracker = BoxTracker()
        for index, position in enumerate((170, 173, 176)):
            tracker.update([Rect(position, 80, 70, 70)], index * 0.02, 0, True)
        missing, _ = tracker.update([], 0.06, 0, True)
        self.assertEqual(len(missing), 0)
        boxes, bounds = tracker.update([Rect(182, 80, 70, 70)], 0.08, 0, True)
        self.assertAlmostEqual(boxes[0, 4], 150)
        np.testing.assert_allclose(bounds[0], [65, 286])

    def test_large_square_opening_preserves_visible_edges_through_score_overlap(self):
        tracker = BoxTracker()
        initial = [Rect(74, 0, 97, 35), Rect(249, 0, 97, 35)]
        boxes, _ = tracker.update(initial, 0.01, 0, True)
        self.assertEqual(square_gaps(boxes), [(-62, 35, 171, 249)])
        tracker.update([Rect(249, 0, 97, 49)], 0.04, 14, True)
        covered = [Rect(74, 0, 103, 72), Rect(243, 0, 103, 72)]
        boxes, _ = tracker.update(covered, 0.07, 37, True)
        self.assertEqual(square_gaps(boxes), [(-25, 72, 171, 249)])

    def test_clipped_large_square_does_not_infer_vertical_motion_from_its_edge(self):
        tracker = BoxTracker()
        for i, (height, scroll) in enumerate(((30, 0), (40, 10), (45, 20))):
            boxes, _ = tracker.update([Rect(101 + 2 * i, 0, 97, height)], i / 60, scroll, True)
        self.assertAlmostEqual(boxes[0, 4], 120)
        self.assertEqual(boxes[0, 5], 0)

    def test_sliding_gap_moves_both_inner_edges_and_keeps_wall_anchors(self):
        tracker = BoxTracker()
        for index, inner in enumerate((87, 85, 83, 81)):
            shift = 6 * index
            right = inner + 170
            boxes, bounds = tracker.update(
                [Rect(1, 180 + shift, inner - 1, 38), Rect(right, 180 + shift, 419 - right, 38)],
                0.025 * index,
                shift,
                True,
            )
        np.testing.assert_allclose(boxes[:, 4], [-80, -80])
        np.testing.assert_allclose(bounds, [[45, 205], [215, 375]])
        projections = []
        for box, bound in zip(boxes, bounds):
            args = box[0], box[2], box[4], 0.5, *bound
            projected = reflected_extent(*args)
            np.testing.assert_allclose(projected, horizontal_extent(*args))
            projections.append(projected)
        self.assertAlmostEqual(projections[0][0], 1)
        self.assertAlmostEqual(sum(projections[1]), 419)
        self.assertAlmostEqual(projections[1][0] - sum(projections[0]), 170)
        self.assertAlmostEqual(sum(projections[0]), 49)

    def test_sliding_bar_can_still_measure_vertical_camera_shift(self):
        earlier = [Rect(1, 240, 109, 38), Rect(280, 240, 139, 38)]
        later = [Rect(1, 258, 102, 38), Rect(273, 258, 146, 38)]
        self.assertEqual(camera_shift(earlier, later), (18.0, True))

    def test_hammer_rate_survives_a_reversal_in_the_observation_window(self):
        samples = [
            (index * 0.017, 0, angle)
            for index, angle in enumerate(
                (0.529, 0.499, 0.512, 0.533, 0.551, 0.561, 0.581, 0.596, 0.608)
            )
        ]
        self.assertAlmostEqual(hammer_pin_rate(samples), 1.05)

    def test_fresh_immediate_jump_can_replace_a_bar_deadline_still_in_the_future(self):
        planner = FastPlanner(
            self.planner.p, route_rules=True, timed_gap_turns=True, wall_approach=True
        )
        controller = OpeningApproach(planner)
        controller.bar_deadline = ScheduledInput(0.200, 1)
        controller.pending.append(controller.bar_deadline)
        state = State(310, 680, 0, 470, 30, 30, ready=True)
        jump = controller.plan(state, 745, 0, 0.045, 0.040)
        self.assertNotEqual(jump.action, 0)
        self.assertGreaterEqual(jump.predicted_horizon, 1.5)
        self.assertFalse(controller.pending)
