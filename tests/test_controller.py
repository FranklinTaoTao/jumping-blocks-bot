import unittest

import cv2
import numpy as np

from jumpbot.motion import MovingShape, Shape, ShapeTracker, collides
from jumpbot.physics import Physics, State, Tracker, calibrate, project_camera
from jumpbot.planner import Planner
from jumpbot.score import ScoreReader
from jumpbot.ui import ScreenHandler, UIAction
from jumpbot.vision import Detector, Observation, Rect, camera_shift


class VisionTests(unittest.TestCase):
    def test_recording_padding_and_header_are_excluded(self):
        frame = np.zeros((850, 500, 3), dtype=np.uint8)
        frame[20:70, 40:460] = 255  # title bar
        frame[72:820, 40:460] = 227
        frame[280:317, 40:170] = 42
        frame[280:317, 340:460] = 42
        roi = Detector.find_board(frame)
        self.assertIsNotNone(roi)
        np.testing.assert_allclose(roi, (40, 72, 420, 748), atol=3)

    def test_player_and_new_square_are_detected(self):
        frame = np.full((745, 420, 3), 227, np.uint8)
        cv2.rectangle(frame, (193, 355), (223, 385), (42, 42, 42), -1)
        cv2.rectangle(frame, (190, 357), (224, 362), (80, 80, 230), -1)
        cv2.rectangle(frame, (198, 365), (218, 373), (227, 227, 227), -1)
        cv2.rectangle(frame, (0, 225), (150, 261), (42, 42, 42), -1)
        d = Detector((0, 0, 420, 745))
        a = d.observe(frame)
        self.assertIsNotNone(a.player)
        self.assertLessEqual(a.player.y, 355)
        self.assertGreater(a.player.h, 27)
        self.assertEqual(len(a.obstacles), 1)
        cv2.rectangle(frame, (310, 450), (335, 475), (42, 42, 42), -1)
        b = d.observe(frame)
        self.assertEqual(len(b.obstacles), 2)

    def test_roi_rounding_stays_inside_native_hidpi_window(self):
        frame = np.full((1574, 836, 3), 228, np.uint8)
        frame[2:90, 2:-2] = 255
        frame[90:92] = 0
        roi = Detector.find_board(frame)
        self.assertIsNotNone(roi)
        x, y, w, h = roi
        self.assertLessEqual(x + w, frame.shape[1])
        self.assertLessEqual(y + h, frame.shape[0])
        self.assertIsNotNone(Detector().observe(frame))

    def test_board_roi_is_stable_when_bar_enters_at_top(self):
        frame = np.full((850, 500, 3), 0, np.uint8)
        frame[20:70, 40:460] = 255
        frame[72:820, 40:460] = 227
        d = Detector()
        initial = d.observe(frame).roi
        cv2.rectangle(frame, (40, 72), (330, 109), (42, 42, 42), -1)
        for _ in range(95):
            obs = d.observe(frame)
        self.assertEqual(obs.roi, initial)

    def test_bar_touching_score_is_still_detected(self):
        frame = np.full((745, 420, 3), 227, np.uint8)
        cv2.putText(frame, "8", (185, 115), cv2.FONT_HERSHEY_SIMPLEX, 2, (42, 42, 42), 5)
        cv2.rectangle(frame, (219, 80), (419, 117), (42, 42, 42), -1)
        obs = Detector((0, 0, 420, 745)).observe(frame)
        self.assertTrue(any(b.x > 180 and b.x + b.w >= 416 and b.y == 80 for b in obs.obstacles))

    def test_large_block_and_close_small_block_remain_obstacles(self):
        frame = np.full((745, 420, 3), 227, np.uint8)
        cv2.rectangle(frame, (250, 190), (370, 330), (42, 42, 42), -1)
        cv2.rectangle(frame, (193, 355), (223, 385), (42, 42, 42), -1)
        cv2.rectangle(frame, (190, 357), (224, 362), (80, 80, 230), -1)
        cv2.rectangle(frame, (198, 365), (218, 373), (227, 227, 227), -1)
        cv2.rectangle(frame, (225, 350), (245, 370), (42, 42, 42), -1)
        obs = Detector((0, 0, 420, 745)).observe(frame)
        self.assertTrue(any(b.h > 100 for b in obs.obstacles))
        self.assertTrue(any(220 < b.x < 240 and b.w < 30 for b in obs.obstacles))

    def test_yellow_outfit_is_tracked(self):
        frame = np.full((745, 420, 3), 227, np.uint8)
        cv2.rectangle(frame, (195, 350), (224, 381), (42, 42, 42), -1)
        cv2.rectangle(frame, (191, 356), (228, 370), (40, 205, 220), -1)
        obs = Detector((0, 0, 420, 745)).observe(frame)
        self.assertIsNotNone(obs.player)
        self.assertGreaterEqual(obs.player.h, 30)

    def test_nonrectangular_hammer_is_not_discarded(self):
        frame = np.full((745, 420, 3), 227, np.uint8)
        cv2.rectangle(frame, (160, 210), (240, 250), (42, 42, 42), -1)
        cv2.rectangle(frame, (190, 150), (210, 240), (42, 42, 42), -1)
        obs = Detector((0, 0, 420, 745)).observe(frame)
        self.assertEqual(len(obs.shapes), 1)

    def test_partly_visible_hammer_is_detected_at_top_edge(self):
        frame = np.full((745, 420, 3), 227, np.uint8)
        cv2.rectangle(frame, (120, -12), (200, 25), (42, 42, 42), -1)
        cv2.rectangle(frame, (150, 15), (170, 85), (42, 42, 42), -1)
        obs = Detector((0, 0, 420, 745)).observe(frame)
        self.assertEqual(len(obs.shapes), 1)
        self.assertEqual(float(obs.shapes[0].points[:, 1].min()), 0)

    def test_camera_shift_ignores_moving_square(self):
        old = [Rect(0, 200, 120, 38), Rect(290, 200, 130, 38), Rect(170, 150, 22, 22)]
        new = [Rect(0, 214, 120, 38), Rect(290, 214, 130, 38), Rect(174, 140, 22, 22)]
        shift, reliable = camera_shift(old, new)
        self.assertTrue(reliable)
        self.assertEqual(shift, 14)


class PhysicsTests(unittest.TestCase):
    def test_camera_waits_for_midpoint_and_keeps_progress_after_a_fall(self):
        y = np.array([550, 500, 400, 370, 300, 350, 400, 350, 300, 250], dtype=float)
        screen, peak = project_camera(y, anchor=370)
        np.testing.assert_allclose(screen, [550, 500, 400, 370, 370, 420, 470, 420, 370, 370])
        np.testing.assert_allclose(peak, [370, 370, 370, 370, 300, 300, 300, 300, 300, 250])

    def test_recorded_jump_rejects_pre_reset_rendered_positions(self):
        import json
        from pathlib import Path

        data = json.loads((Path(__file__).parent / "fixtures/late_jump.json").read_text())
        physics = Physics(**data["physics"])
        tracker = Tracker(physics)
        tracker.clicked(data["initial_direction"], data["initial_click_t"])
        image = np.zeros((745, 420, 3), np.uint8)
        current = None
        for row in data["rows"]:
            obs = Observation(
                image,
                Rect(**row["player"]),
                [Rect(**b) for b in row["obstacles"]],
                True,
                (0, 0, 420, 745),
            )
            state = tracker.update(obs, row["t"])
            absolute_t = row["t"] + data["time_origin"]
            if 109.20 < absolute_t < 109.22:
                current = (row, state)
                break
            if row.get("click"):
                tracker.clicked(row["click"]["direction"], row["click"]["t"])
        row, state = current
        # This future measured position is held out of the tracker fit.
        future = min(data["rows"], key=lambda a: abs(a["t"] - row["t"] - 0.1))
        dt = future["t"] - row["t"]
        measured_y = future["player"]["y"] + future["player"]["h"] / 2
        revised_y = state.y + state.vy * dt + 0.5 * physics.gravity * dt * dt
        original_y = state.y + row["recorded_vy"] * dt + 0.5 * physics.gravity * dt * dt
        self.assertLess(abs(revised_y - measured_y), 6)
        self.assertGreater(abs(original_y - measured_y), 20)

    def test_recovers_ballistics_despite_velocity_resets(self):
        samples = []
        for arc in range(10):
            for t in np.arange(0, 0.45, 1 / 30):
                samples.append(
                    (arc * 0.5 + t, 210 + (-1) ** arc * 150 * t, -arc * 160 - 700 * t + 700 * t * t)
                )
        result = calibrate(samples)["physics"]
        self.assertAlmostEqual(result["gravity"], 1400, delta=3)
        self.assertAlmostEqual(result["speed"], 150, delta=1)
        self.assertAlmostEqual(result["jump"], 700, delta=30)

    def test_scrolling_is_removed_from_vertical_velocity(self):
        tracker = Tracker(Physics())
        image = np.zeros((745, 420, 3), np.uint8)
        for i in range(5):
            obs = Observation(
                image,
                Rect(200, 350, 30, 30),
                [Rect(0, 150 + i * 10, 120, 38), Rect(290, 150 + i * 10, 130, 38)],
                True,
                (0, 0, 420, 745),
            )
            state = tracker.update(obs, i / 30)
        self.assertTrue(state.ready)
        self.assertLess(state.vy, -200)

    def test_tracking_loss_and_timestamp_gap_disarm(self):
        tracker = Tracker(Physics())
        image = np.zeros((745, 420, 3), np.uint8)
        obs = Observation(
            image, Rect(200, 350, 30, 30), [Rect(0, 150, 120, 38)], True, (0, 0, 420, 745)
        )
        for i in range(5):
            state = tracker.update(obs, i / 30)
        self.assertTrue(state.ready)
        self.assertFalse(tracker.update(obs, 2).ready)
        self.assertIsNone(tracker.update(None, 2.1))
        self.assertFalse(tracker.update(obs, 2.2).ready)

    def test_quantized_motion_does_not_disarm_known_live_click(self):
        tracker = Tracker(Physics())
        image = np.zeros((745, 420, 3), np.uint8)
        y = 100
        for i in range(16):
            t = i * 0.02
            y += 9 if i != 13 else 21
            obs = Observation(
                image,
                Rect(210 - i * 3, 350, 30, 30),
                [Rect(0, y, 120, 38), Rect(290, y, 130, 38)],
                True,
                (0, 0, 420, 745),
            )
            state = tracker.update(obs, t)
            if i == 3:
                tracker.clicked(-1, t)
            if i > 6:
                self.assertTrue(state.ready)


class PlannerTests(unittest.TestCase):
    def setUp(self):
        self.planner = Planner(Physics(), horizon=0.36)

    def test_jump_replaces_existing_vertical_velocity(self):
        s = State(210, 500, 0, 450, 30, 30, ready=True)
        d = self.planner.plan(s, 745, latency=0)
        self.assertIn(d.action, (-1, 1))
        # Downward starting velocity must not be added to the jump impulse.
        self.assertAlmostEqual(d.path[0, 1], 500 - 700 * 0.06 + 700 * 0.06**2)

    def test_rising_camera_prevents_retreat_into_lost_screen_space(self):
        # With no further clicks, a jump from y=500 peaks at y=325. The camera
        # then keeps its 45-pixel advance. At 1.25s world y=718.75 would fit in
        # the old viewport, but screen y=763.75 has already fallen offscreen.
        planner = type(self.planner)(Physics(), horizon=1.4, step=0.05, beam=30, cooldown=10)
        state = State(210, 500, 0, -700, 30, 30, age_since_click=0, ready=True)
        decision = planner.plan(state, 740, latency=0)
        self.assertGreaterEqual(decision.predicted_horizon, 1.15)
        self.assertLessEqual(decision.predicted_horizon, 1.20 + 1e-6)

    def test_does_not_jump_into_low_ceiling(self):
        boxes = np.array([[0, 437, 420, 20, 0, 0]], dtype=float)
        s = State(210, 500, 0, 0, 30, 30, ready=True, obstacles=boxes)
        self.assertEqual(self.planner.plan(s, 745, latency=0).action, 0)

    def test_blocks_outside_predicted_endpoint_still_collide(self):
        # A fast upwards jump passes entirely through this thin obstacle in one
        # planning step. Intermediate collision samples must reject both jumps.
        boxes = np.array([[0, 465, 420, 7, 0, 0]], dtype=float)
        s = State(210, 500, 0, 0, 30, 30, ready=True, obstacles=boxes)
        self.assertEqual(self.planner.plan(s, 745, latency=0).action, 0)

    def test_new_obstacle_changes_immediate_action(self):
        s = State(210, 500, 150, 100, 30, 30, ready=True)
        s.obstacles = np.array([[232, 452, 80, 40, 0, 0]], dtype=float)
        d = self.planner.plan(s, 745, latency=0)
        self.assertNotEqual(d.action, 1)

    def test_unreliable_tracking_never_requests_click(self):
        s = State(210, 500, 0, 0, 30, 30, ready=False)
        self.assertEqual(self.planner.plan(s, 745).action, 0)

    def test_cooldown_prevents_duplicate_click(self):
        s = State(210, 500, 150, -650, 30, 30, age_since_click=0.01, ready=True)
        self.assertEqual(self.planner.plan(s, 745, latency=0).action, 0)

    def test_does_not_procrastinate_first_jump_at_unknown_frontier(self):
        p = Planner(Physics())
        s = State(
            213,
            526,
            0,
            0,
            30,
            32,
            ready=True,
            obstacles=np.array([[259, 167, 160, 38, 0, 0], [1, 167, 88, 38, 0, 0]], dtype=float),
        )
        self.assertIn(p.plan(s, 745, latency=0).action, (-1, 1))

    def test_avoids_recorded_subframe_turn_failure(self):
        # The former controller jumped left here, then needed an almost
        # perfectly timed reversal to avoid the square at (160, 232).
        s = State(
            237,
            371.5,
            -150.31,
            -479.83,
            30,
            31,
            0.1329,
            True,
            np.array(
                [
                    [1, 360, 128, 38, 0, 0],
                    [298, 360, 121, 38, 0, 0],
                    [166, 650, 31, 31, 0, 0.56],
                    [140, 480, 31, 30, 0, -0.51],
                    [160, 232, 30, 30, -0.015, -8.08],
                    [152, 61, 31, 31, 0, 0],
                ],
                float,
            ),
        )
        measured = Physics(150.3102866983, 705.1227336916, 1406.7818715834)
        self.assertNotEqual(Planner(measured, click_cost=12).plan(s, 744, latency=0.04).action, -1)


class NativePlannerTests(PlannerTests):
    def setUp(self):
        from jumpbot.fast_planner import FastPlanner

        self.planner = FastPlanner(Physics(), horizon=0.36, step=0.06, beam=120)


class ScreenHandlerTests(unittest.TestCase):
    def test_requires_both_helmet_label_and_no_button(self):
        handler = ScreenHandler()
        image = np.full((790, 420), 112, np.uint8)
        label, no = handler.templates["helmet_label"], handler.templates["helmet_no"]
        image[270 : 270 + label.shape[0], 68 : 68 + label.shape[1]] = label
        frame = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        self.assertIsNone(handler.inspect(frame))
        image[541 : 541 + no.shape[0], 70 : 70 + no.shape[1]] = no
        frame = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        action = handler.inspect(frame)
        self.assertEqual(action.kind, "decline_helmet")
        self.assertLess(action.x, 210)

    def test_ui_debounces_clicks_and_rejects_moving_target(self):
        h = ScreenHandler()
        a = UIAction("restart", 300, 550, 0.99)
        self.assertFalse(h.ready(a, 1))
        self.assertFalse(h.ready(a, 1.02))
        self.assertTrue(h.ready(a, 1.04))
        h.clicked(1.04)
        self.assertFalse(h.ready(a, 1.06))
        self.assertFalse(h.ready(UIAction("restart", 320, 550, 0.99), 3))


class MotionTests(unittest.TestCase):
    def test_future_rotation_hits_but_current_shape_does_not(self):
        points = np.array([[160, 190], [260, 190], [260, 210], [160, 210]], dtype=float)
        shape = MovingShape(points, (210, 200), (0, 0), np.pi / 2, 0)
        px = np.array([[210.0, 210.0]])
        py = np.array([[245.0, 245.0]])
        self.assertTrue(collides(px, py, [shape], np.array([0.0, 1.0]), 15, 15, 0)[0][0])
        self.assertFalse(collides(px[:, :1], py[:, :1], [shape], np.array([0.0]), 15, 15, 0)[0][0])

    def test_recovers_rigid_pivot_and_angular_speed(self):
        tracker = ShapeTracker()
        for t in np.arange(0, 0.51, 0.05):
            angle = 2 * t
            center = np.array([210 + 20 * np.cos(angle), 200 + 20 * np.sin(angle)])
            points = center + np.array([[-20, -10], [20, -10], [20, 10], [-20, 10]])
            shape = Shape(points, *center, angle, 800.0)
            result = tracker.update([shape], float(t), 0)[0]
        np.testing.assert_allclose(result.pivot, (210, 200), atol=0.01)
        self.assertAlmostEqual(result.omega, 2.0, places=3)

    def test_score_reader_recognizes_three_digit_target(self):
        reader = ScoreReader()
        frame = np.full((745, 420, 3), 227, np.uint8)
        import pathlib

        with np.load(pathlib.Path(__file__).parents[1] / "jumpbot/assets/digits.npz") as glyphs:
            x = 149
            for digit in "100":
                g = glyphs[digit]
                patch = frame[60 : 60 + g.shape[0], x : x + g.shape[1]]
                patch[g > 127] = 42
                x += g.shape[1] + 8
        self.assertEqual(reader.read(frame), 100)


if __name__ == "__main__":
    unittest.main()
