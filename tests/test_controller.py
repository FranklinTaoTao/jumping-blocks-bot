import unittest

import cv2
import numpy as np

from jumpbot.motion import MovingShape, Shape, ShapeTracker, collides
from jumpbot.navigation import Navigation
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

    def test_window_scaling_preserves_camera_corrected_ballistic_forecast(self):
        physics = Physics(145, 680, 1396)
        forecasts = []
        for scale in (0.75, 1.0, 1.5, 2.0):
            width, height = round(420 * scale), round(744 * scale)
            detector = Detector((0, 0, width, height))
            tracker = Tracker(physics)
            tracker.clicked(1, 0)
            for t in np.arange(0.03, 0.301, 0.03):
                x = 170 + physics.speed * t
                y = 540 - physics.jump * t + physics.gravity * t * t / 2
                frame = np.full((744, 420, 3), 227, np.uint8)
                frame[150:188, :120] = 42
                frame[150:188, 290:] = 42
                cx, cy = round(x), round(y)
                frame[cy - 15 : cy + 16, cx - 15 : cx + 15] = 42
                frame[cy - 12 : cy - 6, cx - 18 : cx + 19] = (40, 205, 220)
                observation = detector.observe(cv2.resize(frame, (width, height)))
                state = tracker.update(observation, float(t))
            self.assertTrue(state.ready)
            forecasts.append(state.y + state.vy * 0.1 + physics.gravity * 0.1**2 / 2)
        self.assertLess(np.ptp(forecasts), 2.5)

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

    def test_score_badge_does_not_widen_a_narrow_incoming_bar(self):
        detector = Detector((0, 0, 420, 745))
        for top in (0, 15, 30, 45, 60, 80, 100):
            frame = np.full((745, 420, 3), 227, np.uint8)
            frame[top : top + 38, :58] = 42
            frame[top : top + 38, 228:] = 42
            # A colored score badge occludes the narrow bar completely mid-pass.
            frame[40:80, 12:130] = (140, 170, 190)
            observation = detector.observe(frame)
            left = [b for b in observation.obstacles if b.x < 4]
            self.assertEqual(len(left), 1)
            self.assertEqual(left[0].w, 57, (top, left))

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


class WallTests(unittest.TestCase):
    def test_clipped_player_head_is_not_detected_as_an_edge_obstacle(self):
        frame = np.full((744, 420, 3), 227, np.uint8)
        frame[150:188, :120] = 42
        frame[500:531, :15] = 42
        frame[503:509, :19] = (40, 205, 220)
        observation = Detector((0, 0, 420, 744)).observe(frame)
        self.assertIsNotNone(observation.player)
        self.assertFalse(any(b.y > 490 for b in observation.obstacles))
        self.assertTrue(any(b.y == 150 for b in observation.obstacles))

    def test_joint_calibration_recovers_changed_spatial_gravity(self):
        physics = Physics(145, 680, 1400)
        tracker = Tracker(physics, adaptive=True, adaptive_gravity=True)
        image = np.zeros((744, 420, 3), np.uint8)
        for arc in range(24):
            start = arc * 0.5
            direction = 1 if arc % 2 == 0 else -1
            tracker.clicked(direction, start)
            for age in np.arange(0.03, 0.421, 0.03):
                tau = age - 0.012
                x = 210 + direction * 145 * tau
                y = 540 - 680 * tau + 1320 * tau * tau / 2
                observation = Observation(
                    image,
                    Rect(x - 15, y - 15, 30, 30),
                    [Rect(0, 150, 120, 38)],
                    True,
                    (0, 0, 420, 744),
                )
                state = tracker.update(observation, float(start + age))
        self.assertAlmostEqual(state.gravity_gain, 1320 / 1400, delta=0.01)
        self.assertAlmostEqual(state.jump_gain, 1, delta=0.02)

    def test_clipped_body_at_wall_keeps_full_width_and_vertical_tracking(self):
        physics = Physics(145, 680, 1396)
        image = np.zeros((744, 420, 3), np.uint8)
        for adaptive in (False, True):
            with self.subTest(adaptive=adaptive):
                tracker = Tracker(physics, adaptive=adaptive)
                tracker.clicked(-1, 0)
                for t in np.arange(0.03, 0.301, 0.03):
                    y = 540 - 680 * (t - 0.012) + 1396 * (t - 0.012) ** 2 / 2
                    observation = Observation(
                        image,
                        Rect(0, y - 15, 15, 30),
                        [Rect(0, 150, 120, 38)],
                        True,
                        (0, 0, 420, 744),
                    )
                    state = tracker.update(observation, float(t))
                self.assertTrue(state.ready)
                self.assertAlmostEqual(state.x, 0)
                self.assertAlmostEqual(state.w, 30)
                self.assertAlmostEqual(state.vx, 0)
                self.assertGreater(state.wall_dwell, 0.25)
                self.assertAlmostEqual(state.vy, -680 + 1396 * (t - 0.012), delta=2)

    def test_wall_is_a_place_to_wait_beneath_a_bar(self):
        from jumpbot.fast_planner import FastPlanner

        physics = Physics()
        state = State(3, 450, -150, -250, 30, 30, age_since_click=0.2, ready=True)
        state.obstacles = np.array([[0, 300, 180, 38, 0, 0]], dtype=float)
        for planner in (Planner(physics), FastPlanner(physics)):
            decision = planner.plan(state, 744)
            self.assertEqual(decision.action, 0)
            self.assertGreater(decision.predicted_horizon, 1)
            self.assertAlmostEqual(decision.path[:, 0].min(), 0)
            state.wall_dwell = 20
            dwelling = planner.plan(state, 744)
            self.assertEqual(dwelling.action, 0)
            self.assertGreater(dwelling.predicted_horizon, 1)
            self.assertLess(dwelling.score, decision.score)
            state.wall_dwell = 0


class MovingBlockTests(unittest.TestCase):
    def test_large_stationary_block_does_not_acquire_camera_velocity(self):
        from jumpbot.motion import BoxTracker

        tracker = BoxTracker()
        for i in range(8):
            boxes, bounds = tracker.update(
                [Rect(140, 200 + i * 12, 97, 97)], i * 0.03, i * 12, True
            )
        np.testing.assert_allclose(boxes[0, 4:], 0, atol=1e-9)
        np.testing.assert_allclose(bounds, 0)


class NavigationTests(unittest.TestCase):
    def test_guidance_recognizes_a_bar_pocket_before_entering_it(self):
        # The current player is left of the bar. Future positions beneath it
        # must still account for the upward impulse required by a reversal.
        state = State(150, 370, 150, -100, 30, 30, ready=True)
        state.obstacles = np.array([[222, 250, 197, 38, 0, 0]], dtype=float)
        plain = Navigation(state)
        guided = Navigation(state, jump_clearance=165)
        x = np.array([270.0, 270.0, 150.0])
        y = np.array([370.0, 510.0, 370.0])
        extra = guided.cost(x, y) - plain.cost(x, y)
        self.assertGreater(extra[0], 0)
        self.assertEqual(extra[1], 0)
        self.assertEqual(extra[2], 0)

    def test_compiled_grid_preserves_visibility_costs(self):
        state = State(265, 371, 146, -236, 30, 32, ready=True)
        state.obstacles = np.array(
            [
                [1, 21, 134, 38, 0, 0],
                [234, 21, 185, 38, 0, 0],
                [1, 441, 97, 38, 0, 0],
                [268, 441, 151, 38, 0, 0],
                [144, 202, 97, 96, 0, 0],
            ],
            dtype=float,
        )
        navigation = Navigation(state)
        yy, xx = np.mgrid[0:751:10, 0:421:10]
        points = np.stack([xx.ravel(), yy.ravel()], axis=1)
        lengths = np.linalg.norm(points[:, None] - navigation.nodes[None], axis=2)
        lengths += navigation.distance[None]
        lengths[~navigation.visible(points)] = np.inf
        best = lengths.min(1)
        expected = np.where(np.isfinite(best), best, 1000 + abs(points[:, 1] - navigation.frontier))
        np.testing.assert_allclose(navigation.cost(points[:, 0], points[:, 1]), expected)


class PhysicsTests(unittest.TestCase):
    def test_repeated_presentation_leaves_velocity_fit_and_input_unarmed(self):
        physics = Physics(150, 690, 1415)
        tracker = Tracker(physics, reset_phase=0, skip_repeated_frames=True)
        tracker.clicked(1, 0)
        image = np.zeros((744, 420, 3), np.uint8)
        observations = []
        for t in (0.055, 0.075, 0.105, 0.135, 0.202):
            player = Rect(
                185 + physics.speed * t, 585 - physics.jump * t + physics.gravity * t**2 / 2, 30, 30
            )
            observations.append(
                Observation(image, player, [Rect(1, 150, 120, 38)], True, (0, 0, 420, 744))
            )
        for obs, t in zip(observations, (0.055, 0.075, 0.105, 0.135)):
            tracker.update(obs, t)
        self.assertIsNone(tracker.update(observations[3], 0.185))
        self.assertEqual(tracker.previous_t, 0.135)
        next_state = tracker.update(observations[4], 0.202)
        self.assertTrue(next_state.ready)
        self.assertAlmostEqual(next_state.vy, -physics.jump + physics.gravity * 0.202)

    def test_spatial_velocity_uses_position_when_the_last_frame_repeats(self):
        physics = Physics(150, 690, 1415)
        image = np.zeros((744, 420, 3), np.uint8)
        trackers = [
            Tracker(physics, reset_phase=0, spatial_velocity=enabled) for enabled in (False, True)
        ]
        for tracker in trackers:
            tracker.clicked(1, 0)
        for capture, elapsed in zip(
            (0.055, 0.075, 0.105, 0.135, 0.165), (0.055, 0.075, 0.105, 0.135, 0.135)
        ):
            x = 200 + physics.speed * elapsed
            y = 600 - physics.jump * elapsed + physics.gravity * elapsed**2 / 2
            observation = Observation(
                image, Rect(x - 15, y - 15, 30, 30), [Rect(1, 150, 120, 38)], True, (0, 0, 420, 744)
            )
            baseline, spatial = [tracker.update(observation, capture) for tracker in trackers]
        velocity = -physics.jump + physics.gravity * elapsed
        self.assertGreater(abs(velocity - baseline.vy), 25)
        self.assertAlmostEqual(spatial.vy, velocity)

    def test_timestamped_future_impulse_does_not_apply_to_pre_event_pixels(self):
        physics = Physics(150, 680, 1400)
        tracker = Tracker(physics, reset_phase=0)
        tracker.clicked(1, 0.035)
        image = np.zeros((744, 420, 3), np.uint8)
        for t in np.arange(0, 0.171, 0.017):
            tau = max(0, t - 0.035)
            x = 210 + 150 * tau
            y = 540 - 680 * tau + 700 * tau * tau
            observation = Observation(
                image, Rect(x - 15, y - 15, 30, 30), [Rect(0, 150, 120, 38)], True, (0, 0, 420, 744)
            )
            state = tracker.update(observation, float(t))
            if t < 0.035:
                self.assertFalse(state.ready)
        self.assertTrue(state.ready)
        self.assertAlmostEqual(state.vy, -680 + 1400 * (t - 0.035), delta=2)

    def test_effective_reset_strength_learns_from_observed_arcs(self):
        physics = Physics(144.5, 680, 1396.5)
        tracker = Tracker(physics, adaptive=True)
        image = np.zeros((744, 420, 3), np.uint8)
        actual_jump = 730
        for arc in range(8):
            start = arc * 0.5
            tracker.update(None, start - 0.01)
            tracker.clicked(1, start)
            for age in np.arange(0.03, 0.301, 0.03):
                tau = age - 0.012
                x = 210 + physics.speed * tau
                y = 540 - actual_jump * tau + physics.gravity * tau**2 / 2
                obs = Observation(
                    image,
                    Rect(x - 15, y - 16, 30, 32),
                    [Rect(0, 150, 120, 38), Rect(290, 150, 130, 38)],
                    True,
                    (0, 0, 420, 744),
                )
                state = tracker.update(obs, float(start + age))
        self.assertAlmostEqual(state.jump_gain, actual_jump / physics.jump, delta=0.01)

    def test_spatial_estimator_tracks_a_slower_game_clock(self):
        physics = Physics(144.5, 680, 1396.5)
        tracker = Tracker(physics, adaptive=True)
        tracker.clicked(1, 0)
        image = np.zeros((744, 420, 3), np.uint8)
        rate = 0.7
        for t in np.arange(0.03, 0.391, 0.03):
            tau = rate * (t - 0.012)
            x = 210 + physics.speed * tau
            y = 540 - physics.jump * tau + physics.gravity * tau**2 / 2
            obs = Observation(
                image,
                Rect(x - 15, y - 16, 30, 32),
                [Rect(0, 150, 120, 38), Rect(290, 150, 130, 38)],
                True,
                (0, 0, 420, 744),
            )
            state = tracker.update(obs, float(t))
        self.assertTrue(state.ready)
        self.assertAlmostEqual(state.motion_rate, rate, delta=0.02)
        expected_vy = rate * (-physics.jump + physics.gravity * tau)
        self.assertAlmostEqual(state.vy, expected_vy, delta=8)

    def test_pixels_from_before_pending_click_do_not_arm_input(self):
        tracker = Tracker(Physics())
        image = np.zeros((745, 420, 3), np.uint8)
        for i in range(9):
            t = i * 0.02
            obs = Observation(
                image,
                Rect(200 + i * 3, 500, 30, 30),
                [Rect(0, 200, 120, 38), Rect(290, 200, 130, 38)],
                True,
                (0, 0, 420, 745),
            )
            if i == 3:
                # The screen is 70 ms behind the input event clock.
                tracker.clicked(1, t + 0.070)
            state = tracker.update(obs, t)
            if 3 <= i <= 7:
                self.assertFalse(state.ready)
            if i == 8:
                self.assertTrue(state.ready)

    def test_camera_waits_for_midpoint_and_keeps_progress_after_a_fall(self):
        y = np.array([550, 500, 400, 370, 300, 350, 400, 350, 300, 250], dtype=float)
        screen, peak = project_camera(y, anchor=370)
        np.testing.assert_allclose(screen, [550, 500, 400, 370, 370, 420, 470, 420, 370, 370])
        np.testing.assert_allclose(peak, [370, 370, 370, 370, 300, 300, 300, 300, 300, 250])

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
    def test_turns_before_drift_can_carry_player_out_of_upcoming_gap(self):
        from jumpbot.fast_planner import FastPlanner

        state = State(
            199.5,
            366.5,
            150.4,
            -509.1231,
            31,
            31,
            age_since_click=0.1479978,
            ready=True,
            camera_speed=540.9904,
            obstacles=np.array(
                [
                    [1, 0, 64, 33, 0, 0],
                    [1, 414, 53, 38, 0, 0],
                    [234, 0, 185, 33, 0, 0],
                    [224, 414, 195, 38, 0, 0],
                    [259, 595, 96, 96, 0, 0],
                    [65, 595, 96, 96, 0, 0],
                    [253, 175, 96, 97, 0, 0],
                    [71, 175, 96, 97, 0, 0],
                ],
                dtype=float,
            ),
        )
        planner = FastPlanner(
            Physics(150.4, 688.55, 1415),
            step=0.025,
            beam=600,
            cooldown=0.18,
            camera_tau=0.011,
            pocket_penalty=1,
        )
        latency = 0.08137261
        original = planner.plan(state, 733, latency, 0.04)
        self.assertEqual(original.action, 0)
        self.assertLess(original.predicted_horizon, 1)
        planner.narrow_gap_cooldown = 0.12
        reversal = planner.plan(state, 733, latency, 0.04)
        self.assertEqual(reversal.action, -1)
        self.assertGreaterEqual(reversal.predicted_horizon, 1.5)

    def test_waits_for_inward_jump_when_wallward_route_misses_visible_opening(self):
        from jumpbot.fast_planner import FastPlanner

        # Recorded approach to the right wall with the opening at x=99..269.
        state = State(
            342.5,
            366.5,
            154.84257589,
            -520.86843449,
            31,
            31,
            age_since_click=0.0976382,
            ready=True,
            camera_speed=595.29246952,
            motion_rate=1.029538403546919,
            jump_gain=0.9926523496194892,
            obstacles=np.array(
                [
                    [1, 98, 98, 37, 0, 0],
                    [1, 517, 169, 38, 0, 0],
                    [269, 98, 150, 37, 0, 0],
                    [340, 517, 79, 38, 0, 0],
                    [167, 698, 96, 35, 0, 0],
                    [248, 389, 30, 30, 0, 0],
                    [224, 218, 31, 30, 0, 0],
                ],
                dtype=float,
            ),
        )
        physics = Physics(150.4, 688.55, 1415)
        settings = dict(
            step=0.025,
            beam=600,
            cooldown=0.18,
            camera_tau=0.011,
            pocket_penalty=1,
            narrow_gap_cooldown=0.12,
        )
        args = state, 733, 0.0859683, 0.04
        old = FastPlanner(physics, **settings).plan(*args)
        self.assertEqual(old.action, 1)
        self.assertGreater(old.path[:, 1].min(), 98)
        staged = FastPlanner(physics, **settings, wall_approach=True).plan(*args)
        self.assertEqual(staged.action, 0)
        self.assertEqual(staged.planned_inputs[0][1], -1)
        self.assertLess(staged.path[:, 1].min(), 98 - state.h / 2 - 4)

    def test_reverses_toward_upcoming_square_gap_after_clearing_a_bar(self):
        from jumpbot.fast_planner import FastPlanner

        # At score 87, the upper pair was already visible when the player
        # drifted past the left edge of the opening.
        state = State(
            183.5,
            366.5,
            -150.4,
            -622.0208,
            31,
            31,
            age_since_click=0.0470171,
            ready=True,
            camera_speed=558.4966,
            obstacles=np.array(
                [
                    [1, 290, 58, 39, 0, 0],
                    [1, 710, 106, 21, 0, 0],
                    [229, 290, 190, 39, 0, 0],
                    [277, 710, 142, 23, 0, 0],
                    [253, 471, 96, 97, 0, 0],
                    [71, 471, 96, 97, 0, 0],
                    [250, 51, 97, 97, 0, 0],
                    [73, 51, 97, 97, 0, 0],
                ],
                dtype=float,
            ),
        )
        planner = FastPlanner(
            Physics(150.4, 688.55, 1415),
            step=0.025,
            beam=600,
            cooldown=0.18,
            camera_tau=0.011,
            pocket_penalty=1,
        )
        latency = 0.073894
        self.assertEqual(planner.plan(state, 733, latency, 0.04).action, 0)
        planner.narrow_gap_cooldown = 0.12
        self.assertEqual(planner.gap_cooldown(state), 0.18)
        reversal = planner.plan(state, 733, latency, 0.04)
        self.assertEqual(reversal.action, 1)
        self.assertGreaterEqual(reversal.predicted_horizon, 1.5)
        self.assertTrue(reversal.reason.startswith("upper gap reversal"))

    def test_reverses_toward_upper_opening_after_entering_square_gap(self):
        from jumpbot.fast_planner import FastPlanner

        # At score 99, the rightward arc through the squares was about to
        # carry the player past the right edge of the next opening.
        state = State(
            207,
            366.5,
            150.4,
            -621.0098,
            30,
            31,
            age_since_click=0.0477316,
            ready=True,
            camera_speed=611.3528,
            obstacles=np.array(
                [
                    [1, 161, 64, 38, 0, 0],
                    [1, 581, 166, 37, 0, 0],
                    [235, 161, 184, 38, 0, 0],
                    [337, 581, 82, 37, 0, 0],
                    [200, 722, 21, 11, 0, 0],
                    [252, 342, 96, 96, 0, 0],
                    [72, 342, 96, 96, 0, 0],
                ],
                dtype=float,
            ),
        )
        planner = FastPlanner(
            Physics(150.4, 688.55, 1415),
            step=0.025,
            beam=600,
            cooldown=0.18,
            camera_tau=0.011,
            pocket_penalty=1,
        )
        latency = 0.074
        original = planner.plan(state, 733, latency, 0.04)
        self.assertEqual(original.action, 0)
        self.assertLess(original.predicted_horizon, 1)
        planner.narrow_gap_cooldown = 0.12
        reversal = planner.plan(state, 733, latency, 0.04)
        self.assertEqual(reversal.action, -1)
        self.assertGreaterEqual(reversal.predicted_horizon, 1.5)
        self.assertTrue(reversal.reason.startswith("upper gap reversal"))

    def test_local_gap_cooldown_preserves_an_earlier_reversal(self):
        from jumpbot.fast_planner import FastPlanner

        # The last viable observation before contact with the left square.
        state = State(
            206.5,
            366.5,
            -150.4,
            -621.4317385842558,
            31,
            31,
            age_since_click=0.047433400293812156,
            ready=True,
            camera_speed=600.1508876998882,
            obstacles=np.array([[74, 307, 97, 97, 0, 0], [249, 307, 97, 97, 0, 0]], dtype=float),
        )
        physics = Physics(150.4, 688.55, 1415)
        fixed = FastPlanner(physics, step=0.025, beam=600, cooldown=0.18, camera_tau=0.011)
        local = FastPlanner(
            physics, step=0.025, beam=600, cooldown=0.18, camera_tau=0.011, narrow_gap_cooldown=0.12
        )
        latency = 0.07805430987197906
        self.assertEqual(fixed.plan(state, 733, latency, 0.04).predicted_horizon, 0)
        turn = local.plan(state, 733, latency, 0.04)
        self.assertEqual(turn.action, 1)
        self.assertGreater(turn.predicted_horizon, 1)
        state.x = 390
        self.assertEqual(local.gap_cooldown(state), 0.18)

    def test_reset_timing_bound_covers_the_apex_and_both_reset_phases(self):
        from jumpbot.native_search import reset_position_error

        gravity, jump, reset, timing = 1400.0, 700.0, 0.08, 0.025
        for initial_velocity in [-450.0, 500.0]:
            for after in [0.03, 0.5, 0.7]:
                future = reset + after

                def position(at):
                    return (
                        initial_velocity * at
                        + 0.5 * gravity * at**2
                        - jump * (future - at)
                        + 0.5 * gravity * (future - at) ** 2
                    )

                nominal = position(reset)
                bound = reset_position_error(
                    initial_velocity + gravity * reset, -jump + gravity * after, gravity, timing
                )
                for delta in np.linspace(-timing, timing, 31):
                    self.assertLessEqual(abs(position(reset + delta) - nominal), bound + 1e-9)
        # Zero velocity at the apex does not imply zero position uncertainty.
        self.assertGreater(reset_position_error(600.0, 0.0, gravity, timing), 15.0)

    def test_recorded_input_schedule_reconstructs_the_selected_path(self):
        from jumpbot.fast_planner import FastPlanner

        physics = Physics()
        planner = FastPlanner(physics, horizon=0.6, step=0.05, beam=120)
        decision = planner.plan(State(210, 500, 0, 0, 30, 30, ready=True), 745, latency=0.0)
        inputs = {round(t / 0.05): direction for t, direction in decision.planned_inputs}
        x, y, vx, vy = 210.0, 500.0, 0.0, 0.0
        expected = []
        for i in range(len(decision.path)):
            if i in inputs:
                vx, vy = inputs[i] * physics.speed, -physics.jump
            x = np.clip(x + vx * 0.05, 0, 420)
            y += vy * 0.05 + 0.5 * physics.gravity * 0.05**2
            vy += physics.gravity * 0.05
            expected.append((x, y))
        np.testing.assert_allclose(decision.path, expected, atol=1e-8)

    def test_apex_error_floor_prevents_a_bar_hit_with_a_stronger_real_reset(self):
        from jumpbot.fast_planner import FastPlanner

        physics = Physics(150.4, 688.55, 1415.0)
        boxes = np.array([[225, 313, 194, 38, 0, 0]], float)

        def rollout(error_floor):
            planner = FastPlanner(
                physics, step=0.025, beam=300, cooldown=0.18, vertical_error_floor=error_floor
            )
            x, y, vx, vy, t, last, next_control = 420.0, 540.0, 0.0, 0.0, 0.0, -10.0, 0.0
            while t < 0.9:
                if t + 1e-9 >= next_control:
                    state = State(x, y, vx, vy, 31, 30, t - last, True, boxes)
                    decision = planner.plan(state, 736, latency=0.0, control_interval=0.04)
                    if decision.action:
                        vx, vy = decision.action * physics.speed, -705.0
                        last = t
                    next_control = t + 0.04
                x = float(np.clip(x + vx * 0.005, 0, 420))
                y += vy * 0.005 + 0.5 * physics.gravity * 0.005**2
                vy += physics.gravity * 0.005
                if x <= 0 or x >= 420:
                    vx = 0.0
                if x + 15.5 > 225 and x - 15.5 < 419 and y + 15 > 313 and y - 15 < 351:
                    return True
                t += 0.005
            return False

        self.assertTrue(rollout(0.0))
        self.assertFalse(rollout(14.0))

    def setUp(self):
        from jumpbot.fast_planner import FastPlanner

        self.planner = FastPlanner(Physics(), horizon=0.36, step=0.06, beam=120)

    def test_warmup_compiles_the_live_argument_types(self):
        from jumpbot.native_search import search

        self.planner.warmup()
        signatures = list(search.signatures)
        state = State(210.0, 500.0, 150.0, -600.0, 30, 32, ready=True)
        state.obstacles = np.array([[0, 200, 120, 38, 0, 0]], dtype=float)
        self.planner.plan(state, 744, latency=0.043)
        self.assertEqual(search.signatures, signatures)

    def test_camera_accelerates_toward_a_moving_target(self):
        from jumpbot.native_search import advance_scene

        tau = 0.01
        scene = 0.0
        for i in range(10):
            scene = advance_scene(scene, i, i + 1, 0.005, tau)
        expected = 200 * (0.05 - tau * (1 - np.exp(-0.05 / tau)))
        self.assertAlmostEqual(scene, expected)
        self.assertLess(scene, 10)


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
    def test_hammer_preserves_observed_outline_at_a_noisy_swing_limit(self):
        limits = (np.pi / 6, 5 * np.pi / 6)
        pivot = np.array([210, 100])
        points = pivot + np.array([[90, 40], [120, 40], [120, 65], [90, 65]])
        for start, rate, rotation in (
            (limits[0] - 0.075, -1.0, 0.1),
            (limits[1] + 0.075, 1.0, -0.1),
        ):
            with self.subTest(start=start):
                shape = MovingShape(points, tuple(pivot), (0, 0), rate, 2, limits, start)
                observed, forecast = shape.predict([0.0, 0.1])
                np.testing.assert_allclose(observed, points)
                c, s = np.cos(rotation), np.sin(rotation)
                expected = (points - pivot) @ np.array([[c, s], [-s, c]]) + pivot
                np.testing.assert_allclose(forecast, expected)

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

    def test_close_presentation_timestamps_preserve_the_game_step(self):
        physics = Physics()
        tracker = Tracker(physics, reset_phase=0, recover_frame_clock=True)
        tracker.clicked(1, 0.0)
        bar = Rect(1, 240, 95, 38)
        for index, captured in enumerate((0.053, 0.069, 0.086, 0.103, 0.119, 0.126)):
            elapsed = (3 + index) / 60
            x = 200 + physics.speed * elapsed
            y = 550 - physics.jump * elapsed + 0.5 * physics.gravity * elapsed**2
            observation = Observation(
                None, Rect(x - 15, y - 15, 30, 30), [bar], True, (0, 0, 420, 733)
            )
            state = tracker.update(observation, captured)
        self.assertTrue(state.ready)
        self.assertAlmostEqual(state.vy, -physics.jump + physics.gravity * elapsed, places=8)

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

    def test_score_reader_recognizes_windows_one_without_a_base(self):
        from pathlib import Path

        reader = ScoreReader()
        frame = np.full((745, 420, 3), 227, np.uint8)
        with (
            np.load(Path(__file__).parents[1] / "jumpbot/assets/digits.npz") as original,
            np.load(Path(__file__).parents[1] / "jumpbot/assets/digits_windows.npz") as windows,
        ):
            x = 165
            for glyph in (original["2"], windows["1"]):
                patch = frame[60 : 60 + glyph.shape[0], x : x + glyph.shape[1]]
                patch[glyph > 127] = 42
                x += glyph.shape[1] + 8
        self.assertEqual(reader.read(frame), 21)


if __name__ == "__main__":
    unittest.main()
