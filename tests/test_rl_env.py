import unittest

import numpy as np

from jumpbot.native_search import advance_scene
from tools.rl_env import CourseEnv


class TrainingPlantTests(unittest.TestCase):
    def empty(self):
        env = CourseEnv(n=1, tiles=2)
        env.boxes[:] = 0
        env.s[0, 14] = -10000
        return env

    def test_idle_game_does_not_fall_before_first_input(self):
        env = self.empty()
        for _ in range(10):
            env.step([0], auto_reset=False)
        self.assertEqual(env.s[0, 1], 523)
        self.assertEqual(env.s[0, 3], 0)
        self.assertFalse(env.done[0])

    def test_impulse_occurs_after_processing_and_physical_input_delay(self):
        env = self.empty()
        env.params[0, 5] = 0.005
        env.step([2], auto_reset=False)
        for _ in range(6):
            env.step([0], auto_reset=False)
        self.assertEqual(env.s[0, 1], 523)  # 35 ms has elapsed, no early integration.
        env.step([0], auto_reset=False)
        self.assertAlmostEqual(env.s[0, 1], 523 - 688.55 * 0.005 + 1415 * 0.005**2 / 2)
        self.assertAlmostEqual(env.s[0, 0], 209.5 + 150.4 * 0.005)

    def test_sidewall_stops_horizontal_motion_without_killing_player(self):
        env = self.empty()
        env.s[0, :4] = 419, 500, 150.4, -400
        env.s[0, 13] = 1
        env.step([0], auto_reset=False)
        self.assertEqual(env.s[0, 0], 420)
        self.assertEqual(env.s[0, 2], 0)
        self.assertLess(env.s[0, 1], 500)
        self.assertFalse(env.done[0])
        self.assertLess(env.rewards[0], (523 - env.s[0, 4]) / 420)

    def test_camera_easing_matches_reference_and_obstacles_stay_hidden(self):
        env = self.empty()
        env.params[0, 5] = 0.005
        env.s[0, :6] = 210, 300, 0, -600, 300, 65
        env.s[0, 13] = 1
        env.boxes[0, 0] = 1, -1000, 200, 38
        env.step([0], auto_reset=False)
        before = 368 - 300
        after = 368 - env.s[0, 4]
        self.assertAlmostEqual(env.s[0, 5], advance_scene(65, before, after, 0.005, 0.011))
        self.assertEqual(np.count_nonzero(env.obs[0, 8:].reshape(12, 7)[:, -1]), 0)

    def test_thin_bar_collision_is_checked_between_control_frames(self):
        env = self.empty()
        env.s[0, :4] = 210, 280, 0, -688.55
        env.s[0, 13] = 1
        # Camera inactive until the existing minimum is crossed.
        env.s[0, 4] = 200
        env.s[0, 5] = 168
        env.boxes[0, 0] = 180, 248, 60, 3
        env.step([0], auto_reset=False)
        self.assertTrue(env.done[0])
        self.assertEqual(env.reasons[0], 1)
        terminal = env.s[0, :13].copy()
        for _ in range(5):
            env.step([2], auto_reset=False)
        np.testing.assert_array_equal(env.s[0, :13], terminal)
        self.assertEqual(len(env.completed), 1)

    def test_counterfactual_labels_do_not_modify_the_training_plant(self):
        env = self.empty()
        env.s[0, :4] = 210, 400, 150.4, 0
        env.s[0, 13] = 1
        env.boxes[0, 0] = 216, 300, 50, 30
        before = env.s.copy()
        history = env.history.copy()
        risk = env.action_risks()
        np.testing.assert_array_equal(env.s, before)
        np.testing.assert_array_equal(env.history, history)
        self.assertEqual(risk.tolist(), [[0, 0, 1]])

    def test_wait_label_stops_when_a_following_input_can_take_effect(self):
        env = self.empty()
        env.s[0, :4] = 210, 400, 0, 0
        env.s[0, 13] = 1
        env.boxes[0, 0] = 180, 440, 60, 30
        self.assertEqual(env.action_risks()[0, 0], 0)
        # A long unconditional coast does hit this bar, but waiting one frame
        # leaves time to reset upward. That future choice must remain available.
        env.params[0, 5] = 0.3
        env.step([0], auto_reset=False)
        self.assertTrue(env.done[0])

    def test_last_direction_remains_observable_after_pending_input_expires(self):
        env = self.empty()
        env.step([2])
        for _ in range(5):
            env.step([0])
        self.assertEqual(env.obs[0, 5], 1)
        self.assertEqual(env.obs[0, 6], 0)

    def test_large_obstacles_remove_overlapping_smalls_from_both_neighboring_tiles(self):
        env = CourseEnv(n=1, seed=817, stage=2, big_start=5)
        large = [b for b in env.boxes[0] if b[2] == 97]
        small = [b for b in env.boxes[0] if b[2] == 30]
        self.assertTrue(large)
        self.assertTrue(all(abs(s[1] - b[1]) > 150 for s in small for b in large))

    def test_geometry_stage_keeps_physics_fixed_before_domain_randomization(self):
        env = CourseEnv(n=4, stage=2)
        for row in env.params[1:]:
            np.testing.assert_array_equal(row, env.params[0])
        noisy = CourseEnv(n=4, stage=3)
        self.assertGreater(np.ptp(noisy.params[:, 0]), 0)

    def test_tracker_surrogate_matches_real_pixel_globals_across_delayed_jumps(self):
        env = CourseEnv(n=1, stage=0, seed=817, observation_version=3, pixel_envs=1)
        differences = []
        direction = 1
        for _ in range(160):
            if env.obs[0, 4] > 0.05:
                differences.append(env.obs[0, [0, 1, 2, 3, 7]] - env.ideal_obs[0, [0, 1, 2, 3, 7]])
            action = 0
            if env.obs[0, 4] >= 0.999:
                action = direction
                direction = 3 - direction
            env.step([action])
        self.assertGreater(len(differences), 80)
        errors = abs(np.asarray(differences)) * [210, 368, 150, 700, 700]
        np.testing.assert_array_less(np.percentile(errors, 95, axis=0), [0.1, 0.1, 0.1, 0.1, 1.0])


if __name__ == "__main__":
    unittest.main()
