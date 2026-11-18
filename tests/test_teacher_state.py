import unittest

import numpy as np

from jumpbot.fast_planner import FastPlanner
from jumpbot.physics import Physics, State
from tools.distill_planner import teacher_state
from tools.rl_env import CourseEnv
from tools.rl_policy import encode_state


class TeacherEncodingTests(unittest.TestCase):
    def test_root_quality_is_read_only_telemetry_for_each_first_action(self):
        planner = FastPlanner(Physics(150.4, 688.55, 1415), step=0.025, beam=120, camera_tau=0.011)
        state = State(
            210,
            500,
            150.4,
            -100,
            30,
            30,
            ready=True,
            obstacles=np.array([[260, 350, 80, 38, 0, 0]], float),
        )
        baseline = planner.plan(state, 736, 0.055, 0.04)
        quality = np.empty((3, 2))
        recorded = planner.plan(state, 736, 0.055, 0.04, root_quality=quality)
        self.assertEqual(recorded.action, baseline.action)
        self.assertEqual(recorded.score, baseline.score)
        np.testing.assert_array_equal(recorded.path, baseline.path)
        group = recorded.action + 1
        self.assertAlmostEqual(quality[group, 0] * planner.step, recorded.predicted_horizon)
        self.assertAlmostEqual(quality[group, 1], recorded.score)

    def test_teacher_receives_the_same_observed_geometry_as_the_classifier(self):
        env = CourseEnv(
            n=1, seed=817, stage=2, tiles=12, big_start=5, observation_version=3, processing=0.020
        )
        for i in range(24):
            state = teacher_state(env)
            encoded = encode_state(state, 736, env.s[0, 8], observation_version=3)
            if env.obs[0, 4] > 0:
                np.testing.assert_allclose(encoded[:8], env.obs[0, :8], atol=1e-6)
            source = env.obs[0, 8:].reshape(12, 7)
            decoded = encoded[8:].reshape(12, 7)
            source = source[source[:, -1] > 0]
            decoded = decoded[decoded[:, -1] > 0]
            self.assertEqual(len(source), len(decoded))
            for box in source:
                self.assertLess(float(abs(decoded - box).max(axis=1).min()), 1e-6)
            env.step([1 + (i // 4) % 2 if env.obs[0, 4] >= 0.999 else 0])
