import unittest

import numpy as np

from tools.rl_policy import NumpyPolicy


class ActionMaskTests(unittest.TestCase):
    def test_unready_idle_state_has_wait_available_and_ready_idle_state_starts(self):
        policy = NumpyPolicy.__new__(NumpyPolicy)
        policy.observation_version = 3
        policy.weights = {}
        for name, inputs, outputs in [
            ("objects.0", 7, 32),
            ("objects.2", 32, 32),
            ("body.0", 72, 128),
            ("body.2", 128, 128),
            ("actor", 128, 3),
        ]:
            policy.weights[name + ".weight"] = np.zeros((outputs, inputs), np.float32)
            policy.weights[name + ".bias"] = np.zeros(outputs, np.float32)
        obs = np.zeros((2, 92), np.float32)
        obs[1, 4] = 1
        logits = policy.logits(obs)
        self.assertEqual(logits[0, 0], 0)
        self.assertTrue(np.all(logits[0, 1:] < -1e8))
        self.assertLess(logits[1, 0], -1e8)
        self.assertTrue(np.all(logits[1, 1:] == 0))
