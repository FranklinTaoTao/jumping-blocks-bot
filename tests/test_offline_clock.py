import unittest
from unittest.mock import patch

import numpy as np
from numba import set_num_threads

from jumpbot.planner import Decision
from tools.audit_failures import describe_loss
from tools.audit_rl_transfer import audit
from tools.offline_course_bench import simulate


class ScriptedPlanner:
    def __init__(self):
        self.direction = 1

    def plan(self, state, height, latency, interval):
        action = 0
        if state.age_since_click + 0.035 >= 0.12:
            action = self.direction
            self.direction *= -1
        return Decision(action, np.empty((0, 2)), "scripted clock regression")


class OfflineClockTests(unittest.TestCase):
    def test_aligned_delivery_keeps_absolute_window_and_rejects_expired_input(self):
        class TimedJump:
            def plan(self, state, height, latency, interval):
                action = 1 if state.age_since_click > 1 else 0
                return Decision(
                    action,
                    np.empty((0, 2)),
                    "timed jump",
                    planned_inputs=[(0.080, 1)],
                    reset_window=(0.075, 0.083),
                )

        course = {
            "number": 1,
            "initial": [209.5, 523, 0, 0],
            "height": 736,
            "score_step": 420,
            "recorded_score": 0,
            "boxes": [[1, -10000, 50, 38]],
        }
        with patch("tools.offline_course_bench.FastPlanner", return_value=TimedJump()):
            on_time = simulate(
                course, duration=0.2, align_delivery=True, timed_gap_turns=True, processing=0.038
            )
            expired = simulate(
                course, duration=0.2, align_delivery=True, timed_gap_turns=True, processing=0.049
            )
        self.assertEqual(on_time["commands"], 1)
        self.assertGreater(on_time["trace"][-1][1], 209.5)
        self.assertEqual(expired["commands"], 0)
        self.assertEqual(expired["trace"][-1][1], 209.5)

    def test_pixel_plant_and_training_plant_agree_past_clock_rounding_boundary(self):
        set_num_threads(2)
        course = {
            "number": 1,
            "initial": [209.5, 523, 0, 0],
            "height": 736,
            "score_step": 420,
            "recorded_score": 0,
            "boxes": [[1, -10000, 50, 38]],
        }
        for processing in [0.010, 0.020]:
            with self.subTest(processing=processing):
                with patch(
                    "tools.offline_course_bench.FastPlanner", return_value=ScriptedPlanner()
                ):
                    result = simulate(course, duration=8.0, processing=processing)
                trace = np.asarray(result["trace"])
                np.testing.assert_allclose(np.diff(trace[:, 0]), 0.04, atol=1e-9)
                report = audit(result, course)
                self.assertLessEqual(report["max_abs_x"], 0.051)
                self.assertLessEqual(report["max_abs_y"], 0.051)
                self.assertIsNone(report["first_divergence"])


class FailureAuditTests(unittest.TestCase):
    def test_contact_search_skips_capture_gap_followed_by_live_ascent(self):
        frames = []
        for time, y, velocity, boxes in (
            (1.0, 540, 650, []),
            (1.08, 610, 720, []),
            (1.18, 605, -260, []),
            (1.22, 585, -420, [[80, 530, 30, 40, 0, 0]]),
        ):
            frames.append(
                {
                    "t": time,
                    "sent": 0,
                    "telemetry": {"score": 213},
                    "player": {"w": 30, "h": 30},
                    "state": {
                        "x": 100,
                        "y": y,
                        "w": 30,
                        "h": 30,
                        "ready": True,
                        "vx": 150,
                        "vy": velocity,
                        "obstacles": boxes,
                    },
                }
            )
        self.assertEqual(describe_loss(frames[:2], end=1.42)["last_upright_time"], 1.0)
        report = describe_loss(frames, end=1.42)
        self.assertEqual(report["last_upright_time"], 1.22)
        self.assertEqual(
            (report["contact_candidate"], report["contact_face"]), ("block", "underside")
        )


if __name__ == "__main__":
    unittest.main()
