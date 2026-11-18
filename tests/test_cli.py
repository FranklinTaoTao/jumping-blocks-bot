"""Exercise CLI parsing and defaults without opening a desktop connection."""

import argparse
import contextlib
import io
import sys
import unittest
from types import SimpleNamespace

from jumpbot.cli import get_components, main, parse_roi
from jumpbot.diagnostics import CaptureTimeline


class CliTests(unittest.TestCase):
    def test_capture_timeline_reports_interruptions_without_logging_regular_poll_misses(self):
        timeline = CaptureTimeline()
        timeline.captured(1.0, {})
        timeline.interrupted("no_new_frame")
        regular = {}
        timeline.captured(1.02, regular)
        self.assertEqual(regular, {})

        timeline.interrupted("out_of_focus")
        timeline.interrupted("no_new_frame")
        timeline.interrupted("capture_error")
        resumed = {}
        timeline.captured(5.02, resumed)
        self.assertEqual(
            resumed["capture_gap"],
            {
                "ms": 4000.0,
                "interruptions": {"out_of_focus": 1, "no_new_frame": 1, "capture_error": 1},
            },
        )
        self.assertEqual(timeline.interruptions, {})
        unexplained = {}
        timeline.captured(5.32, unexplained)
        self.assertEqual(unexplained["capture_gap"], {"ms": 300.0, "interruptions": {}})

    def test_timing_experiments_cannot_be_combined(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            main(["live", "--align-delivery", "--timing-replan"])
        self.assertEqual(error.exception.code, 2)

    def test_roi_validation(self):
        self.assertEqual(parse_roi("0,12,420,744"), (0, 12, 420, 744))
        for invalid in ("bad", "1,2,3", "0,0,0,744", "-1,0,420,744"):
            with self.subTest(value=invalid), self.assertRaises(argparse.ArgumentTypeError):
                parse_roi(invalid)

    def test_invalid_live_options_fail_before_opening_the_display(self):
        for option in (
            "--fps",
            "--duration",
            "--rounds",
            "--target",
            "--render-delay",
            "--input-lag",
        ):
            value = "0" if option in ("--fps", "--duration") else "-1"
            with self.subTest(option=option), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    main(["live", option, value])
                self.assertEqual(error.exception.code, 2)

    def test_bundled_calibration_is_the_default(self):
        args = SimpleNamespace(physics=None, roi=None, planner="python", click_cost=12.0)
        _, tracker, _ = get_components(args)
        expected = 1403.2053064936888 if sys.platform == "win32" else 1406.7818715834014
        self.assertAlmostEqual(tracker.physics.gravity, expected)


if __name__ == "__main__":
    unittest.main()
