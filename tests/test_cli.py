"""Exercise CLI parsing and defaults without opening a desktop connection."""

import argparse
import contextlib
import io
import unittest
from types import SimpleNamespace

from jumpbot.cli import get_components, main, parse_roi


class CliTests(unittest.TestCase):
    def test_roi_validation(self):
        self.assertEqual(parse_roi("0,12,420,744"), (0, 12, 420, 744))
        for invalid in ("bad", "1,2,3", "0,0,0,744", "-1,0,420,744"):
            with self.subTest(value=invalid), self.assertRaises(argparse.ArgumentTypeError):
                parse_roi(invalid)

    def test_invalid_live_options_fail_before_opening_the_display(self):
        for option in ("--fps", "--duration", "--rounds", "--target"):
            value = "0" if option in ("--fps", "--duration") else "-1"
            with self.subTest(option=option), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    main(["live", option, value])
                self.assertEqual(error.exception.code, 2)

    def test_bundled_calibration_is_the_default(self):
        args = SimpleNamespace(physics=None, roi=None, planner="python", click_cost=12.0)
        _, tracker, _ = get_components(args)
        self.assertAlmostEqual(tracker.physics.gravity, 1406.7818715834014)


if __name__ == "__main__":
    unittest.main()
