import sys
import unittest

import cv2

from jumpbot.native_vision import NativeVision, library_path
from jumpbot.score import ScoreReader
from jumpbot.vision import Detector
from tools.offline_course_bench import assets, render


@unittest.skipUnless(
    sys.platform == "win32" and library_path().is_file(), "Build native vision first"
)
class NativeVisionTests(unittest.TestCase):
    def test_returns_metadata_for_a_board_at_two_capture_sizes(self):
        course = {
            "height": 736,
            "boxes": [[0, 280, 100, 38], [270, 280, 150, 38], [240, 150, 31, 31]],
        }
        image = render(course, (190, 368, 0, 0, 17), *assets())
        score = ScoreReader()
        for width in (420, 722):
            height = round(image.shape[0] * width / image.shape[1])
            resized = cv2.resize(image, (width, height))
            roi = (0, 0, width, height)
            reference = Detector(roi).observe(resized)
            native = NativeVision(roi)
            self.addCleanup(native.close)
            result = native.inspect(cv2.cvtColor(resized, cv2.COLOR_BGR2BGRA))
            measured = result.observation
            self.assertIsNone(measured.image)
            self.assertEqual(result.frame_size, (width, height))
            self.assertEqual(result.board_height, reference.image.shape[0])
            self.assertEqual(measured.player, reference.player)
            self.assertEqual(measured.obstacles, reference.obstacles)
            self.assertEqual(result.score, score.read(reference.image))
