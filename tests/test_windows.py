"""Check Windows input guards without sending actual mouse input."""

import sys
import unittest
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock, patch

if sys.platform == "win32":
    import pywintypes

    from jumpbot.windows import CaptureUnavailable, GameWindow


@unittest.skipUnless(sys.platform == "win32", "Windows backend")
class WindowsInputTests(unittest.TestCase):
    def test_scheduled_input_waits_for_its_planned_button_time(self):
        clock = [10.0]
        events = []

        def sleep(seconds):
            clock[0] += seconds

        with (
            patch("jumpbot.windows.time.perf_counter", side_effect=lambda: clock[0]),
            patch("jumpbot.windows.time.sleep", side_effect=sleep),
            patch("jumpbot.windows.win32gui.WindowFromPoint", return_value=42),
            patch("jumpbot.windows.win32gui.GetAncestor", return_value=42),
            patch("jumpbot.windows.win32api.SetCursorPos"),
            patch("jumpbot.windows.win32api.GetCursorPos", return_value=(200, 300)),
            patch(
                "jumpbot.windows._button", side_effect=lambda flag: events.append((flag, clock[0]))
            ),
        ):
            self.assertTrue(self.game.click_at(100, 100, not_before=10.020, expires_at=10.090))
        self.assertGreaterEqual(events[0][1], 10.020)
        self.assertLess(events[0][1], 10.025)
        self.assertEqual(self.game.last_press_time, events[0][1])
        self.assertAlmostEqual(events[1][1] - events[0][1], 0.008)
        timing = self.game.last_press_timing
        self.assertAlmostEqual(timing["wait_ms"], 20)
        self.assertAlmostEqual(timing["hold_ms"], 8)
        self.assertAlmostEqual(timing["release_ms"], 0)
        self.assertAlmostEqual(
            timing["wait_end_error_ms"] + timing["validate_ms"] + timing["send_ms"],
            (events[0][1] - 10.020) * 1000,
        )

    def test_scheduled_input_cancels_if_focus_changes_during_wait(self):
        clock = [10.0]
        self.game.focused.side_effect = lambda: clock[0] < 10.010

        def sleep(seconds):
            clock[0] += seconds

        with (
            patch("jumpbot.windows.time.perf_counter", side_effect=lambda: clock[0]),
            patch("jumpbot.windows.time.sleep", side_effect=sleep),
            patch("jumpbot.windows.win32gui.WindowFromPoint", return_value=42),
            patch("jumpbot.windows.win32gui.GetAncestor", return_value=42),
            patch("jumpbot.windows.win32api.SetCursorPos"),
            patch("jumpbot.windows._button") as button,
        ):
            self.assertFalse(self.game.click_at(100, 100, not_before=10.020, expires_at=10.090))
            button.assert_not_called()
        self.assertIsNone(self.game.last_press_timing)

    def test_expired_schedule_never_moves_or_clicks(self):
        with (
            patch("jumpbot.windows.time.perf_counter", return_value=10.0),
            patch("jumpbot.windows.win32api.SetCursorPos") as cursor,
            patch("jumpbot.windows._button") as button,
        ):
            self.assertFalse(self.game.click_at(100, 100, not_before=10.1, expires_at=10.09))
            cursor.assert_not_called()
            button.assert_not_called()

    def setUp(self):
        self.game = GameWindow.__new__(GameWindow)
        self.game.window = SimpleNamespace(id=42)
        self.game.focused = Mock(return_value=True)
        self.game.stopped = Mock(return_value=False)
        self.game._geometry = Mock(return_value=(100, 200, 420, 750))
        self.game.last_press_time = None
        self.game.batch_jumps = False
        self.game.message_jumps = False
        self.game._release_worker = None
        self.game._pending_release = None

    def test_unfocused_or_stopped_never_sends_input(self):
        with patch("jumpbot.windows._button") as button:
            self.game.focused.return_value = False
            self.assertFalse(self.game.click_at(100, 100))
            self.game.focused.return_value = True
            self.game.stopped.return_value = True
            self.assertFalse(self.game.click_at(100, 100))
            button.assert_not_called()

    def test_unavailable_desktop_does_not_crash_stop_check_or_send_a_click(self):
        denied = pywintypes.error(5, "Cursor", "Access is denied")
        with (
            patch("jumpbot.windows.win32api.GetAsyncKeyState", return_value=0),
            patch("jumpbot.windows.win32api.GetCursorPos", side_effect=denied),
            patch("jumpbot.windows.win32gui.WindowFromPoint", return_value=42),
            patch("jumpbot.windows.win32gui.GetAncestor", return_value=42),
            patch("jumpbot.windows.win32api.SetCursorPos", side_effect=denied),
            patch("jumpbot.windows._button") as button,
        ):
            self.assertFalse(GameWindow.stopped(self.game))
            self.assertFalse(self.game.click_at(100, 100))
            button.assert_not_called()

    def test_invalid_roi_and_coordinates_never_send_input(self):
        with patch("jumpbot.windows._button") as button:
            self.assertFalse(self.game.click(-1, (0, 0, 421, 750)))
            self.assertFalse(self.game.click_at(420, 50))
            self.assertFalse(self.game.click_at(50, -1))
            button.assert_not_called()

    def test_resize_after_capture_defers_input_until_a_fresh_frame(self):
        self.game._frame_size = (400, 750)
        with patch("jumpbot.windows._button") as button:
            self.assertFalse(self.game.click_at(100, 100))
            button.assert_not_called()

    def test_covering_window_blocks_click(self):
        with (
            patch("jumpbot.windows.win32gui.WindowFromPoint", return_value=99),
            patch("jumpbot.windows.win32gui.GetAncestor", return_value=99),
            patch("jumpbot.windows._button") as button,
        ):
            self.assertFalse(self.game.click_at(100, 100))
            button.assert_not_called()

    def test_successful_click_uses_screen_coordinates_and_releases(self):
        from jumpbot.windows import win32con

        with (
            patch("jumpbot.windows.win32gui.WindowFromPoint", return_value=42),
            patch("jumpbot.windows.win32gui.GetAncestor", return_value=42),
            patch("jumpbot.windows.win32api.SetCursorPos") as cursor,
            patch("jumpbot.windows._button") as button,
            patch("jumpbot.windows.time.sleep"),
        ):
            self.assertTrue(self.game.click_at(100, 100))
            cursor.assert_called_once_with((200, 300))
            self.assertEqual(
                [call.args[0] for call in button.call_args_list],
                [win32con.MOUSEEVENTF_LEFTDOWN, win32con.MOUSEEVENTF_LEFTUP],
            )
            self.assertIsNotNone(self.game.last_press_time)

    def test_batch_jumps_send_down_and_up_together(self):
        from jumpbot.windows import win32con

        self.game.batch_jumps = True
        with (
            patch("jumpbot.windows.win32gui.WindowFromPoint", return_value=42),
            patch("jumpbot.windows.win32gui.GetAncestor", return_value=42),
            patch("jumpbot.windows.win32api.SetCursorPos"),
            patch("jumpbot.windows._button") as button,
            patch("jumpbot.windows.time.sleep") as sleep,
        ):
            self.assertTrue(self.game.click(1, (0, 0, 420, 750)))
            button.assert_called_once_with(
                win32con.MOUSEEVENTF_LEFTDOWN, win32con.MOUSEEVENTF_LEFTUP
            )
            sleep.assert_not_called()

    def test_message_jumps_target_the_game_child_in_client_coordinates(self):
        from jumpbot.windows import win32con

        self.game.message_jumps = True
        self.game._release_worker = ThreadPoolExecutor(max_workers=1)
        self.addCleanup(self.game._release_worker.shutdown)
        with (
            patch("jumpbot.windows.win32gui.WindowFromPoint", return_value=77),
            patch("jumpbot.windows.win32gui.GetAncestor", return_value=42),
            patch("jumpbot.windows.win32gui.ScreenToClient", return_value=(25, 75)) as point,
            patch("jumpbot.windows.win32gui.PostMessage") as post,
            patch("jumpbot.windows.win32api.SetCursorPos"),
            patch("jumpbot.windows._button") as button,
            patch("jumpbot.windows.time.sleep"),
        ):
            self.assertTrue(self.game.click(1, (0, 0, 420, 750)))
            self.assertTrue(self.game.click_at(100, 100))
        point.assert_called_once_with(77, (436, 762))
        self.assertEqual(
            [call.args for call in post.call_args_list],
            [
                (77, win32con.WM_LBUTTONDOWN, win32con.MK_LBUTTON, 25 | (75 << 16)),
                (77, win32con.WM_LBUTTONUP, 0, 25 | (75 << 16)),
            ],
        )
        self.assertEqual(button.call_count, 2)

    def test_posted_click_returns_while_its_release_is_pending(self):
        release = Event()
        self.game.message_jumps = True
        self.game._release_worker = ThreadPoolExecutor(max_workers=1)
        self.addCleanup(self.game._release_worker.shutdown)
        with (
            patch("jumpbot.windows.win32gui.WindowFromPoint", return_value=42),
            patch("jumpbot.windows.win32gui.GetAncestor", return_value=42),
            patch("jumpbot.windows.win32gui.ScreenToClient", return_value=(25, 75)),
            patch("jumpbot.windows.win32api.SetCursorPos"),
            patch("jumpbot.windows._post_button") as button,
            patch.object(
                self.game, "_release_message", side_effect=lambda *_: release.wait(1)
            ) as finish,
        ):
            try:
                self.assertTrue(self.game.click(1, (0, 0, 420, 750)))
                button.assert_called_once_with(42, (25, 75), pressed=True)
                self.assertFalse(self.game._pending_release.done())
            finally:
                release.set()
            self.game._pending_release.result(timeout=1)
        finish.assert_called_once_with(42, (25, 75), self.game.last_press_time + 0.008)

    def test_blocking_ab_click_posts_release_before_returning(self):
        self.game.message_jumps = True
        with (
            patch("jumpbot.windows.win32gui.WindowFromPoint", return_value=42),
            patch("jumpbot.windows.win32gui.GetAncestor", return_value=42),
            patch("jumpbot.windows.win32gui.ScreenToClient", return_value=(25, 75)),
            patch("jumpbot.windows.win32api.SetCursorPos"),
            patch("jumpbot.windows._post_button") as button,
            patch("jumpbot.windows.time.sleep") as sleep,
        ):
            self.assertTrue(self.game.click(1, (0, 0, 420, 750), blocking_release=True))
        sleep.assert_called_once_with(0.008)
        self.assertEqual(
            [event.kwargs["pressed"] for event in button.call_args_list], [True, False]
        )
        self.assertIsNone(self.game._pending_release)

    def test_stale_presentation_restarts_capture_before_returning_pixels(self):
        self.game.capture_kind = "dxgi"
        self.game._capture_region = (100, 200, 520, 950)
        frame = object()
        self.game.capture_backend = Mock(width=1920, height=1080, is_capturing=True)
        self.game.capture_backend.get_latest_frame.side_effect = [(frame, 7.0), (frame, 10.0)]
        with patch("jumpbot.windows.time.perf_counter", return_value=10.0):
            with self.assertRaises(CaptureUnavailable):
                self.game.capture()
            self.assertIs(self.game.capture(), frame)
        self.assertEqual(self.game.last_capture_time, 10.0)
        self.game.capture_backend.stop.assert_called_once()
        self.game.capture_backend.start.assert_called_once_with(
            region=(100, 200, 520, 950), target_fps=120
        )
