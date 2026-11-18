"""Display live geometry in a separate OpenCV process."""

import multiprocessing
import queue
from dataclasses import replace


def _display(updates, stop, ready):
    import cv2
    import numpy as np

    from .render import annotate

    title = "Jumpbot tracking"
    cv2.setNumThreads(1)
    try:
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(title, 420, 660)
        idle = np.full((180, 420, 3), 32, np.uint8)
        cv2.putText(
            idle,
            "Waiting for game frames",
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (245, 245, 245),
            1,
        )
        cv2.imshow(title, idle)
        cv2.waitKey(1)
        ready.set()
        while not stop.is_set():
            try:
                frame = updates.get(timeout=0.03)
            except queue.Empty:
                frame = None
            if frame is not None:
                while True:
                    try:
                        frame = updates.get_nowait()
                    except queue.Empty:
                        break
                cv2.imshow(title, annotate(*frame))
            if (
                cv2.waitKey(1) & 0xFF in (27, ord("q"))
                or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1
            ):
                break
    finally:
        stop.set()
        ready.set()
        cv2.destroyAllWindows()


class LiveView:
    def __init__(self):
        context = multiprocessing.get_context("spawn")
        self.updates = context.Queue(maxsize=2)
        self.stop = context.Event()
        ready = context.Event()
        self.process = context.Process(
            target=_display, args=(self.updates, self.stop, ready), daemon=True
        )
        self.process.start()
        self.next_update = 0.0
        if not ready.wait(5) or self.stop.is_set():
            self.close()
            raise RuntimeError("Could not open the OpenCV tracking window.")

    def publish(self, captured, observation, state, decision, status):
        if observation is None or captured < self.next_update:
            return
        self.next_update = captured + 1 / 30
        try:
            self.updates.put_nowait((replace(observation, image=None), state, decision, status))
        except queue.Full:
            pass  # Only the preview is sampled. Recording is independent.

    def close(self):
        self.stop.set()
        self.process.join(timeout=2)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join()
        self.updates.cancel_join_thread()
        self.updates.close()
