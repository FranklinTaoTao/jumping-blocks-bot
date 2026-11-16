"""Optional asynchronous output recording; never part of the input pipeline."""

import json
import queue
import threading
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np


class Recorder:
    def __init__(self, directory, fps=30):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.log = (self.directory / "frames.jsonl").open("w", buffering=65536)
        self.queue = queue.Queue(maxsize=90)
        self.fps = fps
        self.next_frame = 0.0
        self.dropped = 0
        self.error = None
        self.timestamps = []
        self.worker = threading.Thread(target=self._encode, daemon=True)
        self.worker.start()

    def _encode(self):
        writer = cv2.VideoWriter(
            str(self.directory / "diagnostic.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            (840, 864),
        )
        try:
            if not writer.isOpened():
                raise RuntimeError("Could not open diagnostic video output")
            from .render import annotate

            while True:
                item = self.queue.get()
                if item is None:
                    break
                t, frame, obs, state, decision, sent = item
                canvas = np.zeros((864, 840, 3), np.uint8)
                left = cv2.resize(frame, (420, round(frame.shape[0] * 420 / frame.shape[1])))
                canvas[: min(864, len(left)), :420] = left[:864]
                if obs:
                    right = annotate(obs, state, decision, f"{t:.3f}s sent={sent}")
                    canvas[: min(864, len(right)), 420:] = right[:864]
                cv2.putText(
                    canvas, f"{t:.3f}s", (5, 855), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1
                )
                writer.write(canvas)
        except Exception as e:
            self.error = repr(e)
        finally:
            writer.release()

    def add(self, t, frame, obs, state, decision, sent=0, ui=None, telemetry=None):
        record = {
            "t": t,
            "sent": sent,
            "ui": ui,
            "player": asdict(obs.player) if obs and obs.player else None,
            "roi": obs.roi if obs else None,
            "active": obs.active if obs else None,
            "obstacles": [asdict(b) for b in obs.obstacles] if obs else [],
        }
        record["shapes"] = [asdict(h) for h in obs.shapes] if obs else []
        record["telemetry"] = telemetry or {}
        if state:
            record["state"] = {
                k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in asdict(state).items()
            }
        if decision:
            record["decision"] = {
                "action": decision.action,
                "reason": decision.reason,
                "score": decision.score if np.isfinite(decision.score) else None,
                "predicted_horizon": decision.predicted_horizon,
                "path": decision.path.tolist(),
            }
        self.log.write(
            json.dumps(
                record,
                allow_nan=False,
                default=lambda value: (
                    value.tolist() if isinstance(value, np.ndarray) else value.item()
                ),
            )
            + "\n"
        )
        if t >= self.next_frame:
            self.next_frame = max(self.next_frame + 1 / self.fps, t - 0.1)
            try:
                self.queue.put_nowait((t, frame, obs, state, decision, sent))
                self.timestamps.append(t)
            except queue.Full:
                self.dropped += 1

    def close(self):
        self.log.close()
        if self.worker.is_alive():
            self.queue.put(None, timeout=5)
            self.worker.join(timeout=10)
        (self.directory / "recording.json").write_text(
            json.dumps(
                {
                    "fps": self.fps,
                    "dropped": self.dropped,
                    "error": self.error,
                    "dimensions": [840, 864],
                    "timestamps": self.timestamps,
                },
                indent=2,
            )
        )
