"""Compare native vision with the original OpenCV/Python detector on identical pixels."""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jumpbot.native_vision import NativeVision
from jumpbot.score import ScoreReader
from jumpbot.ui import ScreenHandler
from jumpbot.vision import Detector


def compare(video, crop, stride, width):
    detector, score, ui = Detector(), ScoreReader(), ScreenHandler()
    native = NativeVision()
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        native.close()
        raise RuntimeError(f"Could not open {video}")
    timings = {"gameplay": [], "menu_or_untracked": []}
    differences = Counter()
    counts = Counter()
    try:
        index = 0
        while capture.grab():
            index += 1
            if (index - 1) % stride:
                continue
            ok, frame = capture.retrieve()
            if not ok:
                raise RuntimeError(f"Could not decode frame {index}")
            if crop:
                x, y, w, h = crop
                frame = frame[y : y + h, x : x + w]
            if width:
                frame = cv2.resize(frame, (width, round(width * frame.shape[0] / frame.shape[1])))
            started = time.perf_counter()
            obs = detector.observe(frame)
            read_score = score.read(obs.image) if obs is not None else None
            tracking = bool(obs and obs.active and obs.player)
            action = None if tracking else ui.inspect(frame)
            reference_time = time.perf_counter() - started
            bgra = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
            started = time.perf_counter()
            result = native.inspect(bgra)
            native_time = time.perf_counter() - started
            new = result.observation
            differences["roi"] += (obs.roi if obs else None) != (new.roi if new else None)
            differences["player"] += (obs.player if obs else None) != (new.player if new else None)
            differences["active"] += (obs.active if obs else None) != (new.active if new else None)
            differences["obstacles"] += (obs.obstacles if obs else []) != (
                new.obstacles if new else []
            )
            expected_shapes = obs.shapes if obs else []
            actual_shapes = new.shapes if new else []
            differences["shape_hull"] += len(expected_shapes) != len(actual_shapes) or any(
                not np.array_equal(a.points, b.points)
                for a, b in zip(expected_shapes, actual_shapes)
            )
            differences["score"] += read_score != result.score
            differences["ui"] += (action.kind if action else None) != (
                result.ui.kind if result.ui else None
            )
            if action and result.ui and action.kind == result.ui.kind:
                differences["ui_position"] += (
                    max(abs(action.x - result.ui.x), abs(action.y - result.ui.y)) > 0.01
                )
            counts["shape_frames"] += bool(expected_shapes)
            counts["readable_scores"] += read_score is not None
            counts["recognized_menu_frames"] += action is not None
            group = "gameplay" if tracking else "menu_or_untracked"
            timings[group].append((reference_time * 1000, native_time * 1000))
    finally:
        capture.release()
        native.close()
    if not any(timings.values()):
        raise RuntimeError(f"No frames decoded from {video}")
    return {
        "video": str(video),
        "frames": sum(map(len, timings.values())),
        "counts": dict(counts),
        "different_frames_by_field": dict(differences),
        "python_native_ms_p50_p95": {
            group: {
                "frames": len(values),
                "p50": np.percentile(values, 50, axis=0).round(3).tolist(),
                "p95": np.percentile(values, 95, axis=0).round(3).tolist(),
            }
            for group, values in timings.items()
            if values
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--crop", nargs=4, type=int, metavar=("X", "Y", "W", "H"))
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--width", type=int, help="Resize samples before running both detectors")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.stride < 1 or args.width is not None and args.width < 1:
        parser.error("Stride and width must be positive")
    if args.crop and (min(args.crop[:2]) < 0 or min(args.crop[2:]) < 1):
        parser.error("Crop requires nonnegative offsets and positive dimensions")
    cv2.setNumThreads(1)
    report = json.dumps(compare(args.video, args.crop, args.stride, args.width), indent=2)
    if args.output:
        args.output.write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
