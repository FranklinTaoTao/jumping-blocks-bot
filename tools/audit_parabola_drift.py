"""Fit independent jump arcs in recordings and compare early and late physics."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jumpbot.vision import Rect, camera_shift


def read_frames(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.endswith("\n"):
                continue
            # The replay path is larger than the measurements and isn't used.
            prefix, marker, _ = line.partition(', "decision":')
            yield json.loads(prefix + "}" if marker else prefix)


def fit_arcs(path):
    poses, presses, previous = [], [], []
    scroll = 0.0
    for frame in read_frames(path):
        telemetry = frame.get("telemetry", {})
        if frame.get("sent") in (-1, 1) and "input_pressed_at" in telemetry:
            presses.append((telemetry["input_pressed_at"], frame["sent"]))
        p = frame.get("player")
        if not frame.get("active") or not p or not (27 <= p["w"] <= 33 and 27 <= p["h"] <= 33):
            previous = []
            continue
        boxes = [Rect(**box) for box in frame["obstacles"]]
        shift, tracked = camera_shift(previous, boxes)
        previous = boxes
        scroll += shift
        x, y = p["x"] + p["w"] / 2, p["y"] + p["h"] / 2
        # Keep repeats only when camera-relative geometry actually changes.
        pose = (x, y - scroll, y, float(tracked))
        if not poses or poses[-1][1:] != pose:
            poses.append((frame["t"], *pose))
    return fit_poses(poses, presses)


def fit_hover_arcs(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    delay = data.get("render_delay", 0.0)
    poses = [(row["t"] + delay, row["x"], row["y"], row["y"], 1.0) for row in data["measurements"]]
    presses = [(row["t"], row["direction"]) for row in data["presses"]]
    return fit_poses(poses, presses)


def fit_poses(poses, presses):
    if len(poses) < 12 or len(presses) < 2:
        return []
    data = np.asarray(poses)
    t = data[:, 0]
    delay_grid = np.arange(0.0, 0.091, 0.001)
    arcs = []
    for (prev, last_direction), (press, direction), (following, _) in zip(
        presses, presses[1:], presses[2:] + [(np.inf, 0)]
    ):
        if last_direction == direction or press - prev < 0.19:
            continue
        first, middle, end = np.searchsorted(t, [press - 0.070, press + 0.135, following])
        before = data[first:middle]
        if len(before) < 6 or before[0, 0] > press - 0.035 or before[-1, 0] < press + 0.095:
            continue
        x = before[:, 1]
        if x.min() < 25 or x.max() > 395:
            continue
        elapsed = before[:, 0] - press
        basis = 150.4 * direction * np.abs(elapsed[:, None] - delay_grid)
        errors = x[:, None] - basis
        rms_x = errors.std(axis=0)
        best = int(np.argmin(rms_x))
        if not 0 < best < len(delay_grid) - 1 or rms_x[best] > 2:
            continue
        delay = delay_grid[best]
        after = data[np.searchsorted(t, press + delay + 0.014) : end]
        after = after[after[:, 0] <= press + delay + 0.46]
        if len(after) < 9 or after[-1, 0] - after[0, 0] < 0.21:
            continue
        if np.max(np.diff(after[:, 0])) > 0.070 or np.max(np.abs(np.diff(after[:, 2]))) > 42:
            continue
        time = after[:, 0] - press - delay
        scrolled = np.ptp(after[:, 3] - after[:, 2]) > 1
        if scrolled and np.mean(after[:, 4]) < 0.7:
            continue
        design = np.column_stack((np.ones(len(time)), time, 0.5 * time**2))
        height = after[:, 2]
        params = np.linalg.lstsq(design, height, rcond=None)[0]
        rms_y = float(np.sqrt(np.mean((height - design @ params) ** 2)))
        jump, gravity = -params[1], params[2]
        if rms_y <= 2 and 450 < jump < 950 and 850 < gravity < 2000:
            arcs.append((press, jump, gravity, delay * 1000, rms_y, bool(scrolled), time, height))
    return arcs


def summary(arcs):
    if not arcs:
        return {"arcs": 0}
    _, jump, gravity, delay, rms, _ = np.asarray([arc[:6] for arc in arcs]).T
    return {
        "arcs": len(arcs),
        "jump_speed": np.percentile(jump, [10, 50, 90]).round(2).tolist(),
        "gravity": np.percentile(gravity, [10, 50, 90]).round(2).tolist(),
        "reset_ms": np.percentile(delay, [10, 50, 90]).round(2).tolist(),
        "median_fit_rms": round(float(np.median(rms)), 3),
    }


def held_out_error(arcs, jump, gravity):
    errors = []
    for *_, time, height in arcs:
        curve = -jump * time + 0.5 * gravity * time**2
        anchor = time <= 0.100
        if np.count_nonzero(anchor) >= 2:
            residual = height - curve
            errors.extend(residual[~anchor] - np.mean(residual[anchor]))
    return {
        "samples": len(errors),
        "rms": round(float(np.sqrt(np.mean(np.square(errors)))), 3) if errors else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    reports = {}
    recordings = [(path, fit_arcs) for path in sorted(args.root.rglob("frames.jsonl"))]
    recordings.extend((path, fit_hover_arcs) for path in sorted(args.root.rglob("hover.json")))
    for path, fit in recordings:
        name = path.relative_to(args.root).as_posix()
        arcs = fit(path)
        reports[name] = {}
        for label, group in (
            ("stationary_camera", [a for a in arcs if not a[5]]),
            ("scrolling_camera", [a for a in arcs if a[5]]),
        ):
            half = len(group) // 2
            reports[name][label] = {
                "all": summary(group),
                "early": summary(group[:half]),
                "late": summary(group[half:]),
            }
            if half >= 10:
                early = reports[name][label]["early"]
                reports[name][label]["late_forecast"] = {
                    "fixed": held_out_error(group[half:], 688.55, 1415),
                    "early_fit": held_out_error(
                        group[half:], early["jump_speed"][1], early["gravity"][1]
                    ),
                }
        print(name, len(arcs), flush=True)
    args.output.write_text(json.dumps(reports, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
