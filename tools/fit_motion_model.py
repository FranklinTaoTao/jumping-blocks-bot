"""Fit the native camera/impulse model against frozen, actually delivered inputs.

No game control. Later samples are excluded once a subsequent input is sent or
contact is observed. Recordings are evaluated separately to expose transfer error.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from numba import njit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jumpbot.vision import Rect, camera_shift


def samples(folder, contacts):
    rows = [json.loads(line) for line in (folder / "frames.jsonl").open(encoding="utf-8")]
    previous = None
    scroll = 0.0
    for q in rows:
        boxes = [Rect(**b) for b in q["obstacles"]]
        if q.get("state") and previous is not None:
            scroll += camera_shift(previous, boxes)[0]
        previous = boxes if q.get("state") else None
        q["scroll"] = scroll
    times = np.array([q["t"] for q in rows])
    presses = [q["telemetry"]["input_pressed_at"] for q in rows if q["sent"]]
    presses = np.array(presses)
    out = []
    for i, q in enumerate(rows):
        if not q["sent"] or not q.get("state"):
            continue
        p = q.get("player")
        s = q["state"]
        press = q["telemetry"]["input_pressed_at"]
        if not p or not 27 <= p["h"] <= 33:
            continue
        # Never fit to the post-contact animation, including a body that later
        # becomes axis aligned and is detected again.
        stop = min((r["contact_t"] for r in contacts if r["death_ui_t"] > q["t"]), default=1e20)
        next_press = (
            presses[np.searchsorted(presses, press + 0.001)]
            if np.searchsorted(presses, press + 0.001) < len(presses)
            else 1e20
        )
        for horizon in [0.12, 0.20, 0.30, 0.40, 0.50, 0.60]:
            j = int(np.searchsorted(times, q["t"] + horizon))
            if j >= len(rows):
                continue
            f = rows[j]
            fp = f.get("player")
            dt = f["t"] - q["t"]
            if (
                f["t"] >= min(stop - 0.015, next_press)
                or dt > horizon + 0.05
                or not f.get("state")
                or not fp
                or not 27 <= fp["h"] <= 33
                or not 27 <= fp["w"] <= 33
            ):
                continue
            out.append(
                [
                    s["y"],
                    s["vy"],
                    s["camera_speed"],
                    press - q["t"],
                    dt,
                    f["state"]["y"] - f["scroll"] + q["scroll"],
                    q["t"],
                    horizon,
                ]
            )
    return np.array(out)


@njit(cache=True)
def error(data, jump, delay, tau, gravity):
    errors = np.empty(len(data))
    for i in range(len(data)):
        y, vy, camera, pressed, elapsed, actual = data[i, :6]
        anchor = min(368.0, y)
        lag = max(0.0, camera) * tau
        y -= lag
        if camera > 1 and anchor >= y + lag - 3:
            vy += tau * gravity
        peak = min(anchor - lag, y)
        scene = 0.0
        t = 0.0
        event = pressed + delay
        reset = False
        while t < elapsed - 1e-9:
            if not reset and t >= event - 1e-9:
                vy = -jump
                reset = True
            dt = min(0.005, elapsed - t)
            if not reset:
                dt = min(dt, event - t)
            if dt < 1e-9:
                continue
            before = max(0.0, anchor - peak)
            y += vy * dt + 0.5 * gravity * dt * dt
            vy += gravity * dt
            peak = min(peak, y)
            after = max(0.0, anchor - peak)
            if tau <= 0:
                scene = after
            else:
                speed = (after - before) / dt
                scene = after - tau * speed + (scene - before + tau * speed) * np.exp(-dt / tau)
            t += dt
        predicted = y + max(0.0, anchor - peak) - scene
        errors[i] = actual - predicted
    return errors


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folders", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contacts = json.load(open("artifacts/regression-audit/last20-deaths.json"))
    datasets = []
    for folder in args.folders:
        data = samples(folder, [r for r in contacts if r["session"] == folder.name])
        datasets.append(data)
        print(folder.name, "samples", len(data), flush=True)
    results = []
    for gravity, jump, delay, tau in __import__("itertools").product(
        [1390.0, 1415.0, 1440.0],
        np.arange(675.0, 711.0, 5),
        np.arange(0.0, 0.046, 0.005),
        [0.0, 0.005, 0.011, 0.02],
    ):
        row = {
            "jump": float(jump),
            "input_lag": round(float(delay), 3),
            "camera_tau": tau,
            "gravity": gravity,
            "rms": [],
            "bias": [],
            "balanced_rms": [],
            "rms_by_horizon": [],
        }
        for data in datasets:
            e = error(data, jump, delay, tau, gravity)
            row["rms"].append(round(float(np.sqrt(np.mean(e * e))), 3))
            row["bias"].append(round(float(np.median(e)), 3))
            means = [float(np.mean(e[data[:, 7] == h] ** 2)) for h in np.unique(data[:, 7])]
            row["rms_by_horizon"].append([round(float(np.sqrt(m)), 3) for m in means])
            row["balanced_rms"].append(round(float(np.sqrt(np.mean(means))), 3))
        results.append(row)
    results.sort(key=lambda r: r["balanced_rms"][0])
    print("best training models", json.dumps(results[:8], indent=2))
    baseline = {"jump": 688.55, "input_lag": 0.035, "camera_tau": 0.011, "rms": [], "bias": []}
    for data in datasets:
        e = error(data, 688.55, 0.035, 0.011, 1415.0)
        baseline["rms"].append(round(float(np.sqrt(np.mean(e * e))), 3))
        baseline["bias"].append(round(float(np.median(e)), 3))
        baseline.setdefault("horizons", []).append(
            [
                {
                    "seconds": float(h),
                    "samples": int(np.sum(data[:, 7] == h)),
                    "rms": round(float(np.sqrt(np.mean(e[data[:, 7] == h] ** 2))), 3),
                    "error_percentiles": np.percentile(e[data[:, 7] == h], [1, 5, 50, 95, 99])
                    .round(2)
                    .tolist(),
                }
                for h in np.unique(data[:, 7])
            ]
        )
    print("baseline", json.dumps(baseline))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "folders": [str(p) for p in args.folders],
                "samples": [len(d) for d in datasets],
                "baseline": baseline,
                "results": results,
            },
            indent=2,
        )
        + "\n"
    )
