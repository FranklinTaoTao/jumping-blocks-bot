"""Fit the first ten hover jumps and validate on the untouched remaining jumps."""

import argparse
import json
from pathlib import Path

import numpy as np


def fit_hover(source, training_jumps=10):
    data = json.loads(source.read_text())
    measurements = data["measurements"]
    t = np.array([r["t"] + data["render_delay"] for r in measurements])
    x = np.array([r["x"] for r in measurements])
    y = np.array([r["y"] for r in measurements])
    clicks = data["presses"]
    times = np.array([r["t"] for r in clicks])
    directions = np.array([r["direction"] for r in clicks])
    train = np.array([r["jump"] <= training_jumps for r in measurements])
    candidates_x, candidates_y = [], []
    for delay in np.arange(0, 0.101, 0.0005):
        ages = np.maximum(t[:, None] - times[None, :] - delay, 0)
        distance = ages @ np.diff(np.r_[0, directions])
        design_x = np.column_stack([distance, np.ones(len(t))])
        horizontal = np.linalg.lstsq(design_x[train], x[train], rcond=None)[0]
        error_x = x - design_x @ horizontal
        candidates_x.append((np.mean(error_x[train] ** 2), delay, horizontal, error_x))
        events = times + delay
        arc = np.maximum(0, np.searchsorted(events, t) - 1)
        prefix = np.r_[0, np.cumsum(np.diff(events) ** 2)]
        total_time = t - events[0]
        squared_time = prefix[arc] + (t - events[arc]) ** 2
        design_y = np.column_stack([-total_time, squared_time / 2, np.ones(len(t))])
        vertical = np.linalg.lstsq(design_y[train], y[train], rcond=None)[0]
        error_y = y - design_y @ vertical
        candidates_y.append((np.mean(error_y[train] ** 2), delay, vertical, error_y))
    _, delay_x, horizontal, error_x = min(candidates_x, key=lambda a: a[0])
    _, delay_y, vertical, error_y = min(candidates_y, key=lambda a: a[0])
    return {
        "physics": {
            "speed": float(horizontal[0]),
            "jump": float(vertical[0]),
            "gravity": float(vertical[1]),
        },
        "render_delay": 0.0,
        "input_delay_x": float(delay_x),
        "input_delay_y": float(delay_y),
        "training_jumps": training_jumps,
        "held_out_jumps": data["jumps"] - training_jumps,
        "held_out_rmse_xy": [
            float(np.sqrt(np.mean(error_x[~train] ** 2))),
            float(np.sqrt(np.mean(error_y[~train] ** 2))),
        ],
        "source": str(source),
        "fit_errors": {
            "t": t.tolist(),
            "train": train.tolist(),
            "x": error_x.tolist(),
            "y": error_y.tolist(),
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = fit_hover(args.source)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "fit_errors"}, indent=2))
