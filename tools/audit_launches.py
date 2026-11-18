"""Evaluate recorded, delivered resets without assuming future planned inputs.

Fit pre-reset velocity from camera-corrected observations, then forecast the
actual recorded reset at a fixed horizon. Split each recording chronologically;
select parameters using the first half and assess them on the second half.
This measures motion error, not counterfactual gameplay or route safety.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jumpbot.vision import Rect, camera_shift


def evaluate(folder, rows, gravity, jump, horizon, press_phase=1.0, include_samples=False):
    previous = None
    scroll = 0.0
    history, launches = [], []
    last_press = -100.0

    def press_time(row):
        tele = row["telemetry"]
        send_time = tele.get("input_delivery", {}).get("send_ms")
        if press_phase != 1 and send_time is None:
            raise ValueError("This recording does not contain button-down call timings")
        return tele["input_pressed_at"] - (1 - press_phase) * (send_time or 0) / 1000

    for row in rows:
        boxes = [Rect(**b) for b in row["obstacles"]]
        state = row.get("state")
        if state and previous is not None:
            scroll += camera_shift(previous, boxes)[0]
        previous = boxes if state else None
        if not state:
            history.clear()
            continue
        row["world_y"] = state["y"] - scroll
        history.append((row["t"], row["world_y"]))
        if not row["sent"]:
            continue
        before = np.array([point for point in history[-6:] if point[0] > last_press + 0.08])
        last_press = row["telemetry"]["input_pressed_at"]
        player = row["player"]
        if len(before) < 3 or before[-1, 0] - before[0, 0] <= 0.05 or not 27 <= player["h"] <= 33:
            continue
        times = before[:, 0] - row["t"]
        design = np.column_stack([times, np.ones(len(times))])
        velocity, intercept = np.linalg.lstsq(
            design, before[:, 1] - 0.5 * gravity * times**2, rcond=None
        )[0]
        residual = before[:, 1] - (intercept + velocity * times + 0.5 * gravity * times**2)
        if np.sqrt(np.mean(residual**2)) < 2:
            launches.append((row["t"], last_press, press_time(row), row["world_y"], velocity))
    times = np.array([row["t"] for row in rows])
    presses = np.array([row["telemetry"]["input_pressed_at"] for row in rows if row["sent"]])
    samples = []
    for captured, returned, pressed, y, velocity in launches:
        index = np.searchsorted(times, captured + horizon)
        if index >= len(rows):
            continue
        future = rows[index]
        player = future.get("player")
        dt = future["t"] - captured
        if (
            not player
            or not 27 <= player["w"] <= 33
            or not 27 <= player["h"] <= 33
            or dt > horizon + 0.06
            or "world_y" not in future
            or np.any((presses > returned + 0.001) & (presses < future["t"]))
        ):
            continue
        samples.append((pressed - captured, dt, y, velocity, future["world_y"], captured))
    if len(samples) < 4:
        raise ValueError(f"Too few uninterrupted reset forecasts in {folder}")
    data = np.array(samples)
    # Solve the two ballistic segments for the observed reset time. For
    # nearly identical incoming and reset velocities, the pose does not
    # provide enough information to distinguish nearby reset times.
    age, duration, y, velocity, measured, captured = data.T
    base = y - jump * duration + 0.5 * gravity * duration**2
    slope = velocity + jump - gravity * duration
    informative = slope + 2 * gravity * age >= 250
    discriminant = slope**2 + 4 * gravity * (measured - base)
    latency = (np.sqrt(np.maximum(0, discriminant)) - slope) / (2 * gravity)
    # An effect outside the observed interval belongs to a bad fit or a
    # dropped reset. Report its count separately from valid delay estimates.
    valid = informative & (discriminant >= 0) & (age - 0.008 <= latency) & (latency <= duration)
    observed = (latency[valid] - age[valid]) * 1000
    phase_measurement = {
        "informative": int(np.count_nonzero(informative)),
        "valid": len(observed),
        "delay_ms_p10_p50_p95": np.percentile(observed, [10, 50, 95]).round(2).tolist()
        if len(observed)
        else None,
        "presentation_to_reset_ms_p50_p95": np.percentile(latency[valid] * 1000, [50, 95])
        .round(2)
        .tolist()
        if len(observed)
        else None,
    }
    if include_samples:
        phase_measurement["samples"] = [
            {"t": round(t, 4), "effect_at": round(t + delay, 4), "delay_ms": round(after_call, 2)}
            for t, delay, after_call in zip(captured[valid], latency[valid], observed)
        ]
    results = []
    middle = len(data) // 2
    for delay in np.arange(0, 0.061, 0.005):
        latency = data[:, 0] + delay
        after = data[:, 1] - latency
        predicted = (
            data[:, 2]
            + data[:, 3] * latency
            + 0.5 * gravity * latency**2
            - jump * after
            + 0.5 * gravity * after**2
        )
        error = data[:, 4] - predicted
        results.append(
            {
                "delay_ms": round(delay * 1000),
                "train_rms": round(float(np.sqrt(np.mean(error[:middle] ** 2))), 2),
                "held_out_rms": round(float(np.sqrt(np.mean(error[middle:] ** 2))), 2),
                "median_error": round(float(np.median(error)), 2),
            }
        )
    return {
        "name": folder.name,
        "samples": len(data),
        "press_phase": press_phase,
        "observed_reset": phase_measurement,
        "results": results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folders", type=Path, nargs="+")
    parser.add_argument("--gravity", type=float, default=1415.0)
    parser.add_argument("--jump", type=float, default=688.55)
    parser.add_argument("--horizon", type=float, default=0.16)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--compare-press-phase",
        action="store_true",
        help="Compare start, midpoint and return of the Windows input call",
    )
    parser.add_argument(
        "--include-samples",
        action="store_true",
        help="Include individual estimated reset times in the report",
    )
    args = parser.parse_args()
    phases = (0.0, 0.5, 1.0) if args.compare_press_phase else (1.0,)
    results = []
    for folder in args.folders:
        with (folder / "frames.jsonl").open(encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream if line.endswith("\n")]
        results.extend(
            evaluate(
                folder, rows, args.gravity, args.jump, args.horizon, phase, args.include_samples
            )
            for phase in phases
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    for result in results:
        chosen = min(result["results"], key=lambda row: row["train_rms"])
        print(
            json.dumps(
                {
                    "name": result["name"],
                    "samples": result["samples"],
                    "press_phase": result["press_phase"],
                    "observed_reset": {
                        k: v for k, v in result["observed_reset"].items() if k != "samples"
                    },
                    **chosen,
                }
            )
        )
