"""Compare saved live sessions without controlling the game."""

import argparse
import gc
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jumpbot.vision import Rect, camera_shift


def audit(folder):
    config_path = folder / "configuration.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    step = config.get("arguments", {}).get("search_step") or 0.025
    with (folder / "frames.jsonl").open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.endswith("\n")]
    scores = []
    round_open = False
    awaiting_zero = True
    best = streak = 0
    last_read = None
    last_restart = last_death = -100.0
    previous = None
    scroll = 0.0
    wall_elapsed = wall_longest = wall_time = active_time = stall = longest_stall = 0.0
    last_t = None
    ready = active = requested = missed = 0
    plans, captures, press_times = [], [], []
    delivery_lateness = []
    delivery_phases = {}
    perception = {}
    capture_to_press = []
    capture_to_button_call = []
    ui = Counter()
    for q in rows:
        t = q["t"]
        s = q.get("state")
        tele = q.get("telemetry", {})
        for phase in ("frame_age_at_capture_ms", "detection_ms", "tracking_ms", "score_ms"):
            if phase in tele:
                perception.setdefault(phase, []).append(tele[phase])
        read = tele.get("score")
        streak = streak + 1 if read is not None and read == last_read else 1
        last_read = read
        dt = t - last_t if last_t is not None else 0
        last_t = t
        dt = dt if 0 < dt < 0.15 else 0
        if q.get("ui"):
            ui[q["ui"]] += 1
        if q.get("ui") == "restart" and t - last_restart > 2:
            if round_open:
                scores.append(best)
            round_open = False
            awaiting_zero = True
            best = 0
            stall = 0
            last_restart = t
        if s and awaiting_zero and read == 0:
            round_open = True
            awaiting_zero = False
            best = 0
        if s and round_open:
            stall += dt
            if read is not None and streak >= 3 and read > best:
                best = read
                stall = 0
            longest_stall = max(longest_stall, stall)
        if q.get("ui") == "decline_continue" and t - last_death > 3:
            if round_open:
                scores.append(best)
            round_open = False
            awaiting_zero = True
            last_death = t
        boxes = [Rect(**b) for b in q["obstacles"]]
        if s and previous:
            scroll += camera_shift(previous, boxes)[0]
        q["audit_scroll"] = scroll
        previous = boxes if s else None
        if not s:
            wall_elapsed = 0
            continue
        active += 1
        active_time += dt
        ready += bool(s["ready"])
        at_wall = s["x"] <= 5 or s["x"] >= 415
        if at_wall:
            wall_elapsed += dt
            wall_time += dt
            wall_longest = max(wall_longest, wall_elapsed, s.get("wall_dwell", 0))
        else:
            wall_elapsed = 0
        d = q.get("decision", {})
        if d.get("action"):
            requested += 1
            missed += not bool(q["sent"])
        if s["ready"] and "planning_ms" in tele:
            plans.append(tele["planning_ms"])
        if "capture_ms" in tele:
            captures.append(tele["capture_ms"])
        if q["sent"] and "input_pressed_at" in tele:
            press_times.append(tele["input_pressed_at"])
            age_ms = 1000 * (tele["input_pressed_at"] - t)
            capture_to_press.append(age_ms)
            if "input_target_at" in tele:
                delivery_lateness.append(
                    1000 * (tele["input_pressed_at"] - tele["input_target_at"])
                )
            delivery = tele.get("input_delivery", {})
            if "send_ms" in delivery:
                capture_to_button_call.append(age_ms - delivery["send_ms"])
            for phase, value in delivery.items():
                delivery_phases.setdefault(phase, []).append(value)
    times = np.array([q["t"] for q in rows])
    sent = np.cumsum([bool(q["sent"]) for q in rows])
    errors = []
    for i, q in enumerate(rows):
        if not q["sent"] or not q.get("decision", {}).get("path"):
            continue
        j = int(np.searchsorted(times, q["t"] + 0.12))
        if j >= len(rows) or j <= i or sent[j - 1] != sent[i]:
            continue
        future = rows[j]
        s = future.get("state")
        p = future.get("player")
        dt = future["t"] - q["t"]
        if not s or not p or not 24 <= p["w"] <= 35 or not 24 <= p["h"] <= 35 or dt > 0.17:
            continue
        path = np.array(q["decision"]["path"])
        tele = q["telemetry"]
        action_latency = tele.get("action_latency_ms", tele["planned_latency_ms"]) / 1000
        pt = action_latency + step * np.arange(1, len(path) + 1)
        if not pt[0] <= dt <= pt[-1]:
            continue
        pred = np.array([np.interp(dt, pt, path[:, k]) for k in [0, 1]])
        actual = np.array([s["x"], s["y"] - future["audit_scroll"] + q["audit_scroll"]])
        errors.append(actual - pred)
    errors = np.array(errors)
    gaps = np.diff(press_times)
    gaps = gaps[(gaps > 0) & (gaps < 1)]
    result = {
        "name": folder.name,
        "completed_scores": scores,
        "best_completed": max(scores, default=0),
        "median_completed": float(np.median(scores)) if scores else None,
        "unfinished_score": best if round_open else None,
        "active_frames": active,
        "ready_pct": round(100 * ready / max(1, active), 1),
        "wall_pct": round(100 * wall_time / max(0.001, active_time), 1),
        "longest_wall_seconds": round(wall_longest, 1),
        "longest_score_stall_seconds": round(longest_stall, 1),
        "requested_actions": requested,
        "undelivered_actions": missed,
        "planning_ms_p50_p95": np.percentile(plans, [50, 95]).round(1).tolist() if plans else [],
        "perception_ms_p50_p95": {
            phase: np.percentile(samples, [50, 95]).round(2).tolist()
            for phase, samples in perception.items()
        },
        "capture_to_press_ms_p50_p95": np.percentile(capture_to_press, [50, 95]).round(2).tolist()
        if capture_to_press
        else [],
        "capture_to_button_call_ms_p50_p95": (
            np.percentile(capture_to_button_call, [50, 95]).round(2).tolist()
            if capture_to_button_call
            else []
        ),
        "delivery_lateness_ms_p50_p95": np.percentile(delivery_lateness, [50, 95]).round(2).tolist()
        if delivery_lateness
        else [],
        "delivery_phases_ms_p50_p95": {
            phase: np.percentile(samples, [50, 95]).round(2).tolist()
            for phase, samples in delivery_phases.items()
        },
        "capture_ms_p50_p95": np.percentile(captures, [50, 95]).round(1).tolist()
        if captures
        else [],
        "input_gap_ms_p10_p50": (1000 * np.percentile(gaps, [10, 50])).round(1).tolist()
        if len(gaps)
        else [],
        "forecast_samples": len(errors),
        "forecast_120ms_bias_xy": np.median(errors, axis=0).round(1).tolist()
        if len(errors)
        else [],
        "forecast_120ms_rms_xy": np.sqrt(np.mean(errors**2, axis=0)).round(1).tolist()
        if len(errors)
        else [],
        "ui_counts": dict(ui),
        "configuration": config,
    }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folders", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = []
    for folder in args.folders:
        result = audit(folder)
        results.append(result)
        print(
            json.dumps(
                {k: v for k, v in result.items() if k not in ("configuration", "ui_counts")},
                ensure_ascii=False,
            ),
            flush=True,
        )
        gc.collect()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
