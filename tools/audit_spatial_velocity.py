"""Compare velocity fits on matched, uninterrupted arcs from live telemetry."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jumpbot.motion import Shape
from jumpbot.physics import Physics, Tracker
from jumpbot.vision import Observation, Rect


def audit(folder):
    config = json.loads((folder / "configuration.json").read_text(encoding="utf-8"))
    physics = Physics(**config["physics"])
    reset_phase = 0 if config["arguments"].get("input_lag") else 0.012
    trackers = [
        Tracker(physics, reset_phase=reset_phase, spatial_velocity=enabled)
        for enabled in (False, True)
    ]
    with (folder / "frames.jsonl").open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    times = np.array([row["t"] for row in rows])
    presses = np.cumsum([bool(row["sent"]) for row in rows])
    history = []
    image = np.empty((1, 1, 3), dtype=np.uint8)
    for row in rows:
        observation = None
        if row.get("player") and row.get("state"):
            observation = Observation(
                image,
                Rect(**row["player"]),
                [Rect(**box) for box in row["obstacles"]],
                True,
                tuple(row["roi"]),
            )
        samples = []
        for tracker in trackers:
            state = tracker.update(observation, row["t"])
            samples.append((state.vy, state.y - tracker.scroll) if state and state.ready else None)
            if row["sent"]:
                tracker.clicked(row["sent"], row["telemetry"]["input_effect_at"])
        history.append(samples)

    errors = []
    for i, row in enumerate(rows):
        j = int(np.searchsorted(times, times[i] + 0.10))
        state = row.get("state")
        if (
            j >= len(rows)
            or row["sent"]
            or presses[j - 1] != presses[i]
            or not all(history[i])
            or not all(history[j])
            or not state
            or state["age_since_click"] < 0.15
            or not 5 < state["x"] < 415
            or not 5 < rows[j]["state"]["x"] < 415
        ):
            continue
        if any(not (other.get("state") or {}).get("ready") for other in rows[i : j + 1]):
            continue
        dt = times[j] - times[i]
        if dt > 0.16:
            continue
        gravity_offset = 0.5 * physics.gravity * dt**2
        errors.append(
            [
                end[1] - start[1] - start[0] * dt - gravity_offset
                for start, end in zip(history[i], history[j])
            ]
        )
    if not errors:
        raise ValueError(f"No uninterrupted arcs in {folder}")
    errors = np.asarray(errors)
    return {
        "name": folder.name,
        "samples": len(errors),
        "order": ["time", "spatial"],
        "rms_px": np.sqrt(np.mean(errors**2, axis=0)).round(2).tolist(),
        "p90_absolute_px": np.percentile(abs(errors), 90, axis=0).round(2).tolist(),
    }


def audit_frame_clock(folder):
    config = json.loads((folder / "configuration.json").read_text(encoding="utf-8"))
    physics = Physics(**config["physics"])
    reset_phase = 0 if config["arguments"].get("input_lag") else 0.012
    trackers = [
        Tracker(
            physics, reset_phase=reset_phase, skip_repeated_frames=True, recover_frame_clock=enabled
        )
        for enabled in (False, True)
    ]
    click = 0
    poses = []
    with (folder / "frames.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if not line.endswith("\n"):
                break
            row = json.loads(line)
            player = Rect(**row["player"]) if row["player"] else None
            shapes = [
                Shape(**(shape | {"points": np.asarray(shape["points"])}))
                for shape in row.get("shapes", [])
            ]
            obs = Observation(
                None,
                player,
                [Rect(**b) for b in row["obstacles"]],
                row["active"],
                row["roi"],
                shapes,
            )
            t = row["t"]
            base, corrected = [tracker.update(obs, t) for tracker in trackers]
            if (
                base
                and corrected
                and base.ready
                and row["state"]
                and 6 < base.x < 414
                and abs(base.vx) > 100
            ):
                poses.append(
                    (t, click, base.x, base.y - trackers[0].scroll, base.vx, base.vy, corrected.vy)
                )
            if row["sent"]:
                click += 1
                for tracker in trackers:
                    tracker.clicked(row["sent"], row["telemetry"]["input_effect_at"])
    changed = 0
    errors = []
    for i, (t, click, x, y, vx, base_vy, corrected_vy) in enumerate(poses):
        if abs(base_vy - corrected_vy) <= 10:
            continue
        changed += 1
        for future, next_click, next_x, next_y, *_ in poses[i + 1 : i + 12]:
            if next_click != click or future - t > 0.15:
                break
            travel = (next_x - x) / vx
            if 0.05 <= travel <= 0.12:
                measured = (next_y - y) / travel - 0.5 * physics.gravity * travel
                if abs(measured) <= 900:
                    errors.append((base_vy - measured, corrected_vy - measured))
                break
    errors = np.asarray(errors)
    return {
        "name": folder.name,
        "ready_motion_samples": len(poses),
        "changed_over_10_px_s": changed,
        "validated_coasts": len(errors),
        "order": ["time", "frame_clock"],
        "velocity_rms_px_s": np.sqrt(np.mean(errors**2, axis=0)).round(1).tolist()
        if len(errors)
        else None,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folders", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--frame-clock",
        action="store_true",
        help="Compare the frame-clock correction on recorded timing anomalies",
    )
    args = parser.parse_args()
    compare = audit_frame_clock if args.frame_clock else audit
    reports = [compare(folder) for folder in args.folders]
    args.output.write_text(json.dumps(reports, indent=2) + "\n", encoding="utf-8")
    for report in reports:
        print(json.dumps(report))
