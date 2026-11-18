"""Search complete reset schedules in a recorded static pocket; no live input."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from numba import njit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jumpbot.physics import Physics


@njit(cache=True)
def shoot(
    initial,
    boxes,
    speed,
    jump,
    gravity,
    delays,
    directions,
    height,
    margin,
    goal_y,
    successful=None,
):
    best = -1
    best_time = 1e9
    goals = 0
    for n in range(len(delays)):
        x, y, vx, vy = initial[:4]
        peak = min(height / 2, y)
        t = 0.0
        event = delays[n, 0]
        k = 0
        while t < 6.0:
            if t + 1e-8 >= event:
                vx = directions[n, k] * speed
                vy = -jump
                k += 1
                event = event + delays[n, k] if k < delays.shape[1] else 1e9
            dt = min(0.01, event - t, 6.0 - t)
            if dt < 1e-9:
                continue
            x = min(420.0, max(0.0, x + vx * dt))
            y += vy * dt + 0.5 * gravity * dt * dt
            vy += gravity * dt
            t += dt
            peak = min(peak, y)
            if x <= 0 or x >= 420:
                vx = 0.0
            if y + max(0.0, height / 2 - peak) > height - initial[5] / 2 - max(25.0, margin):
                break
            collided = False
            for b in boxes:
                if (
                    x + initial[4] / 2 + margin > b[0]
                    and x - initial[4] / 2 - margin < b[0] + b[2]
                    and y + initial[5] / 2 + margin > b[1]
                    and y - initial[5] / 2 - margin < b[1] + b[3]
                ):
                    collided = True
                    break
            if collided:
                break
            if y < goal_y:
                if successful is not None:
                    successful[n] = True
                goals += 1
                if t < best_time:
                    best_time = t
                    best = n
                break
    return best, best_time, goals


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--time", type=float, required=True)
    parser.add_argument("--samples", type=int, default=200000)
    parser.add_argument("--goal-y", type=float, default=25)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--stress-trials",
        type=int,
        default=0,
        help="Retest nominal solutions with position, velocity and input perturbations",
    )
    args = parser.parse_args()
    if args.samples < 1 or args.stress_trials < 0:
        parser.error("--samples must be positive and --stress-trials must be nonnegative")
    rows = [json.loads(line) for line in (args.folder / "frames.jsonl").open(encoding="utf-8")]
    row = min((r for r in rows if r.get("state")), key=lambda r: abs(r["t"] - args.time))
    s = row["state"]
    boxes = np.array(s["obstacles"])
    if s["shapes"] or np.any(abs(boxes[:, 4:]) > 1):
        raise ValueError("Static rectangles only")
    initial = np.array([s[k] for k in ["x", "y", "vx", "vy", "w", "h"]], dtype=float)
    p = Physics(**json.loads((args.folder / "configuration.json").read_text())["physics"])
    rng = np.random.default_rng(142)
    delays = rng.uniform(0.18, 1.15, (args.samples, 10))
    short = rng.uniform(0.18, 0.45, delays.shape)
    delays = np.where(rng.random(delays.shape) < 0.6, short, delays)
    target_y = rng.uniform(max(s["y"] + 5, 580), 690, args.samples)
    delays[:, 0] = np.maximum(
        0.035,
        (-s["vy"] + np.sqrt(np.maximum(0, s["vy"] ** 2 + 2 * p.gravity * (target_y - s["y"]))))
        / p.gravity,
    )
    directions = rng.choice(np.array([-1, 1]), delays.shape)
    toward = -1 if s["x"] > 210 else 1
    directions[:, :3] = toward
    height = round(row["roi"][3] * 420 / row["roi"][2])
    successful = np.zeros(args.samples, dtype=np.bool_) if args.stress_trials else None
    best, t, goals = shoot(
        initial,
        boxes,
        p.speed,
        p.jump,
        p.gravity,
        delays,
        directions,
        height,
        8.0,
        args.goal_y,
        successful,
    )
    result = {
        "source_t": row["t"],
        "samples": args.samples,
        "goal_trajectories": goals,
        "best_time": t if best >= 0 else None,
        "delays": delays[best].tolist() if best >= 0 else [],
        "directions": directions[best].tolist() if best >= 0 else [],
        "initial": initial.tolist(),
        "boxes": boxes.tolist(),
        "physics": vars(p),
        "height": height,
    }
    if args.stress_trials:
        candidates = np.flatnonzero(successful)
        outcomes = np.zeros((args.stress_trials, len(candidates)), dtype=np.bool_)
        candidate_delays, candidate_directions = delays[candidates], directions[candidates]
        for success in outcomes:
            changed = initial.copy()
            changed[1] += rng.uniform(-6, 6)
            changed[3] += rng.uniform(-35, 35)
            # Perturb absolute event times, without accumulating timing drift.
            times = np.cumsum(candidate_delays, axis=1) + rng.uniform(-0.012, 0.012, 10)
            perturbed = np.diff(times, axis=1, prepend=0)
            shoot(
                changed,
                boxes,
                p.speed,
                p.jump,
                p.gravity,
                perturbed,
                candidate_directions,
                height,
                8.0,
                args.goal_y,
                success,
            )
        counts = outcomes.sum(axis=0)
        strongest = candidates[np.argmax(counts)] if len(candidates) else -1
        result["stress"] = {
            "trials": args.stress_trials,
            "position_range_px": [-6, 6],
            "vertical_velocity_range_px_per_s": [-35, 35],
            "absolute_input_time_range_ms": [-12, 12],
            "survived_all": int(np.count_nonzero(counts == args.stress_trials)),
            "fastest_successes": int(counts[np.flatnonzero(candidates == best)[0]])
            if best >= 0
            else 0,
            "best_successes": int(counts.max(initial=0)),
            "delays": delays[strongest].tolist() if strongest >= 0 else [],
            "directions": directions[strongest].tolist() if strongest >= 0 else [],
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ["initial", "boxes", "physics"]}))
