"""Exercise closed-loop control on geometry reconstructed from a saved live run.

This is a model regression, not evidence of a counterfactual live score.
"""

import argparse
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jumpbot.fast_planner import FastPlanner
from jumpbot.physics import Physics, Tracker
from jumpbot.vision import Observation, Rect, camera_shift


def extract(path):
    rows = []
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("ui") == "decline_continue":
            break
        rows.append(row)
    groups = []
    scroll = 0.0
    previous = None
    for row in rows:
        if not row.get("player"):
            continue
        obstacles = [Rect(**b) for b in row["obstacles"]]
        if previous:
            shift, _ = camera_shift(previous, obstacles)
            scroll += shift
        previous = obstacles
        bars = [b for b in obstacles if (b.x < 4 or b.x + b.w > 416) and 30 <= b.h <= 42]
        if bars:
            scroll = float(
                np.median([b.y - (165 + 420 * round((b.y - scroll - 165) / 420)) for b in bars])
            )
        for b in obstacles:
            if b.y < 140 or b.y + b.h > 725 or b.h < 25 or b.w < 20:
                continue
            is_bar = b.x < 4 or b.x + b.w > 416
            if not is_bar and not (0.8 < b.w / b.h < 1.25):
                continue
            world = np.array([b.x, b.y - scroll, b.w, b.h], dtype=float)
            group = next(
                (g for g in groups if np.max(abs(np.median(g, axis=0) - world)) < 12), None
            )
            if group is None:
                groups.append([world])
            else:
                group.append(world)
    result = [np.median(g, axis=0) for g in groups if len(g) >= 3]
    return np.array(result)


def simulate(obstacles, physics, step, beam, interval, render_delay, click_cost, duration):
    planner = FastPlanner(physics, step=step, beam=beam, click_cost=click_cost)
    planner.warmup()
    tracker = Tracker(physics)
    image = np.zeros((744, 420, 3), np.uint8)
    x, y, vx, vy = 210.0, 527.0, 0.0, 0.0
    peak = 372.0
    dt, t, next_plan = 0.005, 0.0, 0.04
    history = deque(maxlen=100)
    pending = []
    commands = []
    trace = []
    hit = None
    while t < duration:
        history.append((t, x, y, peak))
        if t >= next_plan:
            captured, ox, oy, op = min(history, key=lambda v: abs(v[0] - (t - render_delay)))
            scroll = max(0, 372 - op)
            visible = []
            for bx, by, bw, bh in obstacles:
                sy = by + scroll
                if sy + bh >= 6 and sy < 738:
                    top = max(0, sy)
                    visible.append(Rect(bx, top, bw, min(744, sy + bh) - top))
            obs = Observation(
                image, Rect(ox - 15, oy + scroll - 16, 30, 32), visible, True, (0, 0, 420, 744)
            )
            state = tracker.update(obs, captured)
            if state:
                decision = planner.plan(
                    state, 744, latency=render_delay + 0.025, control_interval=interval
                )
                trace.append((t, x, y, decision.action, decision.predicted_horizon))
                if decision.action and not pending:
                    pending.append((t + 0.025, decision.action))
                    tracker.clicked(decision.action, t + 0.013)
                    commands.append((t, decision.action))
            next_plan = t + interval
        if pending and t >= pending[0][0]:
            _, action = pending.pop(0)
            vx, vy = action * physics.speed, -physics.jump
        x += vx * dt
        y += vy * dt + 0.5 * physics.gravity * dt**2
        vy += physics.gravity * dt
        peak = min(peak, y)
        t += dt
        collided = [
            b.tolist()
            for b in obstacles
            if x + 15 > b[0] and x - 15 < b[0] + b[2] and y + 16 > b[1] and y - 16 < b[1] + b[3]
        ]
        if collided or x < 15 or x > 405 or y + max(0, 372 - peak) > 728:
            hit = {"obstacles": collided, "x": x, "y": y}
            break
        if y < min(obstacles[:, 1]) - 40:
            break
    return {
        "t": round(t, 3),
        "climb": round(527 - peak),
        "hit": hit,
        "commands": len(commands),
        "trace": trace,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--step", type=float, default=0.025)
    parser.add_argument("--beam", type=int, default=600)
    parser.add_argument("--interval", type=float, default=0.04)
    parser.add_argument("--render-delay", type=float, default=0.04)
    parser.add_argument("--click-cost", type=float, default=12)
    parser.add_argument("--duration", type=float, default=120)
    args = parser.parse_args()
    cache = args.recording / "course.npy"
    if cache.exists():
        obstacles = np.load(cache)
    else:
        obstacles = extract(args.recording / "frames.jsonl")
        np.save(cache, obstacles)
    print(f"Reconstructed {len(obstacles)} obstacles", flush=True)
    result = simulate(
        obstacles,
        Physics.load("jumpbot/assets/physics_windows.json"),
        args.step,
        args.beam,
        args.interval,
        args.render_delay,
        args.click_cost,
        args.duration,
    )
    (args.recording / "simulation.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "trace"}, indent=2))
