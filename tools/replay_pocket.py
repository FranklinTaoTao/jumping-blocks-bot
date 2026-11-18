"""Closed-loop replay of a static recorded pocket using ideal physics.

This deliberately excludes capture noise, camera easing, unknown obstacles and
moving hazards. It can expose planning stalls; it cannot prove a live score.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jumpbot.fast_planner import FastPlanner
from jumpbot.physics import Physics, State


def replay(folder, captured, horizon, beam, guide, duration):
    rows = [json.loads(line) for line in (folder / "frames.jsonl").open(encoding="utf-8")]
    row = min((r for r in rows if r.get("state")), key=lambda r: abs(r["t"] - captured))
    initial = row["state"]
    world = np.array(initial["obstacles"])
    if initial["shapes"] or np.any(abs(world[:, 4:]) > 1):
        raise ValueError("This replay only supports static rectangles")
    config = json.loads((folder / "configuration.json").read_text(encoding="utf-8"))
    physics = Physics(**config["physics"])
    planner = FastPlanner(
        physics,
        horizon=horizon,
        step=0.025,
        beam=beam,
        cooldown=0.18,
        jump_guidance=guide,
    )
    x, y, vx, vy = [initial[key] for key in ["x", "y", "vx", "vy"]]
    height = round(row["roi"][3] * 420 / row["roi"][2])
    anchor = height / 2
    peak = min(anchor, y)
    t, next_plan, wall = 0.0, 0.0, 0.0
    last_effect = -initial["age_since_click"]
    pending = None
    commands, trace = [], []
    hit = False
    while t < duration:
        offset = max(0, anchor - peak)
        if t >= next_plan and pending is None:
            visible = world.copy()
            visible[:, 1] += offset
            visible = visible[(visible[:, 1] + visible[:, 3] > 0) & (visible[:, 1] < height)]
            state = State(
                float(x),
                y + offset,
                vx,
                vy,
                initial["w"],
                initial["h"],
                t - last_effect,
                True,
                visible,
                wall_dwell=wall,
            )
            decision = planner.plan(state, height, latency=0.035, control_interval=0.04)
            trace.append(
                [
                    round(t, 3),
                    round(float(x), 1),
                    round(y, 1),
                    decision.action,
                    round(decision.predicted_horizon, 2),
                ]
            )
            if decision.action:
                pending = (t + 0.035, decision.action)
                last_effect = t + 0.035
                commands.append([round(t, 3), decision.action])
            next_plan = t + 0.04
        if pending is not None and t >= pending[0]:
            vx, vy = pending[1] * physics.speed, -physics.jump
            pending = None
        x = float(np.clip(x + vx * 0.005, 0, 420))
        y += vy * 0.005 + 0.5 * physics.gravity * 0.005**2
        vy += physics.gravity * 0.005
        peak = min(peak, y)
        if x <= 0 or x >= 420:
            vx = 0.0
            wall += 0.005
        else:
            wall = 0.0
        hit = any(
            x + initial["w"] / 2 > b[0]
            and x - initial["w"] / 2 < b[0] + b[2]
            and y + initial["h"] / 2 > b[1]
            and y - initial["h"] / 2 < b[1] + b[3]
            for b in world
        )
        hit = hit or y + max(0, anchor - peak) > height - initial["h"] / 2
        t += 0.005
        if hit or y < 25:
            break
    return {
        "source_t": row["t"],
        "horizon": horizon,
        "beam": beam,
        "guide": guide,
        "seconds": round(t, 2),
        "climb": round(initial["y"] - peak, 1),
        "hit": bool(hit),
        "escaped_visible_scene": bool(y < 25 and not hit),
        "commands": commands,
        "trace": trace,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--time", type=float, required=True)
    parser.add_argument("--horizon", type=float, default=3.0)
    parser.add_argument("--beam", type=int, default=600)
    parser.add_argument("--guide", action="store_true")
    parser.add_argument("--duration", type=float, default=12.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = replay(
        args.folder,
        args.time,
        args.horizon,
        args.beam,
        args.guide,
        args.duration,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ["trace", "commands"]}))
