"""Record controlled opening jumps and fit Windows physics without screen files."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jumpbot.score import ScoreReader
from jumpbot.ui import ScreenHandler
from jumpbot.vision import Detector, Rect, camera_shift


def collect(path, arcs):
    if sys.platform == "win32":
        from jumpbot.windows import GameWindow
    else:
        from jumpbot.x11 import GameWindow

    game = GameWindow()
    detector, ui, reader = Detector(), ScreenHandler(), ScoreReader()
    start = time.perf_counter()
    rows, clicks = [], []
    next_click = None
    try:
        game.focus()
        time.sleep(0.2)
        while time.perf_counter() - start < 25:
            tick = time.perf_counter()
            if game.stopped() or not game.focused():
                raise RuntimeError("Probe stopped or game lost focus")
            frame = game.capture()
            captured = (tick + time.perf_counter()) / 2 - start
            obs = detector.observe(frame)
            if obs and obs.active and obs.player:
                rows.append(
                    {
                        "t": captured,
                        "player": vars(obs.player),
                        "obstacles": [vars(b) for b in obs.obstacles],
                    }
                )
                if next_click is None:
                    if reader.read(obs.image) != 0:
                        raise RuntimeError("Probe requires a fresh round at score zero")
                    next_click = time.perf_counter() + 0.2
                if len(clicks) < arcs and time.perf_counter() >= next_click:
                    direction = -1 if obs.player.cx > 250 else 1
                    if game.click(direction, obs.roi):
                        clicks.append({"t": game.last_press_time - start, "direction": direction})
                        next_click = game.last_press_time + 1.02
                        print(f"Probe jump {len(clicks)}/{arcs}: {direction}", flush=True)
                if len(clicks) == arcs and time.perf_counter() - start > clicks[-1]["t"] + 0.95:
                    break
            elif clicks:
                raise RuntimeError("Probe lost player tracking")
            else:
                action = ui.inspect(frame)
                if ui.ready(action, tick) and action.kind in (
                    "restart",
                    "decline_continue",
                    "decline_helmet",
                    "close_ad",
                ):
                    if game.click_at(action.x, action.y):
                        ui.clicked(time.perf_counter())
            time.sleep(max(0, 1 / 60 - (time.perf_counter() - tick)))
    finally:
        game.close()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"rows": rows, "clicks": clicks}, indent=2))
    if len(clicks) < arcs:
        raise RuntimeError("Probe did not complete all requested jumps")


def fit(path):
    data = json.loads(path.read_text())
    rows, clicks = data["rows"], data["clicks"]
    t = np.array([r["t"] for r in rows])
    x = np.array([r["player"]["x"] + r["player"]["w"] / 2 for r in rows])
    candidates = []
    for delay in np.arange(0, 0.121, 0.001):
        distance = np.zeros_like(t)
        previous = 0
        for click in clicks:
            distance += (click["direction"] - previous) * np.maximum(t - click["t"] - delay, 0)
            previous = click["direction"]
        design = np.column_stack([distance, np.ones(len(t))])
        speed, offset = np.linalg.lstsq(design, x, rcond=None)[0]
        error = x - design @ [speed, offset]
        candidates.append((float(np.mean(error**2)), delay, speed))
    error, delay, speed = min(candidates)
    scroll = 0
    samples = []
    for i, row in enumerate(rows):
        if i:
            shift, _ = camera_shift(
                [Rect(**b) for b in rows[i - 1]["obstacles"]],
                [Rect(**b) for b in row["obstacles"]],
            )
            scroll += shift
        arc = sum(c["t"] + delay <= row["t"] for c in clicks) - 1
        if arc < 0:
            continue
        age = row["t"] - clicks[arc]["t"] - delay
        if 0.06 < age < 0.94:
            p = row["player"]
            samples.append((arc, age, p["y"] + p["h"] / 2 - scroll))
    samples = np.array(samples)
    design = np.zeros((len(samples), len(clicks) + 2))
    design[:, 0] = -samples[:, 1]
    design[:, 1] = samples[:, 1] ** 2 / 2
    for i in range(len(clicks)):
        design[samples[:, 0] == i, i + 2] = 1
    train = np.arange(len(samples)) % 5 != 0
    valid = train.copy()
    for _ in range(4):
        solution = np.linalg.lstsq(design[valid], samples[valid, 2], rcond=None)[0]
        residual = samples[:, 2] - design @ solution
        valid = train & (abs(residual) < 4)
    result = {
        "physics": {"speed": float(speed), "jump": solution[0], "gravity": solution[1]},
        "click_to_visible_delay_ms": round(delay * 1000, 2),
        "horizontal_rmse": round(error**0.5, 3),
        "held_out_vertical_rmse": round(float(np.mean(residual[~train] ** 2) ** 0.5), 3),
        "samples": len(samples),
        "arcs": len(clicks),
        "source": str(path),
    }
    print(json.dumps(result, indent=2))
    path.with_suffix(".fit.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--arcs", type=int, default=4)
    parser.add_argument("--fit-only", action="store_true")
    args = parser.parse_args()
    if not args.fit_only:
        collect(args.output, args.arcs)
    fit(args.output)
