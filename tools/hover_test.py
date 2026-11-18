"""Demonstrate 20 opening jumps and measure the existing calibration's forecasts."""

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jumpbot.diagnostics import Recorder
from jumpbot.physics import Physics, State
from jumpbot.planner import Decision
from jumpbot.score import ScoreReader
from jumpbot.ui import ScreenHandler
from jumpbot.vision import Detector, camera_shift


def estimate_phases(measurements, presses, physics, render_delay):
    samples = [m for m in measurements if m["jump"] >= 2]
    t = np.array([m["t"] + render_delay for m in samples])
    x, y = np.array([[m["x"], m["y"]] for m in samples]).T
    clicks = np.array([p["t"] for p in presses])
    directions = np.array([p["direction"] for p in presses])
    horizontal, vertical = [], []
    for phase in np.arange(-0.08, 0.121, 0.0005):
        age = np.maximum(t[:, None] - clicks[None, :] - phase, 0)
        predicted_x = physics.speed * (age @ np.diff(np.r_[0, directions]))
        residual = x - predicted_x
        horizontal.append((np.mean((residual - residual.mean()) ** 2), phase))
        events = clicks + phase
        arc = np.maximum(0, np.searchsorted(events, t) - 1)
        squared = np.r_[0, np.cumsum(np.diff(events) ** 2)][arc] + (t - events[arc]) ** 2
        predicted_y = -physics.jump * (t - events[0]) + physics.gravity * squared / 2
        residual = y - predicted_y
        vertical.append((np.mean((residual - residual.mean()) ** 2), phase))
    return float(min(horizontal)[1]), float(min(vertical)[1])


def run(directory, jumps, calibration=None, align_jumps=0, same_direction=None, focus=False):
    from jumpbot.windows import GameWindow

    game = GameWindow()
    if focus:
        game.focus()
        time.sleep(0.2)
    config = (
        json.loads(calibration.read_text())
        if calibration
        else {
            "physics": vars(
                Physics.load(Path(__file__).parents[1] / "jumpbot/assets/physics_windows.json")
            ),
            "render_delay": 0.040,
            "input_delay_x": 0.012,
            "input_delay_y": 0.012,
        }
    )
    physics = Physics(**config["physics"])
    render_delay = config["render_delay"]
    input_delay_x, input_delay_y = config["input_delay_x"], config["input_delay_y"]
    detector, reader, ui = Detector(), ScoreReader(), ScreenHandler()
    recorder = Recorder(directory)
    start = time.perf_counter()
    previous = None
    history = deque(maxlen=6)
    scroll, count = 0.0, 0
    last_press = -100.0
    launch = None
    direction = 1
    completed = False
    ready_since = None
    aligned = False
    measurements, presses = [], []
    print(f"Waiting for game focus; testing {jumps} jumps at one height...", flush=True)
    try:
        while time.perf_counter() - start < 120:
            tick = time.perf_counter()
            if game.stopped():
                raise RuntimeError("Hover test stopped")
            if not game.focused():
                if count:
                    raise RuntimeError("Game lost focus during hover test")
                time.sleep(0.03)
                continue
            frame = game.capture()
            captured = (game.last_capture_time or (tick + time.perf_counter()) / 2) - render_delay
            obs = detector.observe(frame)
            if not obs or not obs.active or not obs.player:
                if count:
                    if previous and captured - previous[0] < 0.08:
                        continue
                    raise RuntimeError("Player tracking was lost during hover test")
                action = ui.inspect(frame)
                if ui.ready(action, tick) and action.kind in (
                    "restart",
                    "decline_continue",
                    "decline_helmet",
                    "close_ad",
                ):
                    if game.click_at(action.x, action.y):
                        ui.clicked(time.perf_counter())
                time.sleep(0.015)
                continue
            if not count and reader.read(obs.image) != 0:
                previous = None
                history.clear()
                time.sleep(0.03)
                continue
            if ready_since is None:
                ready_since = tick
            if previous:
                shift, _ = camera_shift(previous[1].obstacles, obs.obstacles)
                scroll += shift
            p = obs.player
            world_y = p.cy - scroll
            history.append((captured, p.cx, world_y))
            vx = vy = 0.0
            if len(history) >= 3:
                samples = np.array(history)
                dt = samples[:, 0] - captured
                design = np.column_stack([dt, np.ones(len(dt))])
                vx = np.linalg.lstsq(design, samples[:, 1], rcond=None)[0][0]
                vy = np.linalg.lstsq(
                    design, samples[:, 2] - physics.gravity * dt**2 / 2, rcond=None
                )[0][0]
            elapsed = captured - last_press - input_delay_y
            elapsed_x = captured - last_press - input_delay_x
            error = None
            path = np.empty((0, 2))
            if launch is not None:
                predicted_x = launch[0] + direction * physics.speed * elapsed_x
                predicted_y = launch[1] - physics.jump * elapsed + physics.gravity * elapsed**2 / 2
                error = (p.cx - predicted_x, world_y - predicted_y)
                if elapsed >= 0.060:
                    measurements.append(
                        {
                            "jump": count,
                            "t": captured - start,
                            "age": elapsed,
                            "x": p.cx,
                            "y": world_y,
                            "predicted_x": predicted_x,
                            "predicted_y": predicted_y,
                            "error_x": error[0],
                            "error_y": error[1],
                        }
                    )
                future = np.linspace(max(0, elapsed), 1.05, 40)
                path = np.column_stack(
                    [
                        launch[0]
                        + direction * physics.speed * (future + input_delay_y - input_delay_x),
                        launch[1]
                        - physics.jump * future
                        + physics.gravity * future**2 / 2
                        + scroll,
                    ]
                )
            # Predict the launch point at input delivery, rather than reacting to old pixels.
            lead = time.perf_counter() - captured + input_delay_y
            future_y = p.cy + vy * lead + physics.gravity * lead**2 / 2
            first = count == 0 and len(history) >= 5 and tick - ready_since >= 0.4
            returned = count > 0 and elapsed > 0.65 and vy > 100 and future_y >= 550
            if returned and align_jumps and count == align_jumps and not aligned:
                input_delay_x, input_delay_y = estimate_phases(
                    measurements, presses, physics, render_delay
                )
                aligned = True
                print(
                    f"Aligned after {count} warm-up jumps: "
                    f"horizontal {input_delay_x * 1000:.1f}ms, vertical {input_delay_y * 1000:.1f}ms",
                    flush=True,
                )
            sent = 0
            if count < jumps and (first or returned):
                next_direction = same_direction or (1 if count == 0 else -direction)
                if game.click(next_direction, obs.roi):
                    press = game.last_press_time
                    lead = press + input_delay_y - captured
                    lead_x = press + input_delay_x - captured
                    launch = (
                        p.cx if first else p.cx + vx * lead_x,
                        world_y if first else world_y + vy * lead + physics.gravity * lead**2 / 2,
                    )
                    last_press = press
                    direction = next_direction
                    count += 1
                    returned = False
                    sent = direction
                    presses.append(
                        {
                            "jump": count,
                            "t": press - start,
                            "direction": direction,
                            "launch_x": launch[0],
                            "launch_y": launch[1],
                        }
                    )
                    history.clear()
                    print(
                        f"Jump {count:02}/{jumps}: {'right' if direction > 0 else 'left'}",
                        flush=True,
                    )
            label = f"CALIBRATION {count}/{jumps}: " + (
                f"error x={error[0]:+.1f}, y={error[1]:+.1f}px" if error else "starting"
            )
            state = State(p.cx, p.cy, vx, vy, p.w, p.h, ready=True)
            recorder.add(
                captured - start,
                frame,
                obs,
                state,
                Decision(0, path, label),
                sent,
                telemetry={"jump": count, "error": error},
            )
            previous = (captured, obs)
            if count == jumps and returned:
                completed = True
                break
            time.sleep(max(0, 1 / 60 - (time.perf_counter() - tick)))
    finally:
        recorder.close()
        game.close()
        directory.mkdir(parents=True, exist_ok=True)
        result = {
            "completed": completed,
            "jumps": count,
            "physics": vars(physics),
            "render_delay": render_delay,
            "input_delay_x": input_delay_x,
            "input_delay_y": input_delay_y,
            "capture": game.capture_kind,
            "warmup_jumps": align_jumps,
            "measurements": measurements,
            "presses": presses,
        }
        if measurements:
            errors = np.array([[m["error_x"], m["error_y"]] for m in measurements])
            result["rmse_xy"] = np.sqrt(np.mean(errors**2, axis=0)).tolist()
            result["absolute_error_p90_xy"] = np.percentile(abs(errors), 90, axis=0).tolist()
            if align_jumps:
                validation = np.array(
                    [[m["error_x"], m["error_y"]] for m in measurements if m["jump"] > align_jumps]
                )
                if len(validation):
                    result["validation_rmse_xy"] = np.sqrt(np.mean(validation**2, axis=0)).tolist()
                    result["validation_jumps"] = count - align_jumps
        (directory / "hover.json").write_text(json.dumps(result, indent=2) + "\n")
        print(
            json.dumps(
                {k: v for k, v in result.items() if k not in ("measurements", "presses")}, indent=2
            )
        )
    if not completed:
        raise RuntimeError("The requested number of full jumps was not completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--jumps", type=int, default=20)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--align-jumps", type=int, default=0)
    parser.add_argument("--same-direction", type=int, choices=(-1, 1))
    parser.add_argument("--focus", action="store_true")
    args = parser.parse_args()
    run(
        args.directory,
        args.jumps,
        args.calibration,
        args.align_jumps,
        args.same_direction,
        args.focus,
    )
