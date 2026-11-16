"""Command-line entry points for live control and offline analysis."""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from .physics import Physics, Tracker, calibrate
from .planner import Planner
from .render import annotate
from .ui import ScreenHandler
from .vision import Detector, camera_shift


def parse_roi(value):
    """Parse a board rectangle before opening a live display."""
    message = "ROI must be x,y,width,height with nonnegative coordinates and positive dimensions"
    try:
        roi = tuple(map(int, value.split(",")))
    except ValueError as error:
        raise argparse.ArgumentTypeError(message) from error
    if len(roi) != 4 or min(roi[:2]) < 0 or min(roi[2:]) <= 0:
        raise argparse.ArgumentTypeError(message)
    return roi


def video_frames(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            # Container timestamps matter: recordings can have variable FPS.
            yield cap.get(cv2.CAP_PROP_POS_MSEC) / 1000, frame
    finally:
        cap.release()


def run_calibration(args):
    detector = Detector(args.roi)
    previous = None
    previous_t = 0
    scroll = 0.0
    samples = []
    for t, frame in video_frames(args.video):
        obs = detector.observe(frame)
        if obs and obs.active and obs.player:
            shift, reliable = (
                camera_shift(previous.obstacles, obs.obstacles)
                if previous and 0 < t - previous_t < 0.15
                else (0, False)
            )
            if previous is None:
                scroll = 0.0
            scroll += shift
            if reliable:
                samples.append((t, obs.player.cx, obs.player.cy - scroll))
            previous, previous_t = obs, t
        else:
            previous = None
    result = calibrate(samples)
    result["source"] = Path(args.video).name
    result["tracked_samples"] = len(samples)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def get_components(args):
    physics_path = args.physics or Path(__file__).parent / "assets" / "physics.json"
    physics = Physics.load(physics_path)
    planner_class = Planner
    if getattr(args, "planner", "native") == "native":
        from .fast_planner import FastPlanner

        planner_class = FastPlanner
    return (
        Detector(args.roi),
        Tracker(physics),
        planner_class(physics, click_cost=getattr(args, "click_cost", 12.0)),
    )


def run_replay(args):
    detector, tracker, planner = get_components(args)
    if hasattr(planner, "warmup"):
        print("Warming up compiled trajectory search...", flush=True)
        planner.warmup()
    writer = None
    counts = {
        "frames": 0,
        "active_frames": 0,
        "tracked_frames": 0,
        "ready_frames": 0,
        "left": 0,
        "right": 0,
        "wait": 0,
        "immediate_failures": 0,
    }
    durations = []
    try:
        for t, frame in video_frames(args.video):
            start = time.perf_counter()
            obs = detector.observe(frame)
            state = tracker.update(obs, t)
            decision = planner.plan(state, obs.image.shape[0]) if state else None
            durations.append((time.perf_counter() - start) * 1000)
            counts["frames"] += 1
            counts["active_frames"] += bool(obs and obs.active)
            counts["tracked_frames"] += state is not None
            counts["ready_frames"] += bool(state and state.ready)
            if decision:
                counts[{-1: "left", 1: "right", 0: "wait"}[decision.action]] += 1
                counts["immediate_failures"] += decision.reason.startswith("no collision")
            if args.output:
                # Fixed output dimensions even before a board is found.
                if writer is None:
                    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
                    writer = cv2.VideoWriter(
                        args.output, cv2.VideoWriter_fourcc(*"mp4v"), 30, (420, 750)
                    )
                    if not writer.isOpened():
                        raise RuntimeError("Could not create annotated output video.")
                image = (
                    annotate(obs, state, decision, f"REPLAY {t:.2f}s (suggestions only)")
                    if obs
                    else np.zeros((750, 420, 3), dtype=np.uint8)
                )
                writer.write(cv2.resize(image, (420, 750)))
    finally:
        if writer:
            writer.release()
    if not durations:
        raise RuntimeError("The video contains no decodable frames.")
    counts["processing_ms_p50_p95"] = np.percentile(durations, [50, 95]).round(2).tolist()
    counts["note"] = (
        "Replay checks perception and suggested actions, not counterfactual game performance."
    )
    print(json.dumps(counts, indent=2))


def run_live(args):
    from .x11 import GameWindow

    game = GameWindow(args.title)
    if args.focus:
        game.focus()
        time.sleep(0.2)
    detector, tracker, planner = get_components(args)
    if hasattr(planner, "warmup"):
        print("Warming up compiled trajectory search...", flush=True)
        planner.warmup()
    start = time.monotonic()
    last_report = start - 2
    last_shape = None
    frames = clicks = 0
    timings = []
    ui = ScreenHandler()
    from .score import ScoreReader

    score_reader = ScoreReader()
    score = best_score = 0
    score_streak = 0
    restarts = 0
    planning_time = 0.025
    recorder = None
    if args.record:
        from .diagnostics import Recorder

        recorder = Recorder(args.record)
    mode = "PLAY" if args.play else "PREVIEW"
    print(
        f"{mode}: {game.title!r}, window {hex(game.window.id)}. F8 or mouse in top-left screen corner stops.",
        flush=True,
    )
    print(
        "Focus the game to arm control. Losing focus pauses clicks. Known dialogs are declined; score screen restarts."
        if args.play
        else "Preview only: no mouse input. Add --play to enable control.",
        flush=True,
    )
    try:
        while time.monotonic() - start < args.duration:
            tick = time.monotonic()
            if game.stopped():
                print("Stop requested.", flush=True)
                break
            if args.play and not game.focused():
                tracker.update(None, tick)
                time.sleep(0.03)
                continue
            capture_start = time.monotonic()
            frame = game.capture()
            captured = (capture_start + time.monotonic()) / 2
            if last_shape is not None and frame.shape != last_shape:
                detector, tracker, planner = get_components(args)
            last_shape = frame.shape
            obs = detector.observe(frame)
            state = tracker.update(obs, captured)
            read_score = score_reader.read(obs.image) if obs else None
            telemetry = {"score": read_score, "capture_ms": (captured - capture_start) * 2000}
            if read_score is not None:
                score_streak = score_streak + 1 if read_score == score else 1
                score = read_score
                if score_streak >= 3:
                    best_score = max(best_score, score)
                if args.target and score >= args.target and score_streak >= 5 and state:
                    target_dir = Path(args.record or "artifacts/success")
                    target_dir.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(target_dir / "target-reached.png"), frame)
                    (target_dir / "target-reached.json").write_text(
                        json.dumps({"score": score, "t": captured - start, "target": args.target})
                    )
                    print(f"TARGET REACHED: live score {score}", flush=True)
                    break
            decision = None
            sent = 0
            ui_action = ui.inspect(frame) if not (obs and obs.active and obs.player) else None
            if ui.ready(ui_action, captured) and args.play:
                if ui_action.kind == "restart" and args.no_auto_restart:
                    print("Score screen reached; automatic restart disabled.", flush=True)
                    break
                if ui_action.kind == "restart" and args.rounds and restarts >= args.rounds:
                    print(f"Completed {restarts} requested rounds.", flush=True)
                    break
                if time.monotonic() - capture_start < 0.15 and game.click_at(
                    ui_action.x, ui_action.y
                ):
                    print(f"UI: {ui_action.kind} ({ui_action.confidence:.3f})", flush=True)
                    ui.clicked(time.monotonic())
                    tracker.update(None, captured)
                    if ui_action.kind == "restart":
                        restarts += 1
                    if recorder:
                        telemetry["ui_pressed_at"] = game.last_press_time - start
                        recorder.add(
                            captured - start,
                            frame,
                            obs,
                            state,
                            decision,
                            ui=ui_action.kind,
                            telemetry=telemetry,
                        )
                    time.sleep(0.15)
                    continue
            if state:
                age = time.monotonic() - captured
                plan_start = time.monotonic()
                latency = age + planning_time + 0.012
                decision = planner.plan(state, obs.image.shape[0], latency=latency)
                measured_planning_time = time.monotonic() - plan_start
                planning_time = 0.6 * planning_time + 0.4 * measured_planning_time
                telemetry.update(
                    {
                        "planned_latency_ms": latency * 1000,
                        "planning_ms": measured_planning_time * 1000,
                    }
                )
                # Never deliver an action based on an old frame.
                if args.play and decision.action and time.monotonic() - capture_start < 0.09:
                    if game.click(decision.action, obs.roi):
                        tracker.clicked(decision.action, game.last_press_time)
                        clicks += 1
                        sent = decision.action
                        telemetry["input_pressed_at"] = game.last_press_time - start
            if recorder:
                telemetry["frame_age_ms"] = (time.monotonic() - captured) * 1000
                recorder.add(
                    captured - start,
                    frame,
                    obs,
                    state,
                    decision,
                    sent,
                    ui=ui_action.kind if ui_action else None,
                    telemetry=telemetry,
                )
            if args.show and obs:
                cv2.imshow("Jumpbot preview (F8 stops)", annotate(obs, state, decision, mode))
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
            elapsed = time.monotonic() - tick
            timings.append(elapsed * 1000)
            frames += 1
            if tick - last_report > 1:
                label = (
                    {-1: "LEFT", 0: "WAIT", 1: "RIGHT"}[decision.action] if decision else "NO TRACK"
                )
                ui_label = ui_action.kind if ui_action else "unknown/wait" if not state else "game"
                print(
                    f"{time.monotonic() - start:5.1f}s {elapsed * 1000:5.1f}ms {label} score={score} best={best_score} clicks={clicks} ui={ui_label} "
                    f"focus={bool(game.focused())} ready={bool(state and state.ready)}",
                    flush=True,
                )
                last_report = tick
            time.sleep(max(0, 1 / args.fps - (time.monotonic() - tick)))
    finally:
        if recorder:
            recorder.close()
        game.close()
        if args.show:
            cv2.destroyAllWindows()
    print(
        json.dumps(
            {
                "frames": frames,
                "clicks": clicks,
                "restarts": restarts,
                "best_score": best_score,
                "loop_ms_p50_p95": np.percentile(timings, [50, 95]).round(2).tolist()
                if timings
                else [],
            }
        )
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Classical-vision jump controller. Preview is the default."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ["calibrate", "replay", "live"]:
        p = sub.add_parser(name)
        p.add_argument(
            "--roi",
            type=parse_roi,
            help="Optional x,y,w,h board rectangle in input pixels",
        )
        if name != "live":
            p.add_argument("video")
        if name == "calibrate":
            p.add_argument("--output", default="artifacts/physics.json")
            p.set_defaults(run=run_calibration)
        else:
            p.add_argument(
                "--physics", help="Calibration JSON; defaults to the bundled calibration"
            )
            p.add_argument(
                "--click-cost",
                type=float,
                default=12.0,
                help="Penalty for unnecessary resets; larger values favor full arcs",
            )
            p.add_argument(
                "--planner",
                choices=["native", "python"],
                default="native",
                help="Compiled broad search or original reference search",
            )
            if name == "replay":
                p.add_argument("--output", help="Optional annotated MP4 output")
                p.set_defaults(run=run_replay)
            else:
                p.add_argument(
                    "--title", help="Game title substring if multiple game windows exist"
                )
                p.add_argument(
                    "--play", action="store_true", help="Enable actual left/right clicks"
                )
                p.add_argument(
                    "--show",
                    action="store_true",
                    help="Show OpenCV overlay; leave game focused to allow clicks",
                )
                p.add_argument(
                    "--record",
                    help="Directory for asynchronous diagnostic video and frame telemetry",
                )
                p.add_argument(
                    "--focus", action="store_true", help="Focus the game once at startup"
                )
                p.add_argument(
                    "--target",
                    type=int,
                    default=0,
                    help="Stop and save evidence at this live score",
                )
                p.add_argument("--duration", type=float, default=60)
                p.add_argument("--fps", type=float, default=60)
                p.add_argument(
                    "--rounds",
                    type=int,
                    default=0,
                    help="Stop after this many restarted rounds (0: duration only)",
                )
                p.add_argument(
                    "--no-auto-restart",
                    action="store_true",
                    help="Stop on a recognized score screen",
                )
                p.set_defaults(run=run_live)
    args = parser.parse_args(argv)
    cv2.setNumThreads(1)
    if args.command == "live" and (
        args.fps <= 0 or args.duration <= 0 or args.rounds < 0 or args.target < 0
    ):
        parser.error(
            "--fps and --duration must be positive; --rounds and --target must be nonnegative"
        )
    try:
        args.run(args)
    except (RuntimeError, ValueError, OSError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Stopped.")
    return 0
