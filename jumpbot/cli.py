"""Command-line entry points for live control and offline analysis."""

import argparse
import json
import random
import sys
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from .controller import (
    INPUT_DEADLINE_RESERVE,
    MAX_INPUT_FRAME_AGE,
    OpeningApproach,
    ResponseTiming,
    WallStall,
)
from .physics import Physics, Tracker, calibrate
from .planner import Decision, Planner
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
    default_physics = "physics_windows.json" if sys.platform == "win32" else "physics.json"
    physics_path = args.physics or Path(__file__).parent / "assets" / default_physics
    physics = Physics.load(physics_path)
    skip_repeated = getattr(args, "skip_repeated_frames", False)
    if getattr(args, "policy_weights", None):
        from .learned import RLPlanner

        return (
            Detector(args.roi),
            Tracker(physics, reset_phase=0.0, skip_repeated_frames=skip_repeated),
            RLPlanner(args.policy_weights, acknowledge=args.command == "live"),
        )
    planner_class = Planner
    if getattr(args, "planner", "native") == "native":
        from .fast_planner import FastPlanner

        planner_class = FastPlanner
    search_options = {}
    if getattr(args, "horizon", None) is not None:
        search_options["horizon"] = args.horizon
    if getattr(args, "search_step", None) is not None:
        search_options["step"] = args.search_step
    if getattr(args, "beam", None) is not None:
        search_options["beam"] = args.beam
    if getattr(args, "cooldown", None) is not None:
        search_options["cooldown"] = args.cooldown
    if getattr(args, "pocket_penalty", 0):
        search_options["pocket_penalty"] = args.pocket_penalty
    if getattr(args, "narrow_gap_cooldown", None) is not None:
        search_options["narrow_gap_cooldown"] = args.narrow_gap_cooldown
    if getattr(args, "wall_approach", False):
        search_options["wall_approach"] = True
    if getattr(args, "coast_recovery", False):
        search_options["coast_recovery"] = True
    if getattr(args, "timed_gap_turns", False):
        search_options["timed_gap_turns"] = True
    if getattr(args, "route_rules", False):
        search_options["route_rules"] = True
    if getattr(args, "camera_tau", 0):
        if planner_class is Planner:
            raise RuntimeError("Camera smoothing currently requires --planner native")
        search_options["camera_tau"] = args.camera_tau
    if getattr(args, "vertical_uncertainty", 0):
        search_options["vertical_uncertainty"] = args.vertical_uncertainty
    if getattr(args, "jump_guidance", False):
        search_options["jump_guidance"] = True
    return (
        Detector(args.roi),
        Tracker(
            physics,
            adaptive=getattr(args, "adaptive_motion", False),
            spatial_velocity=getattr(args, "spatial_velocity", False),
            skip_repeated_frames=skip_repeated,
            recover_frame_clock=getattr(args, "recover_frame_clock", False),
            reset_phase=0.0 if getattr(args, "input_lag", 0) else 0.012,
        ),
        planner_class(physics, click_cost=getattr(args, "click_cost", 12.0), **search_options),
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
    native_vision = args.native_vision
    if sys.platform == "win32":
        from .windows import CaptureUnavailable, GameWindow

        game_options = {
            "batch_jumps": args.batch_jumps,
            "message_jumps": args.message_jumps,
            "native_vision": native_vision,
            "vision_roi": args.roi,
        }
        capture_errors = (CaptureUnavailable,)
    else:
        from .x11 import GameWindow

        game_options = {}
        capture_errors = ()

    detector, tracker, planner = get_components(args)
    opening_approach = (
        OpeningApproach(planner)
        if args.route_rules and args.wall_approach and args.timed_gap_turns
        else None
    )

    def plan_frame(state, height, captured, latency, interval, latest_effect):
        if opening_approach:
            return opening_approach.plan(
                state, height, captured, latency, interval, latest_effect=latest_effect
            )
        return planner.plan(state, height, latency, interval)

    game = GameWindow(args.title, **game_options)
    if hasattr(planner, "warmup"):
        print("Warming up compiled trajectory search...", flush=True)
        planner.warmup()
        if opening_approach:
            opening_approach.warmup()
    last_shape = None
    frames = clicks = 0
    timings = []
    ui = ScreenHandler()
    from .score import ScoreReader

    score_reader = None if native_vision else ScoreReader()
    score = best_score = 0
    round_best_score = None
    score_streak = 0
    restarts = 0
    last_saved_round = -1
    wall_stall = WallStall()
    response_timing = (
        ResponseTiming(tracker.physics.speed, args.input_lag, args.input_delay_range)
        if args.adaptive_input_delay
        else None
    )
    planning_time = 0.025
    delivery_slack = getattr(args, "delivery_slack", None)
    if delivery_slack is None:
        delivery_slack = 0.012
    planning_samples = deque(maxlen=30)
    capture_available = True
    focus_available = True
    recorder = None
    capture_timeline = None
    viewer = None
    release_rng = random.Random(142) if args.release_ab else None
    if args.record:
        from .diagnostics import CaptureTimeline, Recorder

        recorder = Recorder(args.record, video=not native_vision)
        capture_timeline = CaptureTimeline()
        (recorder.directory / "configuration.json").write_text(
            json.dumps(
                {
                    "arguments": {k: v for k, v in vars(args).items() if k != "run"},
                    "physics": asdict(tracker.physics),
                    "platform": sys.platform,
                    "capture_backend": getattr(game, "capture_kind", None),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    mode = "PLAY" if args.play else "PREVIEW"

    def output_frame(captured, frame, obs, state, decision, telemetry, sent=0, ui_kind=None):
        if viewer:
            view_start = time.perf_counter()
            viewer.publish(
                captured, obs, state, decision, f"{mode} | score {score} | sent {sent:+d}"
            )
            telemetry["preview_publish_ms"] = (time.perf_counter() - view_start) * 1000
        if recorder:
            telemetry["frame_age_ms"] = (time.perf_counter() - captured) * 1000
            recorder.add(
                captured - start, frame, obs, state, decision, sent, ui=ui_kind, telemetry=telemetry
            )

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
        if args.show:
            from .live_view import LiveView

            viewer = LiveView()
        if args.focus:
            game.focus()
            time.sleep(0.2)
        start = time.perf_counter()
        last_report = start - 2
        while time.perf_counter() - start < args.duration:
            if args.record and (Path(args.record) / "STOP").exists():
                print("Recording STOP file requested shutdown.", flush=True)
                break
            tick = time.perf_counter()
            if game.stopped() or (viewer and viewer.stop.is_set()):
                print("Stop requested.", flush=True)
                break
            if args.play and not game.focused():
                if capture_timeline:
                    capture_timeline.interrupted("out_of_focus")
                if focus_available:
                    print("Game input paused while the window is out of focus.", flush=True)
                focus_available = False
                tracker.update(None, tick)
                if response_timing:
                    response_timing.clear_observations()
                if hasattr(planner, "reset"):
                    planner.reset()
                if opening_approach:
                    opening_approach.reset()
                time.sleep(0.03)
                continue
            if not focus_available:
                print("Game focus restored.", flush=True)
                focus_available = True
            capture_start = time.perf_counter()
            try:
                sample = game.observe_native() if native_vision else None
                if native_vision and sample is None:
                    if capture_timeline:
                        capture_timeline.interrupted("no_new_frame")
                    # Poll for the next presentation without spending a full
                    # planning interval waiting after a nonblocking miss.
                    time.sleep(min(0.002, 1 / args.fps))
                    continue
                frame = None if native_vision else game.capture()
            except capture_errors as error:
                if capture_timeline:
                    capture_timeline.interrupted("capture_error")
                if capture_available:
                    print(f"Waiting for fresh capture. {error}", flush=True)
                capture_available = False
                tracker.update(None, tick)
                if response_timing:
                    response_timing.clear_observations()
                if hasattr(planner, "reset"):
                    planner.reset()
                if opening_approach:
                    opening_approach.reset()
                time.sleep(0.05)
                continue
            capture_available = True
            capture_end = time.perf_counter()
            if native_vision:
                captured = sample.presented_at - args.render_delay
                frame_shape = sample.frame_size
            else:
                captured = (
                    getattr(game, "last_capture_time", None) or (capture_start + capture_end) / 2
                ) - args.render_delay
                frame_shape = frame.shape
            if last_shape is not None and frame_shape != last_shape:
                last_click, last_direction = tracker.last_click, tracker.last_direction
                motion_rate, jump_gain = tracker.motion_rate, tracker.jump_gain
                gravity_gain = tracker.gravity_gain
                detector, tracker, _ = get_components(args)
                tracker.motion_rate, tracker.jump_gain = motion_rate, jump_gain
                tracker.gravity_gain = gravity_gain
                if last_direction:
                    tracker.clicked(last_direction, last_click)
                if response_timing:
                    response_timing.clear_observations()
                if opening_approach:
                    opening_approach.reset()
                print("Window resized: reacquiring board and motion from fresh frames.", flush=True)
            last_shape = frame_shape
            detection_start = time.perf_counter()
            obs = sample.observation if native_vision else detector.observe(frame)
            detected_at = time.perf_counter()
            state = tracker.update(obs, captured)
            tracked_at = time.perf_counter()
            if native_vision:
                read_score, board_height = sample.score, sample.board_height
                capture_time = sample.capture_seconds
                detection_time = sample.detection_seconds
                capture_age = capture_start + capture_time - captured
                score_time = None  # OCR and UI recognition run inside native detection.
            else:
                read_score = score_reader.read(obs.image) if obs else None
                board_height = obs.image.shape[0] if obs else 0
                capture_time = capture_end - capture_start
                detection_time = detected_at - detection_start
                capture_age = capture_end - captured
                score_time = time.perf_counter() - tracked_at
            telemetry = {
                "score": read_score,
                "capture_ms": capture_time * 1000,
                "frame_age_at_capture_ms": capture_age * 1000,
                "detection_ms": detection_time * 1000,
                "tracking_ms": (tracked_at - detected_at) * 1000,
                "render_delay_ms": args.render_delay * 1000,
            }
            if capture_timeline:
                capture_timeline.captured(capture_end, telemetry)
            if score_time is not None:
                telemetry["score_ms"] = score_time * 1000
            input_lag = args.input_lag
            if response_timing:
                measured = response_timing.observe(state, captured)
                input_lag = response_timing.delay
                telemetry["input_lag_ms"] = input_lag * 1000
                if measured is not None:
                    telemetry["measured_response_ms"] = measured * 1000
            if read_score is not None:
                score_streak = score_streak + 1 if read_score == score else 1
                score = read_score
                if score_streak >= 3:
                    best_score = max(best_score, score)
                    if state and (round_best_score is None or score > round_best_score):
                        round_best_score = score
                if args.target and score >= args.target and score_streak >= 5 and state:
                    output_frame(captured, frame, obs, state, None, telemetry)
                    target_dir = Path(args.record or "artifacts/success")
                    target_dir.mkdir(parents=True, exist_ok=True)
                    screenshot = target_dir / "target-reached.png"
                    if native_vision:
                        game.save_native_capture(screenshot)
                    elif not cv2.imwrite(str(screenshot), frame):
                        raise RuntimeError(f"Could not save score screenshot to {screenshot}.")
                    (target_dir / "target-reached.json").write_text(
                        json.dumps({"score": score, "t": captured - start, "target": args.target})
                    )
                    print(f"TARGET REACHED: live score {score}", flush=True)
                    break
            was_ending = wall_stall.ending
            ending_round = args.play and wall_stall.update(state, captured, round_best_score)
            if ending_round:
                telemetry["round_end_reason"] = "wall stall timeout"
                if not was_ending:
                    print("Ending round after 10 seconds at the wall.", flush=True)
            decision = None
            sent = 0
            ui_action = (
                sample.ui
                if native_vision
                else ui.inspect(frame)
                if not (obs and obs.active and obs.player)
                else None
            )
            if ui.ready(ui_action, captured) and args.play:
                if (
                    native_vision
                    and recorder
                    and ui_action.kind == "restart"
                    and round_best_score is not None
                    and last_saved_round != restarts
                ):
                    game.save_native_capture(
                        recorder.directory / f"round-{restarts + 1:03}-score.png"
                    )
                    last_saved_round = restarts
                if ui_action.kind == "restart" and (
                    args.no_auto_restart or (args.rounds and restarts >= args.rounds)
                ):
                    output_frame(
                        captured, frame, obs, state, decision, telemetry, ui_kind=ui_action.kind
                    )
                    message = (
                        "Score screen reached; automatic restart disabled."
                        if args.no_auto_restart
                        else f"Completed {restarts} requested rounds."
                    )
                    print(message, flush=True)
                    break
                if time.perf_counter() - capture_start < 0.15 and game.click_at(
                    ui_action.x, ui_action.y
                ):
                    print(f"UI: {ui_action.kind} ({ui_action.confidence:.3f})", flush=True)
                    ui.clicked(time.perf_counter())
                    tracker.update(None, captured)
                    if response_timing:
                        response_timing.clear_observations()
                    if hasattr(planner, "reset"):
                        planner.reset()
                    if opening_approach:
                        opening_approach.reset()
                    if ui_action.kind == "restart":
                        restarts += 1
                        wall_stall.reset()
                        round_best_score = None
                        score_streak = 0
                    telemetry["ui_pressed_at"] = game.last_press_time - start
                    output_frame(
                        captured, frame, obs, state, decision, telemetry, ui_kind=ui_action.kind
                    )
                    time.sleep(0.15)
                    continue
            if ending_round:
                decision = Decision(0, np.empty((0, 2)), "ending round after wall stall")
            elif state:
                age = time.perf_counter() - captured
                plan_start = time.perf_counter()
                latency = age + planning_time + delivery_slack + input_lag + args.button_call_delay
                control_interval = max(
                    1 / args.fps, 0.040, capture_end - capture_start + planning_time + 0.008
                )
                expires_at = capture_start + MAX_INPUT_FRAME_AGE
                latest_effect = (
                    expires_at
                    - captured
                    + input_lag
                    + args.button_call_delay
                    - INPUT_DEADLINE_RESERVE
                    if args.align_delivery
                    else None
                )
                decision = plan_frame(
                    state, board_height, captured, latency, control_interval, latest_effect
                )
                measured_planning_time = time.perf_counter() - plan_start
                timing_ok = True
                replanned = False
                if args.timing_replan:
                    if state.ready and abs(measured_planning_time - planning_time) > 0.008:
                        # A percentile from an earlier busy scene can put a cheap
                        # plan's reset tens of milliseconds after actual delivery.
                        # Recompute at the measured delivery horizon while fresh.
                        if time.perf_counter() - capture_start + measured_planning_time < 0.085:
                            replanned = True
                            control_interval = max(
                                1 / args.fps,
                                0.040,
                                capture_end - capture_start + measured_planning_time + 0.008,
                            )
                            latency = (
                                time.perf_counter()
                                - captured
                                + measured_planning_time
                                + 0.008
                                + input_lag
                            )
                            repeat_start = time.perf_counter()
                            decision = plan_frame(
                                state,
                                board_height,
                                captured,
                                latency,
                                control_interval,
                                latest_effect,
                            )
                            measured_planning_time = time.perf_counter() - repeat_start
                        else:
                            timing_ok = False
                    if state.ready:
                        weight = 0.65 if measured_planning_time < planning_time else 0.4
                        planning_time = max(
                            0.004, (1 - weight) * planning_time + weight * measured_planning_time
                        )
                elif state.ready:
                    planning_samples.append(measured_planning_time)
                    planning_time = max(0.010, float(np.percentile(planning_samples, 90)))
                action_latency = latency
                if args.timed_gap_turns and decision.action and decision.planned_inputs:
                    offset, direction = decision.planned_inputs[0]
                    if direction == decision.action:
                        action_latency = max(latency, offset)
                if args.align_delivery and decision.action and decision.reset_window is not None:
                    last_press = (
                        captured + decision.reset_window[1] - input_lag - args.button_call_delay
                    )
                    expires_at = min(expires_at, last_press)
                    timing_ok = timing_ok and time.perf_counter() < expires_at
                telemetry.update(
                    {
                        "planned_latency_ms": latency * 1000,
                        "action_latency_ms": action_latency * 1000,
                        "planning_ms": measured_planning_time * 1000,
                        "control_interval_ms": control_interval * 1000,
                        "timing_replanned": replanned,
                        "timing_ok": timing_ok,
                    }
                )
                # Never deliver an action based on an old frame.
                if args.play and timing_ok and decision.action and time.perf_counter() < expires_at:
                    release_options = {}
                    if release_rng:
                        blocking = bool(release_rng.getrandbits(1))
                        release_options["blocking_release"] = blocking
                        telemetry["release_mode"] = "blocking" if blocking else "background"
                    if args.align_delivery:
                        target_press = (
                            captured + action_latency - input_lag - args.button_call_delay
                        )
                        telemetry["input_target_at"] = target_press - start
                        delivered = game.click(
                            decision.action,
                            obs.roi,
                            not_before=target_press,
                            expires_at=expires_at,
                            **release_options,
                        )
                    else:
                        delivered = game.click(decision.action, obs.roi, **release_options)
                    if delivered:
                        if response_timing:
                            measured = response_timing.pressed(
                                state, captured, game.last_press_time, decision.action
                            )
                            if measured is not None:
                                telemetry["measured_response_ms"] = measured * 1000
                        effect_at = game.last_press_time + input_lag
                        tracker.clicked(decision.action, effect_at)
                        if opening_approach:
                            opening_approach.confirm_sent()
                        if hasattr(planner, "clicked"):
                            planner.clicked(decision.action)
                        telemetry["input_effect_at"] = effect_at - start
                        clicks += 1
                        sent = decision.action
                        telemetry["input_pressed_at"] = game.last_press_time - start
                        if args.align_delivery and game.last_press_timing:
                            telemetry["input_delivery"] = game.last_press_timing
            if recorder or viewer:
                output_frame(
                    captured,
                    frame,
                    obs,
                    state,
                    decision,
                    telemetry,
                    sent,
                    ui_action.kind if ui_action else None,
                )
            elapsed = time.perf_counter() - tick
            timings.append(elapsed * 1000)
            frames += 1
            if tick - last_report > 1:
                label = (
                    {-1: "LEFT", 0: "WAIT", 1: "RIGHT"}[decision.action] if decision else "NO TRACK"
                )
                ui_label = ui_action.kind if ui_action else "unknown/wait" if not state else "game"
                print(
                    f"{time.perf_counter() - start:5.1f}s {elapsed * 1000:5.1f}ms {label} score={score} best={best_score} clicks={clicks} ui={ui_label} "
                    f"focus={bool(game.focused())} ready={bool(state and state.ready)}",
                    flush=True,
                )
                last_report = tick
            time.sleep(max(0, 1 / args.fps - (time.perf_counter() - tick)))
    finally:
        try:
            if recorder:
                recorder.close()
        finally:
            game.close()
            if viewer:
                viewer.close()
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
            p.add_argument("--policy-weights", help="Opt-in NumPy ML policy archive")
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
            p.add_argument(
                "--search-step",
                type=float,
                default=None,
                help="Seconds between planned actions (native default: 0.05)",
            )
            p.add_argument("--beam", type=int, default=None, help="Maximum candidate routes")
            p.add_argument(
                "--pocket-penalty",
                type=float,
                default=0.0,
                help="Penalize projected dwell beneath a bar near the sidewall",
            )
            p.add_argument(
                "--spatial-velocity",
                action="store_true",
                help="Fit vertical velocity from observed horizontal travel within a jump",
            )
            p.add_argument(
                "--skip-repeated-frames",
                action="store_true",
                help="Keep repeated player and obstacle positions out of motion tracking",
            )
            p.add_argument(
                "--recover-frame-clock",
                action="store_true",
                help="Fit motion from horizontal travel when presentation timing slips",
            )
            p.add_argument(
                "--narrow-gap-cooldown",
                type=float,
                help="Try a shorter minimum between jumps if trapped between squares",
            )
            p.add_argument(
                "--wall-approach",
                action="store_true",
                help="Check the route to the upper opening before a wallward jump",
            )
            p.add_argument(
                "--coast-recovery",
                action="store_true",
                help="Check whether a rising player can coast past an imminent obstacle",
            )
            p.add_argument(
                "--timed-gap-turns",
                action="store_true",
                help="Delay a reversal if the safe window through a square gap favors it",
            )
            p.add_argument(
                "--route-rules",
                action="store_true",
                help="Check immediate safety, preserve room at the center and link openings",
            )
            p.add_argument(
                "--jump-guidance",
                action="store_true",
                help="Account for jump clearance when choosing routes beneath bars",
            )
            p.add_argument(
                "--horizon", type=float, default=None, help="Seconds of trajectory lookahead"
            )
            p.add_argument(
                "--cooldown",
                type=float,
                default=None,
                help="Minimum seconds between planned jump resets",
            )
            p.add_argument(
                "--camera-tau",
                type=float,
                default=0.0,
                help="Camera easing time constant in seconds (0: instantaneous)",
            )
            p.add_argument(
                "--vertical-uncertainty",
                type=float,
                default=0.0,
                help="Extra vertical clearance for prediction uncertainty, in normalized pixels",
            )
            p.add_argument(
                "--adaptive-motion",
                action="store_true",
                help="Estimate motion from spatial arcs and adapt to game frame rate",
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
                    help="Show a separate OpenCV tracking window; leave game focused for play",
                )
                p.add_argument(
                    "--release-ab",
                    action="store_true",
                    help="Randomize posted jump releases between blocking and background modes",
                )
                p.add_argument(
                    "--record",
                    help="Directory for telemetry and diagnostic video (telemetry only with native vision)",
                )
                p.add_argument(
                    "--native-vision",
                    action="store_true",
                    help="Windows: use native DXGI capture and detection without Python pixels",
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
                p.add_argument(
                    "--input-lag",
                    type=float,
                    default=0.0,
                    help="Seconds from input call return to the visible velocity reset",
                )
                p.add_argument(
                    "--button-call-delay",
                    type=float,
                    default=0.0,
                    help="Expected Windows button-down call duration for aligned delivery",
                )
                p.add_argument(
                    "--adaptive-input-delay",
                    action="store_true",
                    help="Measure visible jump timing from recent horizontal reversals",
                )
                p.add_argument(
                    "--input-delay-range",
                    type=float,
                    nargs=2,
                    metavar=("MIN", "MAX"),
                    help="Bound the 15-sample adaptive delay estimate in seconds",
                )
                p.add_argument(
                    "--timing-replan",
                    action="store_true",
                    help="Experimental replanning when computation time changes",
                )
                p.add_argument(
                    "--align-delivery",
                    action="store_true",
                    help="Windows: deliver inputs at their planned button-down time",
                )
                p.add_argument(
                    "--delivery-slack",
                    type=float,
                    help="Extra planning time in seconds for aligned input (default: 0.012)",
                )
                jumps = p.add_mutually_exclusive_group()
                jumps.add_argument(
                    "--batch-jumps",
                    action="store_true",
                    help="Windows: send jump button down and up in one native call",
                )
                jumps.add_argument(
                    "--message-jumps",
                    action="store_true",
                    help="Windows: post jump button messages to the focused game window",
                )
                p.add_argument("--duration", type=float, default=60)
                p.add_argument("--fps", type=float, default=60)
                p.add_argument(
                    "--render-delay",
                    type=float,
                    help="Additional age of displayed pixels (0.040s on Windows, 0 for native vision)",
                )
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
    if getattr(args, "native_vision", False) and sys.platform != "win32":
        parser.error("--native-vision requires Windows")
    if args.command == "live" and args.render_delay is None:
        args.render_delay = 0.040 if sys.platform == "win32" and not args.native_vision else 0.0
    if getattr(args, "align_delivery", False):
        if sys.platform != "win32":
            parser.error("--align-delivery currently requires Windows")
        if args.timing_replan:
            parser.error("--align-delivery and --timing-replan are separate experiments")
    if getattr(args, "batch_jumps", False) and sys.platform != "win32":
        parser.error("--batch-jumps requires Windows")
    if getattr(args, "message_jumps", False) and sys.platform != "win32":
        parser.error("--message-jumps requires Windows")
    if getattr(args, "release_ab", False) and not (
        args.message_jumps and args.play and args.record
    ):
        parser.error("--release-ab requires --message-jumps, --play and --record")
    if getattr(args, "adaptive_input_delay", False) and not args.align_delivery:
        parser.error("--adaptive-input-delay requires --align-delivery")
    delay_range = getattr(args, "input_delay_range", None)
    if delay_range is not None and (
        not args.adaptive_input_delay
        or not all(np.isfinite(delay_range))
        or not 0 <= delay_range[0] <= args.input_lag <= delay_range[1] <= 0.112
    ):
        parser.error(
            "--input-delay-range requires adaptive delay and bounds containing --input-lag"
        )
    if getattr(args, "button_call_delay", 0) and (
        args.button_call_delay < 0 or not args.align_delivery
    ):
        parser.error("--button-call-delay must be nonnegative and requires --align-delivery")
    slack = getattr(args, "delivery_slack", None)
    if slack is not None and (not np.isfinite(slack) or slack < 0 or not args.align_delivery):
        parser.error(
            "--delivery-slack must be finite and nonnegative and requires --align-delivery"
        )
    cv2.setNumThreads(1)
    if getattr(args, "search_step", None) is not None and args.search_step <= 0:
        parser.error("--search-step must be positive")
    if getattr(args, "horizon", None) is not None and args.horizon <= 0:
        parser.error("--horizon must be positive")
    if getattr(args, "beam", None) is not None and args.beam < 3:
        parser.error("--beam must be at least 3")
    pocket_penalty = getattr(args, "pocket_penalty", 0)
    if not np.isfinite(pocket_penalty) or pocket_penalty < 0:
        parser.error("--pocket-penalty must be finite and nonnegative")
    if pocket_penalty and (args.planner != "native" or args.policy_weights):
        parser.error("--pocket-penalty requires the native planner")
    if getattr(args, "coast_recovery", False) and (args.planner != "native" or args.policy_weights):
        parser.error("--coast-recovery requires the native planner")
    if getattr(args, "route_rules", False) and (args.planner != "native" or args.policy_weights):
        parser.error("--route-rules requires the native planner")
    if getattr(args, "timed_gap_turns", False) and (
        not getattr(args, "align_delivery", False)
        or args.planner != "native"
        or args.policy_weights
    ):
        parser.error("--timed-gap-turns requires the native planner and --align-delivery")
    gap_cooldown = getattr(args, "narrow_gap_cooldown", None)
    if gap_cooldown is not None and (
        not np.isfinite(gap_cooldown)
        or gap_cooldown <= 0
        or args.planner != "native"
        or args.policy_weights
    ):
        parser.error("--narrow-gap-cooldown must be positive and requires the native planner")
    if getattr(args, "wall_approach", False) and (args.planner != "native" or args.policy_weights):
        parser.error("--wall-approach requires the native planner")
    if getattr(args, "spatial_velocity", False) and (args.adaptive_motion or args.policy_weights):
        parser.error("--spatial-velocity requires fixed physics and the trajectory planner")
    if getattr(args, "cooldown", None) is not None and args.cooldown <= 0:
        parser.error("--cooldown must be positive")
    if getattr(args, "camera_tau", 0) < 0 or getattr(args, "vertical_uncertainty", 0) < 0:
        parser.error("--camera-tau and --vertical-uncertainty must be nonnegative")
    if args.command == "live" and (
        args.fps <= 0
        or args.duration <= 0
        or args.rounds < 0
        or args.target < 0
        or args.render_delay < 0
        or args.input_lag < 0
    ):
        parser.error(
            "--fps and --duration must be positive; "
            "--rounds, --target and --render-delay must be nonnegative"
        )
    try:
        args.run(args)
    except (RuntimeError, ValueError, OSError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Stopped.")
    return 0
