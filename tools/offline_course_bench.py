"""Offline pixel-and-physics regression on the last twenty static courses.

The environment is reconstructed from recorded observations, not game internals.
A result proves behavior only in this model, never a counterfactual live score.
It models nonfatal horizontal clamping, camera easing, delayed controls, and
pixels passing through the actual detector/tracker. Unknown terrain is excluded.
"""

import argparse
import json
import sys
from collections import deque
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jumpbot.controller import (
    INPUT_DEADLINE_RESERVE,
    MAX_INPUT_FRAME_AGE,
    OpeningApproach,
    TimedController,
)
from jumpbot.fast_planner import FastPlanner
from jumpbot.native_search import advance_scene
from jumpbot.physics import Physics, Tracker
from jumpbot.vision import Detector, Rect, camera_shift


def reconstruct(deaths):
    cache = {}
    last = {}
    courses = []
    for death in deaths:
        name = death["session"]
        folder = Path("artifacts") / name
        if name not in cache:
            rows = [json.loads(line) for line in (folder / "frames.jsonl").open(encoding="utf-8")]
            previous = None
            scroll = 0.0
            for row in rows:
                boxes = [Rect(**b) for b in row["obstacles"]]
                if row.get("state") and previous is not None:
                    scroll += camera_shift(previous, boxes)[0]
                previous = boxes if row.get("state") else None
                row["world_scroll"] = scroll
            cache[name] = rows
        rows = [r for r in cache[name] if last.get(name, -1) < r["t"] < death["contact_t"] - 0.02]
        last[name] = death["death_ui_t"]
        active = [r for r in rows if r.get("state") and r.get("player") and r["player"]["h"] <= 35]
        first = next(r for r in active if r["telemetry"].get("score") == 0)
        origin = first["world_scroll"]
        groups = []
        for row in active:
            if row["t"] < first["t"]:
                continue
            shift = row["world_scroll"] - origin
            for b in row["obstacles"]:
                edge = b["x"] < 4 or b["x"] + b["w"] > 416
                square = 25 <= b["w"] <= 105 and abs(b["w"] - b["h"]) <= 3
                if not ((edge and 30 <= b["h"] <= 42) or square):
                    continue
                value = np.array([b["x"], b["y"] - shift, b["w"], b["h"]], float)
                group = None
                for g in reversed(groups):
                    center = np.median(g, axis=0)
                    if (
                        abs(center[1] - value[1]) < 6
                        and abs(center[0] - value[0]) < 4
                        and (edge or abs(center[2] - value[2]) < 4)
                    ):
                        group = g
                        break
                if group is None:
                    groups.append([value])
                else:
                    group.append(value)
        boxes = [np.median(g, axis=0).tolist() for g in groups if len(g) >= 5]
        s = first["state"]
        minimum = min(r["state"]["y"] - r["world_scroll"] + origin for r in active)
        courses.append(
            {
                "number": death["number"],
                "source": name,
                "recorded_score": death["score"],
                "initial": [s[k] for k in ["x", "y", "vx", "vy"]],
                "height": round(first["roi"][3] * 420 / first["roi"][2]),
                "boxes": boxes,
                "score_step": (s["y"] - minimum) / max(1, death["score"]),
            }
        )
    return courses


def assets():
    sprite = np.full((30, 31, 3), 42, np.uint8)
    sprite[5:11, :] = (30, 193, 250)
    sprite[:, -1] = 227
    badge = np.full((47, 128, 3), 227, np.uint8)
    with np.load("jumpbot/assets/digits.npz") as archive:
        digits = {k: archive[k].copy() for k in archive.files}
    with np.load("jumpbot/assets/digits_windows.npz") as archive:
        digits.update({k: archive[k].copy() for k in archive.files})
    return sprite, badge, digits


def render(course, state, sprite, badge, digits):
    x, y, target, scene, score = state
    height = course["height"]
    frame = np.full((height, 420, 3), 227, np.uint8)
    for bx, by, bw, bh in course["boxes"]:
        left, right = max(0, round(bx)), min(420, round(bx + bw))
        top, bottom = max(0, round(by + scene)), min(height, round(by + scene + bh))
        if left < right and top < bottom:
            frame[top:bottom, left:right] = 42
    frame[54:101, 7:135] = badge
    glyphs = [digits[c] for c in str(max(0, score))]
    width = sum(g.shape[1] for g in glyphs) + 3 * (len(glyphs) - 1)
    left = round(210 - width / 2)
    for glyph in glyphs:
        h, w = glyph.shape
        patch = frame[55 : 55 + h, left : left + w]
        patch[glyph > 127] = 42
        left += w + 3
    left, top = round(x - sprite.shape[1] / 2), round(y + target - sprite.shape[0] / 2)
    right, bottom = left + sprite.shape[1], top + sprite.shape[0]
    x0, x1, y0, y1 = max(0, left), min(420, right), max(0, top), min(height, bottom)
    if x0 < x1 and y0 < y1:
        frame[y0:y1, x0:x1] = sprite[y0 - top : y1 - top, x0 - left : x1 - left]
    return frame


def simulate(
    course,
    legacy=False,
    duration=100.0,
    seed=142,
    jitter=0.0,
    scale=1.0,
    vertical=0.0,
    recorded_timing=False,
    debug_dir=None,
    error_floor=0.0,
    align_delivery=False,
    reset_timing=0.0,
    commit_inputs=False,
    cooldown=0.18,
    rl_weights=None,
    processing=0.020,
    control_period=0.040,
    search_step=0.025,
    beam=600,
    horizon=3.0,
    pocket_penalty=0.0,
    wait_reserve=0.0,
    spatial_velocity=False,
    skip_repeated_frames=False,
    recover_frame_clock=False,
    narrow_gap_cooldown=None,
    wall_approach=False,
    coast_recovery=False,
    timed_gap_turns=False,
    route_rules=False,
    staged_opening=False,
):
    physics = Physics(150.4, 688.55, 1415.0)
    height = course["height"]
    detector = Detector(
        (0, 0, round(420 * scale), round(height * scale)), recover_score_blocks=not legacy
    )
    tracker = Tracker(
        physics,
        reset_phase=0.0,
        spatial_velocity=spatial_velocity,
        skip_repeated_frames=skip_repeated_frames,
        recover_frame_clock=recover_frame_clock,
    )
    planner = FastPlanner(
        physics,
        step=search_step,
        beam=beam,
        horizon=horizon,
        pocket_penalty=pocket_penalty,
        narrow_gap_cooldown=narrow_gap_cooldown,
        wall_approach=wall_approach,
        coast_recovery=coast_recovery,
        timed_gap_turns=timed_gap_turns,
        route_rules=route_rules,
        cooldown=cooldown,
        camera_tau=0.011,
        vertical_uncertainty=vertical,
        vertical_error_floor=error_floor,
        reset_timing_uncertainty=reset_timing,
    )
    if rl_weights:
        from rl_policy import RLPlanner

        planner = RLPlanner(rl_weights)
        cooldown = 0.12
    sprite, badge, digits = assets()
    controller = TimedController(planner) if commit_inputs else None
    opening_approach = OpeningApproach(planner) if staged_opening else None
    x, y, vx, vy = course["initial"]
    peak = min(height / 2, y)
    scene = 0.0
    t = 10.0
    start = t
    next_control = t + control_period
    pending = deque()
    history = deque(maxlen=100)
    rng = np.random.default_rng(seed)
    last_progress = t
    last_peak = peak
    commands = 0
    started = False
    ready = frames = 0
    trace = []
    reason = "duration"
    collision = None
    recent = deque(maxlen=30)
    boxes = np.array(course["boxes"])
    goal = min(boxes[:, 1]) - 35
    while t - start < duration:
        target = max(0, height / 2 - peak)
        score = int(max(0, course["initial"][1] - peak) / max(1, course["score_step"]))
        history.append((t, (x, y, target, scene, score)))
        if t + 1e-9 >= next_control:
            timing = course["timings"][frames % len(course["timings"])] if recorded_timing else None
            _, visual = min(history, key=lambda r: abs(r[0] - (t - 0.01)))
            # Report acquisition time, as MSS does. Pixels have a separate
            # presentation delay. The configured 35ms visible input response
            # comprises 25ms plant response plus that 10ms presentation delay.
            captured = t
            image = render(course, visual, sprite, badge, digits)
            if scale != 1:
                image = cv2.resize(image, (round(420 * scale), round(height * scale)))
            observation = detector.observe(image)
            state = tracker.update(observation, captured)
            frames += 1
            sent = False
            button_delay = timing[3] if timing else processing
            if timing and state and state.ready and not timing[4]:
                button_delay += max(0.0, course["planning_median"] - timing[5])
            if state:
                ready += state.ready
                latency = timing[1] if timing else t - captured + 0.055
                interval = (timing[2] if timing else control_period) + wait_reserve
                command = (
                    controller.plan(state, height, captured, latency, interval)
                    if controller
                    else None
                )
                if opening_approach:
                    latest_effect = (
                        MAX_INPUT_FRAME_AGE + 0.035 - INPUT_DEADLINE_RESERVE
                        if align_delivery
                        else None
                    )
                    decision = opening_approach.plan(
                        state, height, captured, latency, interval, latest_effect=latest_effect
                    )
                elif command:
                    decision = command.decision
                else:
                    decision = planner.plan(state, height, latency, interval)
                if command and command.effect_at is not None:
                    button_delay = max(button_delay, command.effect_at - captured - 0.035)
                if debug_dir is not None:
                    recent.append(
                        (
                            image.copy(),
                            {
                                "state": asdict(state),
                                "physics": asdict(physics),
                                "latency": timing[1] if timing else t - captured + 0.055,
                                "control_interval": interval,
                                "elapsed": t - start,
                                "captured": captured - start,
                                "ground_truth": [x, y, vx, vy, peak, scene],
                                "decision": {
                                    "action": decision.action,
                                    "reason": decision.reason,
                                    "horizon": decision.predicted_horizon,
                                    "inputs": decision.planned_inputs,
                                    "path": decision.path,
                                    "reset_window": decision.reset_window,
                                },
                                "pending": list(pending),
                            },
                        )
                    )
                trace.append(
                    [
                        round(t - start, 3),
                        round(x, 1),
                        round(y, 1),
                        state.ready,
                        decision.action,
                        round(decision.predicted_horizon, 3),
                    ]
                )
                if decision.action:
                    action_latency = latency
                    if timed_gap_turns and decision.planned_inputs:
                        offset, direction = decision.planned_inputs[0]
                        if direction == decision.action:
                            action_latency = max(latency, offset)
                    if align_delivery:
                        # The delivery target is absolute relative to capture.
                        # Processing that already ran consumes part of the wait.
                        press_delay = max(button_delay, action_latency - 0.035)
                        expires = MAX_INPUT_FRAME_AGE
                        if decision.reset_window is not None:
                            expires = min(expires, decision.reset_window[1] - 0.035)
                    else:
                        press_delay = button_delay + (action_latency - latency)
                        expires = np.inf
                    if press_delay < expires:
                        expected = captured + press_delay + 0.035
                        effect = expected - 0.010 + float(rng.uniform(-jitter, jitter))
                        pending.append((effect, decision.action))
                        tracker.clicked(decision.action, expected)
                        if controller:
                            controller.confirm_sent(expected)
                        if opening_approach:
                            opening_approach.confirm_sent()
                        button_delay = press_delay
                        commands += 1
                        sent = True
            elif controller:
                controller.reset()
            elif opening_approach:
                opening_approach.reset()
            next_control = t + (
                max(0.015, timing[0], button_delay + (0.008 if sent else 0))
                if timing
                else control_period
            )
        while pending and t + 1e-9 >= pending[0][0]:
            _, direction = pending.popleft()
            vx = direction * physics.speed
            vy = -physics.jump
            started = True
        dt = 0.005
        before = max(0, height / 2 - peak)
        x = float(np.clip(x + vx * dt, 0, 420))
        if started:
            y += vy * dt + 0.5 * physics.gravity * dt * dt
            vy += physics.gravity * dt
        peak = min(peak, y)
        target = max(0, height / 2 - peak)
        scene = advance_scene(scene, before, target, dt, 0.011)
        t += dt
        if x <= 0 or x >= 420:
            vx = 0.0
        relative_y = y + target - scene
        contacts = [
            b.tolist()
            for b in boxes
            if x + 15.5 > b[0]
            and x - 15.5 < b[0] + b[2]
            and relative_y + 15 > b[1]
            and relative_y - 15 < b[1] + b[3]
        ]
        if contacts or y + target + 15 >= height:
            reason = "collision"
            collision = contacts
            break
        if relative_y + 15 < goal:
            reason = "known_course_cleared"
            break
        if peak < last_peak - 10:
            last_progress = t
            last_peak = peak
        if t - last_progress > 12:
            reason = "stalled"
            break
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        for i, (image, snapshot) in enumerate(recent):
            cv2.imwrite(str(debug_dir / f"frame-{i:02}.png"), image)
        (debug_dir / "snapshots.json").write_text(
            json.dumps(
                [s for _, s in recent],
                indent=2,
                default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x,
            )
            + "\n"
        )
    return {
        "number": course["number"],
        "time_contract_version": 4,
        "processing": processing,
        "control_period": control_period,
        "aligned_delivery": align_delivery,
        "reset_timing_uncertainty": reset_timing,
        "committed_inputs": commit_inputs,
        "cooldown": cooldown,
        "search_step": search_step,
        "beam": beam,
        "horizon": horizon,
        "pocket_penalty": pocket_penalty,
        "wait_reserve": wait_reserve,
        "spatial_velocity": spatial_velocity,
        "skip_repeated_frames": skip_repeated_frames,
        "recover_frame_clock": recover_frame_clock,
        "narrow_gap_cooldown": narrow_gap_cooldown,
        "wall_approach": wall_approach,
        "coast_recovery": coast_recovery,
        "timed_gap_turns": timed_gap_turns,
        "route_rules": route_rules,
        "staged_opening": staged_opening,
        "rl_weights": str(rl_weights) if rl_weights else None,
        "commitment_stats": controller.stats if controller else {},
        "legacy_detection": legacy,
        "recorded_timing": recorded_timing,
        "seed": seed,
        "jitter": jitter,
        "scale": scale,
        "vertical_uncertainty": vertical,
        "vertical_error_floor": error_floor,
        "reason": reason,
        "seconds": round(t - start, 3),
        "climb": round(course["initial"][1] - peak, 1),
        "recorded_score": course["recorded_score"],
        "commands": commands,
        "ready_fraction": round(ready / max(1, frames), 3),
        "collision": collision,
        "trace": trace,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=int, nargs="*")
    parser.add_argument("--legacy", action="store_true")
    parser.add_argument("--recorded-timing", action="store_true")
    parser.add_argument("--dump-failures", action="store_true")
    parser.add_argument("--duration", type=float, default=100.0)
    parser.add_argument(
        "--seed", type=int, default=142, help="Random seed for input delivery jitter"
    )
    parser.add_argument("--jitter", type=float, default=0.0)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--vertical", type=float, default=0.0)
    parser.add_argument("--error-floor", type=float, default=0.0)
    parser.add_argument("--align-delivery", action="store_true")
    parser.add_argument("--reset-timing", type=float, default=0.0)
    parser.add_argument("--commit-inputs", action="store_true")
    parser.add_argument("--cooldown", type=float, default=0.18)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rl-weights", type=Path)
    parser.add_argument("--processing", type=float, default=0.020)
    parser.add_argument("--control-period", type=float, default=0.040)
    parser.add_argument("--search-step", type=float, default=0.025)
    parser.add_argument("--beam", type=int, default=600)
    parser.add_argument("--horizon", type=float, default=3.0)
    parser.add_argument("--pocket-penalty", type=float, default=0.0)
    parser.add_argument("--wait-reserve", type=float, default=0.0)
    parser.add_argument("--spatial-velocity", action="store_true")
    parser.add_argument("--skip-repeated-frames", action="store_true")
    parser.add_argument("--recover-frame-clock", action="store_true")
    parser.add_argument("--narrow-gap-cooldown", type=float)
    parser.add_argument("--wall-approach", action="store_true")
    parser.add_argument("--coast-recovery", action="store_true")
    parser.add_argument("--timed-gap-turns", action="store_true")
    parser.add_argument("--route-rules", action="store_true")
    parser.add_argument("--staged-opening", action="store_true")
    args = parser.parse_args()
    if args.search_step <= 0 or args.horizon < args.search_step or args.beam < 3:
        parser.error(
            "Use a positive --search-step, a --horizon of at least one step and --beam >= 3"
        )
    if args.staged_opening and (
        not (args.route_rules and args.wall_approach and args.timed_gap_turns)
        or args.commit_inputs
        or args.rl_weights
    ):
        parser.error("--staged-opening requires the route rules, timed turns and wall approach")
    cache = Path("artifacts/regression-audit/known-courses.json")
    if cache.exists():
        courses = json.loads(cache.read_text())
    else:
        courses = reconstruct(json.load(open("artifacts/regression-audit/last20-deaths.json")))
        cache.write_text(json.dumps(courses, indent=2) + "\n")
    if args.recorded_timing and courses[0].get("timings_version") != 2:
        recordings = {}
        boundaries = {}
        deaths = json.load(open("artifacts/regression-audit/last20-deaths.json"))
        for course, death in zip(courses, deaths):
            name = course["source"]
            if name not in recordings:
                recordings[name] = [
                    json.loads(line)
                    for line in (Path("artifacts") / name / "frames.jsonl").open(encoding="utf-8")
                ]
            rows = [
                r
                for r in recordings[name]
                if boundaries.get(name, -1) < r["t"] < death["contact_t"]
            ]
            boundaries[name] = death["death_ui_t"]
            timings = []
            for row, following in zip(rows, rows[1:]):
                tele = row["telemetry"]
                if not row.get("state") or "planned_latency_ms" not in tele:
                    continue
                button = (
                    tele["input_pressed_at"] - row["t"]
                    if row["sent"]
                    else tele["frame_age_ms"] / 1000
                )
                timings.append(
                    [
                        min(0.15, following["t"] - row["t"]),
                        tele["planned_latency_ms"] / 1000,
                        tele["control_interval_ms"] / 1000,
                        button,
                        bool(row["state"]["ready"]),
                        tele["planning_ms"] / 1000,
                    ]
                )
            course["timings"] = timings
            course["timings_version"] = 2
            course["planning_median"] = float(np.median([r[5] for r in timings if r[4]]))
        cache.write_text(json.dumps(courses, indent=2) + "\n")
    results = []
    for course in courses:
        if args.case and course["number"] not in args.case:
            continue
        result = simulate(
            course,
            args.legacy,
            args.duration,
            seed=args.seed,
            jitter=args.jitter,
            scale=args.scale,
            vertical=args.vertical,
            recorded_timing=args.recorded_timing,
            error_floor=args.error_floor,
            align_delivery=args.align_delivery,
            reset_timing=args.reset_timing,
            commit_inputs=args.commit_inputs,
            cooldown=args.cooldown,
            rl_weights=args.rl_weights,
            processing=args.processing,
            control_period=args.control_period,
            search_step=args.search_step,
            beam=args.beam,
            horizon=args.horizon,
            pocket_penalty=args.pocket_penalty,
            wait_reserve=args.wait_reserve,
            spatial_velocity=args.spatial_velocity,
            skip_repeated_frames=args.skip_repeated_frames,
            recover_frame_clock=args.recover_frame_clock,
            narrow_gap_cooldown=args.narrow_gap_cooldown,
            wall_approach=args.wall_approach,
            coast_recovery=args.coast_recovery,
            timed_gap_turns=args.timed_gap_turns,
            route_rules=args.route_rules,
            staged_opening=args.staged_opening,
            debug_dir=args.output.parent / args.output.stem / f"case-{course['number']:02}"
            if args.dump_failures
            else None,
        )
        results.append(result)
        print(json.dumps({k: v for k, v in result.items() if k != "trace"}), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n")
