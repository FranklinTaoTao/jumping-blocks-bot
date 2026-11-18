"""Train the unchanged tiny actor from successful procedural planner rollouts.

The twenty recording courses are evaluated only, never used to collect labels.
The heavy planner is a training teacher and is not part of policy inference.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from numba import set_num_threads

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jumpbot.fast_planner import FastPlanner
from jumpbot.physics import Physics, State
from tools.rl_env import CourseEnv
from tools.rl_policy import NumpyPolicy


def teacher_state(env):
    obs = env.obs[0]
    height = env.params[0, 3]
    x, y = (obs[0] + 1) * 210, (obs[1] + 1) * height / 2
    boxes = []
    for dx, dy, w, h, vx, vy, valid in obs[8:].reshape(12, 7):
        if valid:
            boxes.append(
                [
                    x + dx * 210 - w * 105,
                    y + dy * 420 - h * 210,
                    w * 210,
                    h * 420,
                    vx * 150,
                    vy * 700,
                ]
            )
    return State(
        x,
        y,
        obs[2] * 150,
        obs[3] * 700,
        30,
        30,
        age_since_click=env.s[0, 6] - 0.035,
        ready=bool(obs[4] >= 0.999),
        obstacles=np.asarray(boxes, np.float64).reshape(-1, 6),
        camera_speed=obs[7] * 700,
    )


def collect(output, courses, tiles, seed, beam=600, student_weights=None, pixels=False):
    set_num_threads(2)
    output.mkdir(parents=True, exist_ok=True)
    planner = FastPlanner(
        Physics(150.4, 688.55, 1415), step=0.025, beam=beam, cooldown=0.12, camera_tau=0.011
    )
    observations, labels = [], []
    accepted, qualities = [], []
    student = NumpyPolicy(student_weights) if student_weights else None
    reports = []
    started = time.perf_counter()
    for i in range(courses):
        env = CourseEnv(
            n=1,
            seed=seed + i,
            stage=2,
            tiles=tiles,
            big_start=5,
            observation_version=3,
            processing=0.020,
            large_x=(100, 225),
            pair_probability=0.35,
            pixel_envs=int(pixels),
        )
        episode_obs, episode_labels = [], []
        episode_accepted, episode_quality = [], []
        interventions = 0
        short = 0
        for _ in range(4001):
            state = teacher_state(env)
            action = 0
            if state.ready:
                quality = np.empty((3, 2)) if student else None
                decision = planner.plan(
                    state,
                    env.params[0, 3],
                    latency=0.055,
                    control_interval=0.04,
                    root_quality=quality,
                )
                if student is None and env.obs[0, 5] == 0 and decision.action == 0:
                    options = [
                        planner.plan(state, env.params[0, 3], 0.055, 0.04, first_action=a)
                        for a in [-1, 1]
                    ]
                    decision = max(options, key=lambda d: (d.predicted_horizon, d.score))
                action = {0: 0, -1: 1, 1: 2}[decision.action]
                if student:
                    quality = quality[[1, 0, 2]].copy()  # wait, left, right
                    quality[:, 0] *= planner.step
                    logits = student.logits(env.obs)[0]
                    valid = logits > -1e8
                    longest = max(quality[valid, 0])
                    comparable = valid & (quality[:, 0] >= longest - 1e-6)
                    best_score = max(quality[comparable, 1])
                    allowed = comparable & (quality[:, 1] >= best_score - 20.0)
                    proposed = int(logits.argmax())
                    choices = np.flatnonzero(comparable)
                    action = (
                        proposed
                        if allowed[proposed]
                        else int(choices[np.argmax(quality[choices, 1])])
                    )
                    interventions += action != proposed
                short += decision.predicted_horizon < 0.5
                if decision.predicted_horizon >= 0.5 and env.s[0, 11] < 2:
                    episode_obs.append(env.obs[0].copy())
                    episode_labels.append(action)
                    if student:
                        episode_accepted.append(allowed)
                        episode_quality.append(quality)
            env.step([action], auto_reset=False)
            if env.done[0]:
                break
        report = {
            "course": i,
            "seed": seed + i,
            "reason": int(env.reasons[0]),
            "climb": round(float(523 - env.s[0, 4]), 1),
            "seconds": round(float(env.s[0, 10]), 2),
            "large_rectangles": int(np.count_nonzero(env.boxes[0, :, 3] == 97)),
            "examples": len(episode_obs),
            "short_forecasts": short,
        }
        report["interventions"] = interventions
        if report["reason"] == 3:
            observations.extend(episode_obs)
            labels.extend(episode_labels)
            accepted.extend(episode_accepted)
            qualities.extend(episode_quality)
        reports.append(report)
        print(json.dumps(report), flush=True)
        data = {
            "observations": np.asarray(observations, np.float32).reshape(-1, 92),
            "labels": np.asarray(labels, np.int64),
        }
        if student:
            data.update(
                accepted=np.asarray(accepted, bool).reshape(-1, 3),
                quality=np.asarray(qualities, np.float64).reshape(-1, 3, 2),
            )
        np.savez_compressed(output / "dataset.npz", **data)
        (output / "collection.json").write_text(
            json.dumps(
                {
                    "courses": reports,
                    "teacher": {"beam": beam, "cooldown": 0.12, "step": 0.025},
                    "large_x": [100, 225],
                    "pair_probability": 0.35,
                    "student_weights": str(student_weights) if student_weights else None,
                    "pixels": pixels,
                    "root_score_margin": 20.0,
                    "tiles": tiles,
                    "seed": seed,
                    "seconds": time.perf_counter() - started,
                },
                indent=2,
            )
        )
    return output / "dataset.npz"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--courses", type=int, default=4)
    parser.add_argument("--tiles", type=int, default=12)
    parser.add_argument("--seed", type=int, default=11000)
    parser.add_argument("--beam", type=int, default=600)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--student", type=Path)
    parser.add_argument("--pixels", action="store_true")
    args = parser.parse_args()
    collect(args.output, args.courses, args.tiles, args.seed, args.beam, args.student, args.pixels)


if __name__ == "__main__":
    main()
