"""Compare ideal and actual detector features at identical simulated states."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from numba import set_num_threads

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.rl_env import CourseEnv
from tools.rl_policy import NumpyPolicy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights", type=Path)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--observation-version", type=int, default=2, choices=(2, 3))
    args = parser.parse_args()
    set_num_threads(2)
    policy = NumpyPolicy(args.weights)
    env = CourseEnv(
        n=1,
        seed=89373,
        stage=2,
        tiles=12,
        big_start=5,
        pixel_envs=1,
        observation_version=args.observation_version,
    )
    errors = []
    disagreements = eligible = actual_bad = ideal_bad = count_mismatches = 0
    for _ in range(args.steps):
        env.params[0, 8] = 0.020
        ideal = env.ideal_obs
        actual = env.obs.copy()
        actual_action = int(policy.logits(actual).argmax())
        ideal_action = int(policy.logits(ideal).argmax())
        if actual[0, 4] > 0:
            errors.append(
                (actual[0, [0, 1, 2, 3, 7]] - ideal[0, [0, 1, 2, 3, 7]])
                * [210, env.params[0, 3] / 2, 150, 700, 700]
            )
            ideal_count = ideal[0, 8:].reshape(12, 7)[:, -1].sum()
            actual_count = actual[0, 8:].reshape(12, 7)[:, -1].sum()
            count_mismatches += int(ideal_count != actual_count)
        if actual[0, 4] >= 0.999:
            eligible += 1
            disagreements += int(actual_action != ideal_action)
            risks = env.action_risks()
            actual_bad += int(risks[0, actual_action])
            ideal_bad += int(risks[0, ideal_action])
        env.step([actual_action])
    errors = np.asarray(errors)
    report = {
        "steps": args.steps,
        "samples": len(errors),
        "eligible_actions": eligible,
        "observation_version": args.observation_version,
        "action_disagreements": disagreements,
        "actual_unsafe": actual_bad,
        "ideal_unsafe": ideal_bad,
        "obstacle_count_mismatches": count_mismatches,
        "errors": {
            name: {
                "median": round(float(np.median(errors[:, j])), 3),
                "p95_abs": round(float(np.percentile(abs(errors[:, j]), 95)), 3),
            }
            for j, name in enumerate(["x_px", "y_px", "vx_px_s", "vy_px_s", "camera_px_s"])
        },
        "episodes": len(env.completed),
        "scope": "Independent procedural seed, same physical trajectory, actual pixel encoder.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
