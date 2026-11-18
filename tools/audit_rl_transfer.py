"""Replay pixel-benchmark inputs through the training plant to audit its clock."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from numba import set_num_threads

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.rl_env import CourseEnv


def audit(result, course):
    env = CourseEnv(n=1, courses=[course])
    if result.get("recorded_timing") or result.get("aligned_delivery") or result.get("jitter", 0):
        raise ValueError("Action traces alone cannot reconstruct variable input delivery")
    env.params[0, 8] = result.get("processing", 0.020)
    trace = result["trace"]
    env.params[0, 5] = trace[0][0]
    env.step([0], auto_reset=False)
    errors = []
    for i, row in enumerate(trace):
        elapsed, x, y, ready, action, horizon = row
        errors.append([elapsed, env.s[0, 0] - x, env.s[0, 1] - y])
        if i + 1 < len(trace):
            env.params[0, 5] = trace[i + 1][0] - elapsed
            env.step([{0: 0, -1: 1, 1: 2}[action]], auto_reset=False)
    errors = np.asarray(errors)
    return {
        "case": result["number"],
        "frames": len(errors),
        "time_contract_version": result["time_contract_version"],
        "max_abs_x": round(float(np.max(abs(errors[:, 1]))), 3),
        "max_abs_y": round(float(np.max(abs(errors[:, 2]))), 3),
        "median_control_ms": round(float(np.median(np.diff(errors[:, 0]))) * 1000, 3),
        "first_divergence": next(
            (r.tolist() for r in errors if abs(r[1]) > 1 or abs(r[2]) > 3), None
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    set_num_threads(2)
    courses = json.loads(Path("artifacts/regression-audit/known-courses.json").read_text())
    results = json.loads(args.results.read_text())
    report = [audit(r, next(c for c in courses if c["number"] == r["number"])) for r in results]
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
