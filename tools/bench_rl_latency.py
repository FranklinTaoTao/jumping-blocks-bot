"""Measure actual single-observation CPU policy cost, including feature encoding."""

import argparse
import json
import platform
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from rl_policy import NumpyPolicy, encode_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights", type=Path)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--budget-ms", type=float, default=2.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter_ns()
    policy = NumpyPolicy(args.weights)
    load_ms = (time.perf_counter_ns() - started) / 1e6
    states = []
    with args.frames.open(encoding="utf-8") as stream:
        for i, line in enumerate(stream):
            if i % 19:
                continue
            row = json.loads(line)
            if row.get("state"):
                state = row["state"]
                state["obstacles"] = np.asarray(state["obstacles"])
                states.append(SimpleNamespace(**state))
            if len(states) >= 256:
                break
    if not states:
        raise ValueError("No recorded states available")
    samples = []
    for i in range(args.iterations + 500):
        state = states[i % len(states)]
        started = time.perf_counter_ns()
        obs = encode_state(
            state,
            736,
            direction=1 if state.vx > 0 else -1,
            observation_version=policy.observation_version,
        )
        policy.logits(obs).argmax()
        elapsed = (time.perf_counter_ns() - started) / 1e6
        if i >= 500:
            samples.append(elapsed)
    result = {
        "weights": str(args.weights),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "parameters": sum(v.size for v in policy.weights.values()),
        "weight_bytes": sum(v.nbytes for v in policy.weights.values()),
        "archive_bytes": args.weights.stat().st_size,
        "load_ms": round(load_ms, 3),
        "recorded_states": len(states),
        "iterations": args.iterations,
        "feature_and_inference_ms": {
            name: round(float(np.percentile(samples, percentile)), 4)
            for name, percentile in [
                ("p50", 50),
                ("p95", 95),
                ("p99", 99),
                ("p99_9", 99.9),
                ("max", 100),
            ]
        },
        "p99_budget_ms": args.budget_ms,
        "latency_gate_passed": bool(np.percentile(samples, 99) <= args.budget_ms),
        "scope": "Warm CPU feature extraction plus policy plus argmax; excludes capture/tracking/input.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    if not result["latency_gate_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
