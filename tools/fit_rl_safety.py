"""Diagnose and fit the tiny actor on informative simulated action outcomes.

Training and validation use independent procedural seeds. No recorded holdout
layout supplies labels. Only training runs counterfactual physics; deployment
still consists of a single classifier evaluation.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from numba import set_num_threads

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.rl_env import CourseEnv
from tools.train_rl import Policy, evaluate


@torch.no_grad()
def collect(policy, device, seed, steps, n=128):
    env = CourseEnv(n, seed=seed, stage=1)
    observations, labels = [], []
    for _ in range(steps):
        obs = torch.as_tensor(env.obs, device=device)
        logits = policy(obs)[0]
        risk = env.action_risks()
        valid = logits.cpu().numpy() > -1e8
        informative = ((risk < 0.5) & valid).any(1) & ((risk > 0.5) & valid).any(1)
        if informative.any():
            observations.append(env.obs[informative].copy())
            labels.append(risk[informative])
        actions = torch.distributions.Categorical(logits=logits).sample().cpu().numpy()
        env.step(actions)
    return np.concatenate(observations), np.concatenate(labels)


@torch.no_grad()
def assess(policy, obs, risk):
    logits = policy(obs)[0]
    probs = logits.softmax(-1)
    chosen = logits.argmax(-1)
    bad = risk.gather(1, chosen[:, None]).squeeze(1)
    return {
        "unsafe_greedy_fraction": round(float(bad.mean()), 4),
        "unsafe_probability": round(float((probs * risk).sum(1).mean()), 4),
        "count": len(obs),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--collect-steps", type=int, default=1200)
    parser.add_argument("--epochs", type=int, default=60)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    set_num_threads(4)
    torch.manual_seed(317)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    policy = Policy().to(device)
    policy.load_state_dict(checkpoint["policy"])
    print("Collecting independent procedural training and validation states", flush=True)
    x, y = collect(policy, device, 314159, args.collect_steps)
    vx, vy = collect(policy, device, 271828, args.collect_steps // 2)
    np.savez_compressed(
        args.output / "dataset.npz", train_obs=x, train_risk=y, val_obs=vx, val_risk=vy
    )
    x, y, vx, vy = [torch.as_tensor(v, device=device) for v in (x, y, vx, vy)]
    result = {"before_train": assess(policy, x, y), "before_val": assess(policy, vx, vy)}
    print(json.dumps(result), flush=True)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4, eps=1e-5)
    for epoch in range(args.epochs):
        order = torch.randperm(len(x), device=device)
        for idx in order.split(256):
            logits = policy(x[idx])[0]
            safe_logits = logits.masked_fill(y[idx] > 0.5, -1e9)
            loss = (logits.logsumexp(1) - safe_logits.logsumexp(1)).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
        if epoch % 10 == 9:
            print(
                json.dumps(
                    {
                        "epoch": epoch + 1,
                        "train": assess(policy, x, y),
                        "val": assess(policy, vx, vy),
                    }
                ),
                flush=True,
            )
    result.update(after_train=assess(policy, x, y), after_val=assess(policy, vx, vy))
    checkpoint.update(
        policy=policy.state_dict(),
        optimizer=optimizer.state_dict(),
        observation_version=2,
        safety_prefit=True,
    )
    torch.save(checkpoint, args.output / "checkpoint.pt")
    np.savez(
        args.output / "weights.npz",
        _observation_version=2,
        **{k: v.detach().cpu().numpy() for k, v in policy.state_dict().items()},
    )
    courses = json.loads(Path("artifacts/regression-audit/known-courses.json").read_text())
    result["recorded_holdout"] = evaluate(policy, device, 1, courses=courses)
    result["procedural"] = evaluate(policy, device, 1)
    (args.output / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "after_val": result["after_val"],
                "recorded_cleared": sum(r["reason"] == 3 for r in result["recorded_holdout"]),
                "procedural_cleared": sum(r["reason"] == 3 for r in result["procedural"]),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
