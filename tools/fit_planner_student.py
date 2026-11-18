"""Distill successful procedural routes into the unchanged three-action actor."""

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
from numba import set_num_threads
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.rl_policy import NumpyPolicy
from tools.train_rl import Policy, evaluate


def mirror(observations, labels):
    obs = observations.copy()
    obs[:, [0, 2, 5]] *= -1
    objects = obs[:, 8:].reshape(-1, 12, 7)
    objects[:, :, [0, 4]] *= -1
    return obs, np.where(labels == 0, 0, 3 - labels)


@torch.no_grad()
def classification(policy, obs, labels, accepted=None):
    logits = policy(obs)[0]
    report = {
        "loss": round(float(F.cross_entropy(logits, labels)), 5),
        "accuracy": round(float((logits.argmax(1) == labels).float().mean()), 5),
    }
    if accepted is not None:
        report["accepted_action_fraction"] = round(
            float(accepted.gather(1, logits.argmax(1)[:, None]).float().mean()), 5
        )
    return report


@torch.no_grad()
def validation_rollout(policy, device, pixels=False):
    count = 16 if pixels else 32
    result = evaluate(
        policy,
        device,
        2,
        seed=43301 if pixels else 33301,
        n=count,
        processing=0.020,
        tiles=12,
        big_start=5,
        large_x=(100, 225),
        pair_probability=0.35,
        pixel_envs=count if pixels else 0,
    )
    return {
        "cleared": sum(r["reason"] == 3 for r in result),
        "count": len(result),
        "mean_climb": round(float(np.mean([r["climb"] for r in result])), 2),
    }, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    set_num_threads(4)
    torch.manual_seed(612)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with np.load(args.train) as archive:
        x, y = archive["observations"], archive["labels"]
        accepted = archive["accepted"] if "accepted" in archive else None
    with np.load(args.validation) as archive:
        vx, vy = archive["observations"], archive["labels"]
        val_accepted = archive["accepted"] if "accepted" in archive else None
    train_info = json.loads((args.train.parent / "collection.json").read_text())
    val_info = json.loads((args.validation.parent / "collection.json").read_text())
    assert len(x) == sum(r["examples"] for r in train_info["courses"] if r["reason"] == 3)
    assert len(vx) == sum(r["examples"] for r in val_info["courses"] if r["reason"] == 3)
    assert not (
        {r["seed"] for r in train_info["courses"]} & {r["seed"] for r in val_info["courses"]}
    )
    mx, my = mirror(x, y)
    x, y = np.concatenate([x, mx]), np.concatenate([y, my])
    if accepted is not None:
        accepted = torch.as_tensor(
            np.concatenate([accepted, accepted[:, [0, 2, 1]]]), device=device
        )
        val_accepted = torch.as_tensor(val_accepted, device=device)
    if not len(x) or not len(vx):
        raise ValueError("Both independently collected datasets must contain successful courses")
    x, y, vx, vy = [torch.as_tensor(v, device=device) for v in (x, y, vx, vy)]
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    policy = Policy(observation_version=3).to(device)
    policy.load_state_dict(checkpoint["policy"])
    with torch.no_grad():
        assert (policy(x)[0].gather(1, y[:, None]) > -1e8).all()
        assert (policy(vx)[0].gather(1, vy[:, None]) > -1e8).all()
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4, eps=1e-5)
    with torch.no_grad():
        reference_probs = policy(x)[0].softmax(1)
    baseline, baseline_routes = validation_rollout(policy, device)
    pixel_baseline, _ = validation_rollout(policy, device, pixels=True)
    best_key = (pixel_baseline["cleared"], baseline["cleared"], pixel_baseline["mean_climb"])
    best_weights = copy.deepcopy(policy.state_dict())
    best_epoch = 0
    summaries = [
        {
            "epoch": 0,
            "classification": classification(policy, vx, vy, val_accepted),
            "closed_loop": baseline,
            "pixel_loop": pixel_baseline,
        }
    ]
    print(
        json.dumps(
            {"train_examples": len(x), "validation_examples": len(vx), "baseline": summaries[0]}
        ),
        flush=True,
    )
    for epoch in range(args.epochs):
        for idx in torch.randperm(len(x), device=device).split(512):
            logits = policy(x[idx])[0]
            valid = (logits > -1e8).float()
            if accepted is None:
                targets = 0.99 * F.one_hot(y[idx], 3) + 0.01 * valid / valid.sum(1, keepdim=True)
                loss = -(targets * logits.log_softmax(1)).sum(1).mean()
            else:
                allowed = accepted[idx] & (valid > 0)
                informative = ((~allowed) & (valid > 0)).any(1)
                allowed_logits = logits.masked_fill(~allowed, -1e9)
                mass_loss = logits.logsumexp(1) - allowed_logits.logsumexp(1)
                loss = (mass_loss * informative).sum() / informative.sum().clamp(min=1)
                loss = loss + 0.1 * F.kl_div(
                    logits.log_softmax(1), reference_probs[idx], reduction="batchmean"
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
        if epoch % 10 == 9:
            routes, _ = validation_rollout(policy, device)
            pixel_routes, _ = validation_rollout(policy, device, pixels=True)
            summary = {
                "epoch": epoch + 1,
                "train": classification(policy, x, y, accepted),
                "validation": classification(policy, vx, vy, val_accepted),
                "closed_loop": routes,
                "pixel_loop": pixel_routes,
            }
            summaries.append(summary)
            print(json.dumps(summary), flush=True)
            key = (pixel_routes["cleared"], routes["cleared"], pixel_routes["mean_climb"])
            if key > best_key:
                best_key = key
                best_epoch = epoch + 1
                best_weights = copy.deepcopy(policy.state_dict())
    policy.load_state_dict(best_weights)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4, eps=1e-5)
    checkpoint.update(
        policy=policy.state_dict(),
        optimizer=optimizer.state_dict(),
        observation_version=3,
        processing=0.020,
        distillation=True,
        selected_epoch=best_epoch,
        source_train=str(args.train),
        source_validation=str(args.validation),
        tiles=train_info["tiles"],
        large_x=train_info["large_x"],
        pair_probability=train_info["pair_probability"],
    )
    torch.save(checkpoint, args.output / "checkpoint.pt")
    np.savez(
        args.output / "weights.npz",
        _observation_version=3,
        **{k: v.detach().cpu().numpy() for k, v in policy.state_dict().items()},
    )
    with torch.no_grad():
        expected = policy(vx)[0].cpu().numpy()
    actual = NumpyPolicy(args.output / "weights.npz").logits(vx.cpu().numpy())
    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
    courses = json.loads(Path("artifacts/regression-audit/known-courses.json").read_text())
    recorded = evaluate(policy, device, 2, courses=courses, processing=0.020)
    result = {
        "selected_epoch": best_epoch,
        "training": summaries,
        "training_examples": len(x),
        "validation_examples": len(vx),
        "baseline_procedural": baseline_routes,
        "recorded_holdout": recorded,
        "recorded_cleared": sum(r["reason"] == 3 for r in recorded),
        "scope": "Procedural route distillation; recording courses evaluated only.",
    }
    (args.output / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps({"selected_epoch": best_epoch, "recorded_cleared": result["recorded_cleared"]}),
        flush=True,
    )


if __name__ == "__main__":
    main()
