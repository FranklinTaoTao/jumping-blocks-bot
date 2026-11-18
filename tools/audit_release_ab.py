"""Compare posted mouse release modes from a single randomized live recording."""

import argparse
import json
from pathlib import Path

import numpy as np
from audit_launches import evaluate


def visible_reversals(rows):
    for i, row in enumerate(rows):
        mode = row["telemetry"].get("release_mode")
        direction = row["sent"]
        state = row.get("state")
        if not mode or not direction or not state or direction * state["vx"] > -100:
            continue
        if not 35 < state["x"] < 385:
            continue
        prior = rows[i - 1] if i else None
        if not prior or not prior.get("player") or row["t"] - prior["t"] > 0.050:
            continue
        x = row["player"]["x"] + row["player"]["w"] / 2
        prior_x = prior["player"]["x"] + prior["player"]["w"] / 2
        if (x - prior_x) * direction > -1:
            continue
        pressed = row["telemetry"]["input_pressed_at"]
        confirmed = 0
        for following in rows[i + 1 : i + 13]:
            delay = following["t"] - pressed
            player = following.get("player")
            if following["sent"] or not player or delay > 0.15:
                break
            next_x = player["x"] + player["w"] / 2
            if not 15 < next_x < 405:
                break
            movement = (next_x - x) * direction
            if delay >= 0 and movement >= 0.6:
                if confirmed == 0:
                    onset = delay
                confirmed += 1
                if confirmed == 2:
                    yield mode, pressed, 1000 * onset
                    break
            elif movement < -0.6:
                confirmed = 0
            x = next_x


def summary(values):
    if not values:
        return {"n": 0}
    p10, p25, median, p75, p90, p95 = np.percentile(values, [10, 25, 50, 75, 90, 95])
    return {
        "n": len(values),
        "p10_ms": round(p10, 2),
        "median_ms": round(median, 2),
        "p95_ms": round(p95, 2),
        "iqr_ms": round(p75 - p25, 2),
        "p90_minus_p10_ms": round(p90 - p10, 2),
    }


def difference_intervals(samples):
    """Resample time blocks to keep nearby observations together."""
    data = np.array([(t // 10, mode == "background", delay) for mode, t, delay in samples])
    blocks = [data[data[:, 0] == key] for key in np.unique(data[:, 0])] if len(data) else []
    if len(blocks) < 5:
        return None
    rng = np.random.default_rng(142)
    differences = []
    for pick in rng.integers(len(blocks), size=(1200, len(blocks))):
        selected = np.concatenate([blocks[i] for i in pick])
        groups = [selected[selected[:, 1] == arm, 2] for arm in (0, 1)]
        if any(len(group) == 0 for group in groups):
            continue
        q = np.percentile(groups[1], [25, 50, 75, 95]) - np.percentile(groups[0], [25, 50, 75, 95])
        differences.append((q[1], q[2] - q[0], q[3]))
    return dict(
        zip(
            ("median_ms", "iqr_ms", "p95_ms"),
            np.percentile(differences, [2.5, 97.5], axis=0).T.round(2).tolist(),
        )
    )


def report(folder):
    with (folder / "frames.jsonl").open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.endswith("\n")]
    physics = json.loads((folder / "configuration.json").read_text(encoding="utf-8"))["physics"]
    mode_at = {round(r["t"], 4): r["telemetry"].get("release_mode") for r in rows if r["sent"]}
    visible = list(visible_reversals(rows))
    fitted = evaluate(folder, rows, physics["gravity"], physics["jump"], 0.16, include_samples=True)
    fitted = fitted["observed_reset"]["samples"]
    results = {}
    for mode in ("blocking", "background"):
        sent = [r for r in rows if r["sent"] and r["telemetry"].get("release_mode") == mode]
        results[mode] = {
            "delivered": len(sent),
            "first_visible_reversal": summary(
                [delay for name, _, delay in visible if name == mode]
            ),
            "fitted_reset": summary([r["delay_ms"] for r in fitted if mode_at[r["t"]] == mode]),
            "press_to_next_capture_complete": summary(
                [
                    lag_ms
                    for r, following in zip(rows, rows[1:])
                    if r["sent"]
                    and r["telemetry"].get("release_mode") == mode
                    and 0
                    <= (
                        lag_ms := 1000 * (following["t"] - r["telemetry"]["input_pressed_at"])
                        + following["telemetry"]["frame_age_at_capture_ms"]
                    )
                    <= 150
                ]
            ),
        }
    results["background_minus_blocking_95pct_intervals"] = difference_intervals(visible)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = json.dumps(report(args.folder), indent=2) + "\n"
    args.output.write_text(output, encoding="utf-8")
    print(output, end="")
