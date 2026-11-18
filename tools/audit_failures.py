"""Record a contact candidate and visible gap approach for every completed loss."""

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jumpbot.motion import MovingShape, collides
from jumpbot.navigation import bar_gaps, square_gaps


def distance(state, box):
    x, y, width, height = box[:4]
    dx = max(x - state["x"] - state["w"] / 2, state["x"] - state["w"] / 2 - x - width, 0)
    dy = max(y - state["y"] - state["h"] / 2, state["y"] - state["h"] / 2 - y - height, 0)
    return math.hypot(dx, dy)


def approach(row):
    state = row["state"]
    gaps = [(gap, "bars") for gap in bar_gaps(state["obstacles"], include_moving=True)]
    gaps.extend((gap, "squares") for gap in square_gaps(state["obstacles"]))
    for (top, bottom, left, right), kind in sorted(gaps, reverse=True):
        if bottom < state["y"]:
            middle = (left + right) / 2
            gap_vx = 0
            if kind == "bars":
                gap_vx = next(
                    (b[4] for b in state["obstacles"] if b[0] == right and abs(b[1] - top) <= 5), 0
                )
            return {
                "t": round(row["t"], 3),
                "x": state["x"],
                "vx": round(state["vx"], 1),
                "gap": [left, right],
                "gap_kind": kind,
                "top": top,
                "bottom": bottom,
                "gap_vx": round(gap_vx, 1),
                "outside_px": round(max(left - state["x"], state["x"] - right, 0), 1),
                "moving_away": (state["x"] - middle) * (state["vx"] - gap_vx) > 0,
            }
    return None


def describe_loss(round_rows, end):
    tracked = [
        r
        for r in round_rows
        if r.get("state")
        and r.get("player")
        and 24 <= r["player"]["h"] <= 35
        and (
            24 <= r["player"]["w"] <= 35
            or (r["player"]["w"] < 24 and (r["state"]["x"] <= 5 or r["state"]["x"] >= 415))
        )
    ]
    reads = Counter(
        r["telemetry"]["score"] for r in tracked if r["telemetry"].get("score") is not None
    )
    score = max((value for value, count in reads.items() if count >= 3), default=None)
    result = {"score_estimate": score, "continue_screen_time": round(end, 3)}
    ended = next((r for r in round_rows if r["telemetry"].get("round_end_reason")), None)
    if ended:
        result["round_end_reason"] = ended["telemetry"]["round_end_reason"]
        result["round_end_requested_at"] = round(ended["t"], 3)
    if not tracked:
        return result | {"contact_candidate": "unknown", "why": "No upright tracked frame"}
    # A capture gap during a fall can resemble the first frame of a death
    # animation. A later tracked rise shows that the round was still active.
    consecutive = list(zip(tracked, tracked[1:]))
    last_ascent = next(
        (
            b["t"]
            for a, b in reversed(consecutive)
            if end - 1.1 <= b["t"]
            and 0.008 <= b["t"] - a["t"] <= 0.080
            and b["state"]["ready"]
            and b["state"]["vy"] < -100
            and a["state"]["y"] - b["state"]["y"] >= 2
        ),
        float("-inf"),
    )
    before_gap = [
        a
        for a, b in consecutive
        if a["t"] >= last_ascent
        and end - 1.1 <= a["t"]
        and b["t"] - a["t"] > 0.045
        and b["state"]["y"] - a["state"]["y"] > a["state"]["h"]
    ]
    contact = next(
        (r for r in reversed(before_gap) if r["state"]["ready"] and abs(r["state"]["vy"]) < 1200),
        before_gap[0] if before_gap else tracked[-1],
    )
    state = contact["state"]
    collision_frames = [r for r in tracked if 0 <= contact["t"] - r["t"] <= 0.10]
    candidates = [
        (distance(r["state"], b), r, b) for r in collision_frames for b in r["state"]["obstacles"]
    ]
    gap_frames = [approach(r) for r in tracked if 0.2 <= contact["t"] - r["t"] <= 1.2]
    gap_frames = [r for r in gap_frames if r is not None]
    turns = [r for r in round_rows if r["sent"] and r["t"] <= contact["t"]]
    wall_since = None
    longest_wall_dwell = 0.0
    longest_wall_start = None
    for row in round_rows:
        if row["t"] > contact["t"]:
            break
        if row["state"]["x"] <= 5 or row["state"]["x"] >= 415:
            if wall_since is None:
                wall_since = row["t"]
            dwell = max(row["t"] - wall_since, row["state"].get("wall_dwell", 0))
            if dwell > longest_wall_dwell:
                longest_wall_dwell = dwell
                longest_wall_start = row["t"] - dwell
        else:
            wall_since = None
    result.update(
        last_upright_time=round(contact["t"], 3),
        last_upright_position=[state["x"], state["y"]],
        longest_wall_dwell_s=round(longest_wall_dwell, 2),
        gap_approach=gap_frames[::5],
        moving_away_outside_gap=any(r["outside_px"] > 20 and r["moving_away"] for r in gap_frames),
        contact_candidate="unknown",
    )
    if longest_wall_start is not None and longest_wall_dwell >= 1.0:
        result["longest_wall_started_at"] = round(longest_wall_start, 3)
        incoming = []
        for earlier, later in zip(turns, turns[1:]):
            if not longest_wall_start - 2 <= later["t"] <= longest_wall_start:
                continue
            decision = earlier.get("decision", {})
            planned = decision.get("planned_inputs", [])
            if len(planned) < 2 or decision.get("predicted_horizon", 0) < 1.5:
                continue
            offset, planned_direction = planned[1]
            x = later["state"]["x"]
            if planned_direction == later["sent"] or (x - 210) * later["sent"] < 75:
                continue
            incoming.append(
                {
                    "t": round(later["t"], 3),
                    "x": round(x, 1),
                    "sent": later["sent"],
                    "previous_forecast_at": round(earlier["t"], 3),
                    "previous_turn_at": round(earlier["t"] + offset, 3),
                    "previous_turn_direction": planned_direction,
                }
            )
        if incoming:
            result["replaced_turns_before_wall"] = incoming
    if turns:
        last = turns[-1]
        result["last_jump"] = {
            "t": round(last["t"], 3),
            "direction": last["sent"],
            "forecast_s": last.get("decision", {}).get("predicted_horizon"),
        }
        before = next((r for r in reversed(round_rows) if r["t"] < last["t"]), None)
        if before and all(
            before.get(key) == last.get(key) for key in ("player", "obstacles", "shapes")
        ):
            result["repeated_pose_before_last_jump_ms"] = round(1000 * (last["t"] - before["t"]), 1)
    commits = [
        r
        for r in turns
        if 0 <= contact["t"] - r["t"] <= 1.5
        and r.get("decision", {}).get("predicted_horizon", 0) >= 1.5
    ]
    if commits:
        commit = commits[-1]
        result["last_jump_with_route"] = {
            "t": round(commit["t"], 3),
            "direction": commit["sent"],
            "forecast_s": commit["decision"]["predicted_horizon"],
        }
        collapse = next(
            (
                r
                for r in tracked
                if commit["t"] < r["t"] <= contact["t"]
                and r["state"]["ready"]
                and r.get("decision", {}).get("predicted_horizon", 3) < 1
            ),
            None,
        )
        if collapse:
            result["first_short_forecast"] = {
                "t": round(collapse["t"], 3),
                "forecast_s": collapse["decision"]["predicted_horizon"],
            }
    if candidates:
        clearance, row, box = min(candidates, key=lambda v: v[0])
        if clearance <= 18 and end - contact["t"] < 1.1:
            x, y, width, height = box[:4]
            player = row["state"]
            horizontal = abs(player["x"] - x - width / 2) - (width + player["w"]) / 2
            vertical = abs(player["y"] - y - height / 2) - (height + player["h"]) / 2
            face = (
                ("left" if player["x"] < x + width / 2 else "right")
                if horizontal > vertical
                else ("top" if player["y"] < y + height / 2 else "underside")
            )
            result.update(
                contact_candidate="edge bar"
                if 30 <= height <= 42 and (x < 4 or x + width > 416)
                else "block",
                contact_face=face,
                box=box[:4],
                clearance_px=round(clearance, 1),
                box_velocity_px_s=[round(v, 1) for v in box[4:6]],
                candidate_time=round(row["t"], 3),
            )
    if result["contact_candidate"] == "unknown" and end - contact["t"] < 1.1:
        for row in reversed(collision_frames):
            if not row.get("shapes"):
                continue
            pose = row["state"]
            shapes = [
                MovingShape(np.asarray(h["points"]), (0, 0), (0, 0), 0, 0) for h in row["shapes"]
            ]
            hit, _ = collides(
                np.array([[pose["x"]]]),
                np.array([[pose["y"]]]),
                shapes,
                np.zeros(1),
                pose["w"] / 2,
                pose["h"] / 2,
                6,
            )
            if hit[0]:
                result.update(contact_candidate="hammer", candidate_time=round(row["t"], 3))
                break
    if result["contact_candidate"] == "unknown" and contact.get("roi"):
        _, _, width, height = contact["roi"]
        if state["y"] >= 420 * height / width - state["h"] and state["vy"] > 0:
            result["contact_candidate"] = "bottom boundary"
    return result


def audit(folder):
    results, round_rows = [], []
    with (folder / "frames.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if not line.endswith("\n"):
                break
            row = json.loads(line)
            if row["ui"] == "decline_continue" and round_rows:
                results.append(describe_loss(round_rows, row["t"]))
                round_rows = []
            elif row["ui"] == "restart":
                if any(r["sent"] or r["telemetry"].get("round_end_reason") for r in round_rows):
                    results.append(describe_loss(round_rows, row["t"]))
                round_rows = []
            elif row.get("state"):
                round_rows.append(row)
    return {
        "session": folder.name,
        "losses": results,
        "method": "Candidate from tracked geometry. Confirm contact against the recorded video.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folders", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = [audit(folder) for folder in args.folders]
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for session in report:
        print(
            session["session"],
            [
                (r["score_estimate"], r.get("round_end_reason", r["contact_candidate"]))
                for r in session["losses"]
            ],
        )
