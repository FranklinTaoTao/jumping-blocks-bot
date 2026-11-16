"""Export per-round evidence from a saved Jumpbot recording. No live UI access."""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jumpbot.score import ScoreReader


def review(source, output):
    rows = [json.loads(line) for line in (source / "frames.jsonl").open()]
    metadata = json.loads((source / "recording.json").read_text())
    timestamps = np.array(metadata["timestamps"])
    output.mkdir(parents=True, exist_ok=True)
    starts = []
    for row in rows:
        if row.get("ui") == "restart" and (not starts or row["t"] - starts[-1] > 2):
            starts.append(row["t"])
    if not starts:
        starts = [rows[0]["t"]]
    rounds = []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else rows[-1]["t"] + 0.01
        part = [row for row in rows if start + 0.2 < row["t"] < end]
        death = None
        for j in range(1, len(part)):
            if not part[j].get("state") and part[j - 1].get("state"):
                if any(
                    r.get("ui") == "decline_continue" and 0 < r["t"] - part[j]["t"] < 2
                    for r in part[j:]
                ):
                    death = part[j]["t"]
                    break
        rounds.append(
            {
                "round": i + 1,
                "start": start + 0.2,
                "end": end,
                "completed": death is not None,
                "death_t": death,
                "peak": 0,
                "peak_t": None,
            }
        )
    reader = ScoreReader()
    times = np.array([row["t"] for row in rows])
    cap = cv2.VideoCapture(str(source / "diagnostic.mp4"))
    peaks = {}
    index = 0
    while cap.grab():
        if index % 3 == 0 and index < len(timestamps):
            t = timestamps[index]
            part = next((r for r in rounds if r["start"] <= t < r["end"]), None)
            row = rows[min(len(rows) - 1, int(np.searchsorted(times, t)))]
            # Do not count the previous round's score screen or an unplayed
            # opening frame as a fresh high score.
            if part and row.get("state") and row.get("active"):
                ok, frame = cap.retrieve()
                if ok:
                    score = reader.read(frame[46:790, :420])
                    if score is not None and score > part["peak"]:
                        part["peak"], part["peak_t"] = score, float(t)
                        peaks[part["round"]] = frame.copy()
        index += 1
    cap.release()
    for part in rounds:
        n = part["round"]
        if n in peaks:
            cv2.imwrite(str(output / f"round{n}-peak.png"), peaks[n])
        if part["death_t"] is None:
            continue
        death = part["death_t"]
        cap = cv2.VideoCapture(str(source / "diagnostic.mp4"))
        sheet = np.zeros((800, 2100, 3), np.uint8)
        for k, t in enumerate(np.linspace(death - 1.6, death + 0.2, 10)):
            idx = int(np.argmin(abs(timestamps - t)))
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if ok:
                sheet[k // 5 * 400 : (k // 5 + 1) * 400, k % 5 * 420 : (k % 5 + 1) * 420] = (
                    cv2.resize(frame, (420, 400))
                )
        cv2.imwrite(str(output / f"round{n}-contact.jpg"), sheet)
        first = int(np.searchsorted(timestamps, death - 3))
        last = min(len(timestamps) - 1, int(np.searchsorted(timestamps, death + 0.3)))
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        writer = cv2.VideoWriter(
            str(output / f"round{n}-failure.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, size
        )
        cap.set(cv2.CAP_PROP_POS_FRAMES, first)
        for _ in range(first, last + 1):
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
        cap.release()
        writer.release()
    report = {
        "source": str(source.resolve()),
        "rounds": rounds,
        "recording_dropped_frames": metadata["dropped"],
        "recording_error": metadata["error"],
    }
    (output / "rounds.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/review"))
    args = parser.parse_args()
    review(args.source, args.output)
