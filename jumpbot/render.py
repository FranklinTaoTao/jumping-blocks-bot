"""Draw diagnostic overlays without accessing the live game window."""

import cv2
import numpy as np


def annotate(obs, state=None, decision=None, status=""):
    image = obs.image.copy()
    for b in obs.obstacles:
        cv2.rectangle(
            image, (int(b.x), int(b.y)), (int(b.x + b.w), int(b.y + b.h)), (255, 140, 0), 1
        )
    for shape in obs.shapes:
        cv2.polylines(image, [shape.points.astype(np.int32)], True, (255, 0, 255), 2)
    if state:
        for shape in state.shapes:
            future = shape.predict([0.2, 0.4])
            for polygon in future:
                cv2.polylines(image, [polygon.astype(np.int32)], True, (150, 30, 150), 1)
    if obs.player:
        p = obs.player
        cv2.rectangle(
            image, (int(p.x), int(p.y)), (int(p.x + p.w), int(p.y + p.h)), (60, 200, 30), 1
        )
    if decision is not None and len(decision.path):
        points = decision.path.astype(np.int32)
        cv2.polylines(image, [points], False, (0, 160, 255), 2)
        cv2.circle(image, tuple(points[-1]), 4, (0, 160, 255), -1)
    label = {-1: "LEFT", 0: "WAIT", 1: "RIGHT"}[decision.action] if decision else "NO TRACK"
    lines = [status, label + (" | " + decision.reason if decision else "")]
    if state:
        lines.append(
            f"v=({state.vx:.0f}, {state.vy:.0f}) | blocks={len(state.obstacles)} hammers={len(state.shapes)}"
        )
    # Keep diagnostics outside the board: a header painted over the first 60
    # pixels hides exactly the incoming hazards that need early inspection.
    footer = np.full((len(lines) * 20, 420, 3), 32, np.uint8)
    for i, line in enumerate(lines):
        cv2.putText(
            footer,
            line,
            (5, 15 + i * 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.39,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
    return np.vstack([image, footer])
