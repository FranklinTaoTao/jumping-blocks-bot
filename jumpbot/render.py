"""Draw diagnostic overlays without accessing the live game window."""

import cv2
import numpy as np

from .motion import horizontal_extent


def annotate(obs, state=None, decision=None, status=""):
    if obs.image is None:
        _, _, width, height = obs.roi
        image = np.full((round(420 * height / width), 420, 3), 32, np.uint8)
    else:
        image = obs.image.copy()
    for b in obs.obstacles:
        cv2.rectangle(
            image, (int(b.x), int(b.y)), (int(b.x + b.w), int(b.y + b.h)), (255, 140, 0), 1
        )
    for shape in obs.shapes:
        cv2.polylines(image, [shape.points.astype(np.int32)], True, (255, 0, 255), 2)
    if state:
        for index, (x, y, w, h, vx, vy) in enumerate(state.obstacles):
            if index >= len(state.obstacle_bounds):
                continue
            lower, upper = state.obstacle_bounds[index]
            if upper <= lower:
                continue
            for dt in (0.25, 0.5):
                xx, width = horizontal_extent(x, w, vx, dt, lower, upper)
                yy = y + vy * dt
                cv2.rectangle(
                    image,
                    (round(float(xx)), round(yy)),
                    (round(float(xx + width)), round(yy + h)),
                    (130, 110, 30),
                    1,
                )
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
    if decision and decision.reset_window is not None:
        early, late = decision.reset_window
        lines.append(f"Reset window {early * 1000:.0f}-{late * 1000:.0f} ms")
    if state:
        lines.append(f"pos=({state.x:.0f}, {state.y:.0f}) | v=({state.vx:.0f}, {state.vy:.0f})")
        lines.append(
            f"camera={state.camera_speed:.0f} px/s | blocks={len(state.obstacles)} | hammers={len(state.shapes)}"
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
