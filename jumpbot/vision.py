from dataclasses import dataclass, field

import cv2
import numpy as np

from .motion import Shape

_SOLID_KERNEL = np.ones((9, 9), np.uint8)
_SCORE_KERNEL = np.ones((23, 23), np.uint8)


@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    w: float
    h: float

    @property
    def cx(self):
        return self.x + self.w / 2

    @property
    def cy(self):
        return self.y + self.h / 2

    def intersects(self, other, margin=0):
        return (
            self.x < other.x + other.w + margin
            and self.x + self.w > other.x - margin
            and self.y < other.y + other.h + margin
            and self.y + self.h > other.y - margin
        )


@dataclass
class Observation:
    image: np.ndarray | None
    player: Rect | None
    obstacles: list[Rect]
    active: bool
    roi: tuple[int, int, int, int]
    shapes: list = field(default_factory=list)


class Detector:
    """All measurements use a board width of 420 units, independent of DPI."""

    def __init__(self, roi=None, recover_score_blocks=True):
        self.roi = roi
        self.locked_roi = roi is not None
        self.frames = 0
        self.previous_bars = []
        self.recover_score_blocks = recover_score_blocks

    @staticmethod
    def find_board(frame):
        # Find the broad, flat grey playfield, then reject its thin connected
        # window border using row/column coverage. Works with recording padding.
        scale = min(1.0, 420 / frame.shape[1])
        small = cv2.resize(frame, None, fx=scale, fy=scale)
        grey = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        blue, green, red = cv2.split(small)
        spread = cv2.subtract(
            cv2.max(cv2.max(blue, green), red), cv2.min(cv2.min(blue, green), red)
        )
        flat = spread < 12
        eligible = flat & (grey > 170) & (grey < 245)
        if eligible.sum() < 1000:
            return None
        mode = np.bincount(grey[eligible], minlength=256).argmax()
        mask = np.uint8(flat & (abs(grey.astype(int) - mode) < 8))
        _, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        if len(stats) < 2:
            return None
        x, y, w, h, area = max(stats[1:], key=lambda s: s[4])
        if area < small.shape[0] * small.shape[1] * 0.2 or h < w:
            return None
        crop = mask[y : y + h, x : x + w]
        rows = np.where(crop.mean(1) > 0.65)[0]
        if len(rows) < h * 0.35:
            return None
        # Long runs avoid an isolated grey title-bar scanline.
        starts = [r for r in rows if r + 8 < h and crop[r : r + 8].mean() > 0.65]
        if not starts:
            return None
        top, bottom = starts[0], rows[-1] + 1
        white_rows = np.flatnonzero(
            (grey[: min(len(grey) // 5, 130), x : x + w] > 247).mean(1) > 0.55
        )
        if len(white_rows) >= 8 and white_rows[0] < y + top:
            # The stable white title bar ends at a black separator. Obstacles
            # inside the board must not move the inferred board boundary.
            top = max(0, int(white_rows[-1]) + 2 - y)
        bottom = h
        cols = np.where(crop[top:bottom].mean(0) > 0.45)[0]
        if len(cols) < w * 0.6:
            return None
        left, right = cols[0], cols[-1] + 1
        x0, y0 = int(round((x + left) / scale)), int(round((y + top) / scale))
        x1 = min(frame.shape[1], int(round((x + right) / scale)))
        y1 = min(frame.shape[0], int(round((y + bottom) / scale)))
        return x0, y0, x1 - x0, y1 - y0

    def observe(self, frame):
        self.frames += 1
        if not self.locked_roi and self.roi is None and self.frames % 6 == 1:
            found = self.find_board(frame)
            if found:
                self.roi = found
        if self.roi is None:
            return None
        x, y, w, h = self.roi
        if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
            self.roi = None
            return None
        image = frame[y : y + h, x : x + w]
        image = image.copy() if w == 420 else cv2.resize(image, (420, round(h * 420 / w)))
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, light = cv2.threshold(grey, 170, 255, cv2.THRESH_BINARY)
        light_balance = 2 * cv2.countNonZero(light) - grey.size
        # When the two middle pixels straddle the threshold, their average
        # determines the median. Otherwise just count the light pixels.
        active = light_balance > 0 if light_balance else float(np.median(grey)) > 170
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # Outfits change the headband color (red/yellow/etc.), not the square
        # body geometry. Detect saturation, rather than one costume's hue.
        red = cv2.inRange(hsv, (0, 71, 101), (179, 255, 255))
        contours, _ = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        _, dark = cv2.threshold(grey, 77, 255, cv2.THRESH_BINARY_INV)
        for contour in contours:
            rx, ry, rw, rh = cv2.boundingRect(contour)
            if not (8 < rw < 55 and 2 <= rh < 28 and ry > 90):
                continue
            x0, y0 = max(0, rx - 3), max(0, ry - 6)
            patch = dark[y0 : min(image.shape[0], ry + 42), x0 : min(420, rx + rw + 8)]
            bodies, _ = cv2.findContours(patch, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for body in bodies:
                bx, by, bw, bh = cv2.boundingRect(body)
                if (
                    12 < bw < 42
                    and 8 <= bh < 42
                    and abs(x0 + bx + bw / 2 - (rx + rw / 2)) < 9
                    and y0 + by <= ry + rh + 8
                    and y0 + by + bh >= ry + rh - 3
                ):
                    # The red band splits the black head into components. Include
                    # the band/forehead in the collision box, not just the torso.
                    top = min(y0 + by, ry - 3)
                    for cap in bodies:
                        tx, ty, tw, th = cv2.boundingRect(cap)
                        if abs(tx - bx) <= 3 and abs(tw - bw) <= 3 and ry - 10 <= y0 + ty < ry:
                            top = min(top, y0 + ty)
                    candidates.append(
                        (cv2.contourArea(body), Rect(x0 + bx, top, bw, y0 + by + bh - top))
                    )
        player = max(candidates, key=lambda c: c[0])[1] if candidates else None
        obstacles = []
        shapes = []
        # Detect edge bars row-by-row, even when a score numeral touches them.
        # A connected-component rectangle test alone loses the entire bar when
        # that happens, because the combined score+bar contour is irregular.
        filled = dark > 0
        raw_filled = filled
        occluded_rows = np.zeros(len(filled), dtype=bool)
        # At higher scores a colored promotional badge can cover the left end
        # of an incoming bar. Bridge only that known occluded patch on rows
        # where the bar is confirmed at the board edge; never interpret the
        # badge's light pixels as a safe opening through a black bar.
        colored = cv2.inRange(hsv[:110, :160], (0, 46, 76), (179, 255, 255))
        if cv2.countNonZero(colored) > 15:
            cx, cy, cw, ch = cv2.boundingRect(colored)
            # The portrait can be grayscale; the colored caption alone does
            # not describe the badge's whole occluding rectangle.
            ox1 = min(160, max(135, cx + cw + 9))
            oy0, oy1 = max(0, cy - 28), min(110, cy + ch + 9)
            occluded_rows[oy0:oy1] = True
            patched = filled.copy()
            edge_rows = np.flatnonzero(filled[oy0:oy1, 2]) + oy0
            patched[edge_rows, :ox1] = True
            filled = patched
        for side in ("left", "right"):
            edge = filled[:, 2] if side == "left" else filled[:, -3]
            transitions = np.diff(np.r_[False, edge, False].astype(np.int8))
            for top, bottom in zip(
                np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1)
            ):
                if not 6 <= bottom - top <= 250:
                    continue
                rows = filled[top:bottom]
                if side == "left":
                    lengths = np.argmin(rows[:, 2:], axis=1) + 2
                    extent = int(np.median(lengths))
                    visible = ~occluded_rows[top:bottom]
                    if np.count_nonzero(visible) >= 2:
                        # The badge must not turn a narrow bar into a 135px bar.
                        # Prefer its unobscured rows; bridge only missing evidence.
                        raw_lengths = np.argmin(raw_filled[top:bottom, 2:], axis=1) + 2
                        extent = int(np.median(raw_lengths[visible]))
                    elif occluded_rows[top:bottom].any():
                        matches = [
                            b
                            for b in self.previous_bars
                            if b.x < 4 and abs(b.y - top) < 65 and abs(b.h - (bottom - top)) < 25
                        ]
                        if matches:
                            previous = min(matches, key=lambda b: abs(b.y - top))
                            extent = int(previous.w + 1)
                    if extent >= 8:
                        obstacles.append(Rect(1, int(top), extent - 1, int(bottom - top)))
                else:
                    lengths = np.argmin(rows[:, -3::-1], axis=1) + 2
                    extent = int(np.median(lengths))
                    if extent >= 8:
                        obstacles.append(
                            Rect(420 - extent, int(top), extent - 1, int(bottom - top))
                        )
        if player:
            obstacles = [
                b
                for b in obstacles
                if not (
                    b.x >= player.x - 1
                    and b.x + b.w <= player.x + player.w + 1
                    and b.y >= player.y - 1
                    and b.y + b.h <= player.y + player.h + 1
                )
            ]
        self.previous_bars = obstacles.copy() if active else []
        # Remove thin text strokes before finding solid central rectangles.
        solid = cv2.morphologyEx(dark, cv2.MORPH_OPEN, _SOLID_KERNEL)
        contours, _ = cv2.findContours(solid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            bx, by, bw, bh = cv2.boundingRect(contour)
            # Incoming blocks/hammers are useful evidence even while clipped
            # at the top. Waiting for their whole contour loses reaction time.
            if not (6 <= bh <= 250 and bw >= 6):
                continue
            rect = Rect(bx, by, bw, bh)
            # Only remove the actual torso contour. A nearby obstacle must
            # remain visible to the planner right up to contact.
            if (
                player
                and rect.x >= player.x - 1
                and rect.x + rect.w <= player.x + player.w + 1
                and rect.y >= player.y
                and rect.y + rect.h <= player.y + player.h + 1
            ):
                continue
            if bx < 4 or bx + bw > 416:
                continue  # represented by the row detector above
            fill = cv2.contourArea(contour) / (bw * bh)
            occluded_square = by < 130 and bw >= 50 and 0.8 < bw / bh < 1.25 and fill > 0.65
            # Walls and squares are solid rectangles. Score digits and icons
            # contain holes/curves; only edge-connected bars may enter score area.
            if fill < 0.78 and not occluded_square:
                if cv2.contourArea(contour) > 450 and bw > 40 and bh > 40:
                    moments = cv2.moments(contour)
                    if moments["m00"]:
                        cx = moments["m10"] / moments["m00"]
                        cy = moments["m01"] / moments["m00"]
                        angle = 0.5 * np.arctan2(
                            2 * moments["mu11"], moments["mu20"] - moments["mu02"]
                        )
                        hull = cv2.convexHull(contour).reshape(-1, 2).astype(float)
                        shapes.append(Shape(hull, cx, cy, float(angle), moments["m00"]))
                continue
            if by < 130 and fill < 0.90 and not occluded_square:
                continue
            if bw / bh < 0.7:
                continue
            obstacles.append(rect)
        if not self.recover_score_blocks:
            return Observation(image, player, obstacles, active, self.roi, shapes)
        # A score digit can touch an incoming 30px square and turn their joint
        # contour into an irregular shape. A wider opening removes numeral
        # strokes while retaining the square's solid interior.
        cores = cv2.morphologyEx(dark[:180], cv2.MORPH_OPEN, _SCORE_KERNEL)
        contours, _ = cv2.findContours(cores, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            bx, by, bw, bh = cv2.boundingRect(contour)
            if (
                by < 130
                and 25 <= bw <= 36
                and 25 <= bh <= 36
                and 0.85 < bw / bh < 1.18
                and cv2.contourArea(contour) / (bw * bh) > 0.85
                and not any(
                    abs(b.x - bx) < 3
                    and abs(b.y - by) < 3
                    and abs(b.w - bw) < 4
                    and abs(b.h - bh) < 4
                    for b in obstacles
                )
            ):
                obstacles.append(Rect(bx, by, bw, bh))
        return Observation(image, player, obstacles, active, self.roi, shapes)


def camera_shift(previous, current):
    """Consensus vertical displacement of wall bars and stable squares."""
    proposals = []
    for a in previous:
        edge_bar = (a.x < 4 or a.x + a.w > 416) and 30 <= a.h <= 42
        large_square = a.w >= 60 and 0.8 < a.w / a.h < 1.25
        if not (edge_bar or large_square):
            continue
        for b in current:
            if abs(a.h - b.h) >= 4 or abs(a.y - b.y) >= 65:
                continue
            aligned = abs(a.x - b.x) < 3 and abs(a.w - b.w) < 4
            same_wall = (a.x < 4 and b.x < 4) or (a.x + a.w > 416 and b.x + b.w > 416)
            # A sliding gap changes its width against each wall without
            # changing its vertical position in the stage.
            if aligned or (edge_bar and same_wall and 30 <= b.h <= 42 and abs(a.w - b.w) <= 16):
                proposals.append(b.y - a.y)
    if not proposals:
        return 0.0, False
    best = max(proposals, key=lambda d: sum(abs(e - d) < 3 for e in proposals))
    group = [d for d in proposals if abs(d - best) < 3]
    return float(np.median(group)), True
