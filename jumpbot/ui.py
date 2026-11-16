"""Classical template matching for known non-game screens.

Templates are loaded once, not captured/written/read in the control loop.
Unknown dialogs never produce a UI action.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class UIAction:
    kind: str
    x: float
    y: float
    confidence: float


class ScreenHandler:
    def __init__(self):
        with np.load(Path(__file__).parent / "assets" / "ui_templates.npz") as data:
            self.templates = {name: data[name].copy() for name in data.files}
        self.previous = None
        self.previous_point = None
        self.stable = 0
        self.last_click = -100.0

    def match(self, grey, name):
        template = self.templates[name]
        if grey.shape[0] < template.shape[0] or grey.shape[1] < template.shape[1]:
            return 0.0, (0, 0)
        result = cv2.matchTemplate(grey, template, cv2.TM_CCOEFF_NORMED)
        _, confidence, _, position = cv2.minMaxLoc(result)
        return confidence, position

    def inspect(self, frame):
        factor = 420 / frame.shape[1]
        small = cv2.resize(frame, (420, round(frame.shape[0] * factor)))
        grey = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        if "outfit_label" in self.templates:
            outfit, op = self.match(grey, "outfit_label")
            use, up = self.match(grey, "outfit_use")
            if outfit > 0.9 and use > 0.9 and up[0] > 220 and up[1] > op[1] + 150:
                th, tw = self.templates["outfit_use"].shape
                return UIAction(
                    "use_outfit",
                    (up[0] + tw / 2) / factor,
                    (up[1] + th / 2) / factor,
                    min(outfit, use),
                )
        if "ad_close" in self.templates:
            ad, ap = self.match(grey, "ad_close")
            # This is the circular thin-stroke X above an interstitial card,
            # not the title-bar close control. Confirm the white card below it.
            if ad > 0.91 and ap[0] > 280 and 150 < ap[1] < 420:
                card = grey[min(len(grey) - 1, ap[1] + 200) : min(len(grey), ap[1] + 400), 50:370]
                if card.size and np.mean(card > 235) > 0.55:
                    th, tw = self.templates["ad_close"].shape
                    return UIAction(
                        "close_ad", (ap[0] + tw / 2) / factor, (ap[1] + th / 2) / factor, ad
                    )
        score, sp = self.match(grey, "score")
        restart, rp = self.match(grey, "restart")
        if (
            score > 0.88
            and restart > 0.88
            and sp[1] < grey.shape[0] * 0.5
            and rp[1] > grey.shape[0] * 0.5
        ):
            th, tw = self.templates["restart"].shape
            return UIAction(
                "restart", (rp[0] + tw / 2) / factor, (rp[1] + th / 2) / factor, min(score, restart)
            )
        cont, cp = self.match(grey, "continue_label")
        close, xp = self.match(grey, "continue_close")
        if cont > 0.88 and close > 0.88 and 90 < xp[0] - cp[0] < 190 and -70 < xp[1] - cp[1] < 10:
            th, tw = self.templates["continue_close"].shape
            return UIAction(
                "decline_continue",
                (xp[0] + tw / 2) / factor,
                (xp[1] + th / 2) / factor,
                min(cont, close),
            )
        # A verified helmet/no template can be added without touching gameplay.
        if "helmet_label" in self.templates and "helmet_no" in self.templates:
            helmet, hp = self.match(grey, "helmet_label")
            no, np_ = self.match(grey, "helmet_no")
            if helmet > 0.88 and no > 0.88 and np_[1] > hp[1]:
                th, tw = self.templates["helmet_no"].shape
                return UIAction(
                    "decline_helmet",
                    (np_[0] + tw / 2) / factor,
                    (np_[1] + th / 2) / factor,
                    min(helmet, no),
                )
        return None

    def ready(self, action, t):
        kind = action.kind if action else None
        point = (action.x, action.y) if action else None
        unmoved = bool(
            point
            and self.previous_point
            and abs(point[0] - self.previous_point[0]) + abs(point[1] - self.previous_point[1]) < 8
        )
        self.stable = self.stable + 1 if kind and kind == self.previous and unmoved else 1
        self.previous = kind
        self.previous_point = point
        return action is not None and self.stable >= 3 and t - self.last_click > 1.0

    def clicked(self, t):
        self.last_click = t
        self.stable = 0
