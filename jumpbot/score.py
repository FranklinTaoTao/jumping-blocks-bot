"""Template-only recognition of the game's score numerals."""

from pathlib import Path

import cv2
import numpy as np


class ScoreReader:
    def __init__(self):
        with np.load(Path(__file__).parent / "assets" / "digits.npz") as data:
            self.templates = {int(k): self.normalize(data[k]) for k in data.files}

    @staticmethod
    def normalize(glyph):
        # Keep aspect ratio; center the glyph in a common binary canvas.
        scale = 52 / glyph.shape[0]
        resized = cv2.resize(glyph, (max(1, round(glyph.shape[1] * scale)), 52))
        out = np.zeros((56, 54), np.float32)
        w = min(50, resized.shape[1])
        out[2:54, (54 - w) // 2 : (54 - w) // 2 + w] = resized[:, :w] / 255
        return out

    def read(self, board):
        grey = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
        mask = np.uint8(grey[35:135] < 78) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = sorted([cv2.boundingRect(c) for c in contours])
        boxes = [b for b in boxes if 40 <= b[3] <= 72 and 3 < b[2] < 60 and 80 < b[0] < 340]
        if not 1 <= len(boxes) <= 4:
            return None
        if max(b[1] for b in boxes) - min(b[1] for b in boxes) > 4:
            return None
        digits = []
        for x, y, w, h in boxes:
            g = self.normalize(mask[y : y + h, x : x + w])
            errors = sorted(
                (float(np.mean((g - template) ** 2)), digit)
                for digit, template in self.templates.items()
            )
            if errors[0][0] > 0.11:
                return None
            digits.append(str(errors[0][1]))
        return int("".join(digits))
