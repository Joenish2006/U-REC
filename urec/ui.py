"""Drawing helpers shared by the register and identify screens.

Owner: Sathish
"""
from __future__ import annotations

import cv2
import numpy as np

# OpenCV colours are BGR
GREEN = (70, 200, 60)
RED = (60, 60, 230)
YELLOW = (0, 200, 255)
WHITE = (245, 245, 245)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def label_box(frame: np.ndarray, box, color, text: str, thickness: int = 3) -> None:
    """Draw a rectangle with a filled label on its top edge."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    x1, x2 = max(0, x1), min(w - 1, x2)
    y1, y2 = max(0, y1), min(h - 1, y2)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    scale, weight = 0.8, 2
    (tw, th), base = cv2.getTextSize(text, FONT, scale, weight)
    top = y1 - th - base - 10
    if top < 0:  # no room above the box: put the label just inside it
        top = y1
    cv2.rectangle(frame, (x1, top), (x1 + tw + 14, top + th + base + 10), color, -1)
    cv2.putText(frame, text, (x1 + 7, top + th + 5), FONT, scale, WHITE, weight, cv2.LINE_AA)


def banner(frame: np.ndarray, lines: list[str]) -> None:
    """Dark strip across the top with one or more lines of text."""
    height = 18 + 34 * len(lines)
    strip = frame[:height].copy()
    cv2.rectangle(strip, (0, 0), (frame.shape[1], height), (20, 20, 20), -1)
    cv2.addWeighted(strip, 0.65, frame[:height], 0.35, 0, frame[:height])
    for i, line in enumerate(lines):
        scale = 0.85 if i == 0 else 0.75
        cv2.putText(frame, line, (16, 38 + 34 * i), FONT, scale, WHITE, 2, cv2.LINE_AA)


def progress(frame: np.ndarray, fraction: float, color=GREEN) -> None:
    h, w = frame.shape[:2]
    x1, x2, y = 16, w - 16, h - 24
    cv2.rectangle(frame, (x1, y - 10), (x2, y + 10), (60, 60, 60), -1)
    fill = x1 + int((x2 - x1) * max(0.0, min(1.0, fraction)))
    cv2.rectangle(frame, (x1, y - 10), (fill, y + 10), color, -1)
