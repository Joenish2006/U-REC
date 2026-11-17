"""Follows each person between frames and smooths their label by voting.

Owner: Sathish

Without this, the label would flicker whenever one frame is misread. Each track keeps
the last few recognition results ("votes"); it is shown as a name only when that name
has enough votes. A person who turns away keeps their label while still tracked.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field

import numpy as np


def iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Track:
    id: int
    box: np.ndarray
    votes: deque
    misses: int = 0
    scores: dict = field(default_factory=dict)

    def vote(self, name: str | None, score: float) -> None:
        self.votes.append(name)
        if name is not None:
            self.scores[name] = score

    def label(self, min_votes: int) -> tuple[str | None, float]:
        counts = Counter(self.votes)
        named = [(n, c) for n, c in counts.items() if n is not None]
        if not named:
            return None, 0.0
        name, count = max(named, key=lambda nc: nc[1])
        if count >= min_votes and count >= counts.get(None, 0):
            return name, self.scores.get(name, 0.0)
        return None, 0.0


class Tracker:
    def __init__(self, window: int = 8, max_misses: int = 15, iou_threshold: float = 0.3):
        self.window = window
        self.max_misses = max_misses
        self.iou_threshold = iou_threshold
        self.tracks: list[Track] = []
        self._next_id = 1

    def update(self, boxes: list) -> list[Track]:
        """Match this frame's boxes to tracks. Returns one track per box, same order."""
        pairs = sorted(
            ((iou(t.box, b), ti, bi) for ti, t in enumerate(self.tracks) for bi, b in enumerate(boxes)),
            reverse=True)
        result: list[Track | None] = [None] * len(boxes)
        used: set[int] = set()
        for score, ti, bi in pairs:
            if score < self.iou_threshold:
                break
            if ti in used or result[bi] is not None:
                continue
            used.add(ti)
            track = self.tracks[ti]
            track.box = np.asarray(boxes[bi], dtype=np.float32)
            track.misses = 0
            result[bi] = track
        for ti, track in enumerate(self.tracks):
            if ti not in used:
                track.misses += 1
        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]
        for bi, box in enumerate(boxes):
            if result[bi] is None:
                track = Track(self._next_id, np.asarray(box, dtype=np.float32),
                              deque(maxlen=self.window))
                self._next_id += 1
                self.tracks.append(track)
                result[bi] = track
        return result  # type: ignore[return-value]
