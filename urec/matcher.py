"""Decides whose face an embedding is, or "Unknown".

Owner: Joenish

Each registered person has many stored embeddings (one per scan frame). A new face
gets the name of the closest stored embedding (cosine similarity), but only if that
similarity reaches the threshold. Below the threshold it is Unknown, so strangers are
never forced into a teammate's name.
"""
from __future__ import annotations

import numpy as np


class Matcher:
    def __init__(self, gallery: dict[str, np.ndarray], threshold: float):
        self.threshold = float(threshold)
        names: list[str] = []
        rows: list[np.ndarray] = []
        for name, vectors in gallery.items():
            vectors = np.atleast_2d(np.asarray(vectors, dtype=np.float32))
            rows.append(vectors)
            names.extend([name] * len(vectors))
        self.names = names
        self.matrix = np.vstack(rows) if rows else np.zeros((0, 0), dtype=np.float32)

    @property
    def people(self) -> int:
        return len(set(self.names))

    def match(self, embedding: np.ndarray) -> tuple[str | None, float]:
        """Return (name or None, best similarity)."""
        if not self.names:
            return None, 0.0
        sims = self.matrix @ np.asarray(embedding, dtype=np.float32)
        i = int(np.argmax(sims))
        score = float(sims[i])
        return (self.names[i] if score >= self.threshold else None), score
