"""Face detection + embedding behind ONE interface, so the model can be swapped.

Owner: Joenish

    backend = "opencv"       YuNet + SFace, 128 numbers, MIT / Apache-2.0 (default in the repo)
    backend = "insightface"  SCRFD + ArcFace, 512 numbers, best accuracy, non-commercial models

Everything else in U-REC only sees `Face` objects and calls `embedder.faces(frame)`.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from . import models
from .runtime import providers


@dataclass
class Face:
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels
    score: float                            # detection confidence 0..1
    landmarks: np.ndarray                   # 5x2: two eyes, nose, two mouth corners
    embedding: np.ndarray                   # L2-normalised vector


def _normalise(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).ravel()
    return v / (np.linalg.norm(v) + 1e-9)


class OpenCVEmbedder:
    name = "opencv"

    def __init__(self, cfg: dict):
        self.min_score = float(cfg["recognition"]["min_face_score"])
        self.detector = cv2.FaceDetectorYN.create(
            str(models.get("yunet")), "", (320, 320), self.min_score, 0.3, 5000)
        self.recognizer = cv2.FaceRecognizerSF.create(str(models.get("sface")), "")
        self._size: tuple[int, int] | None = None

    def faces(self, frame: np.ndarray) -> list[Face]:
        h, w = frame.shape[:2]
        if self._size != (w, h):
            self.detector.setInputSize((w, h))
            self._size = (w, h)
        _, rows = self.detector.detect(frame)
        result: list[Face] = []
        if rows is None:
            return result
        for row in rows:
            x, y, bw, bh = (float(v) for v in row[:4])
            aligned = self.recognizer.alignCrop(frame, row)
            result.append(Face(
                box=(x, y, x + bw, y + bh),
                score=float(row[14]),
                landmarks=row[4:14].reshape(5, 2).astype(np.float32),
                embedding=_normalise(self.recognizer.feature(aligned)),
            ))
        return result


class InsightFaceEmbedder:
    name = "insightface"

    def __init__(self, cfg: dict):
        try:
            from insightface.app import FaceAnalysis
        except ImportError as exc:
            raise RuntimeError(
                "InsightFace is not installed. Run `./setup.sh --insightface` "
                "(or `setup.bat --insightface` on Windows), or set backend = \"opencv\" "
                "in config.toml.") from exc
        rec = cfg["recognition"]
        self.app = FaceAnalysis(
            name=rec["insightface_model"],
            allowed_modules=["detection", "recognition"],
            providers=providers(rec["device"]),
        )
        self.app.prepare(ctx_id=0, det_thresh=float(rec["min_face_score"]), det_size=(640, 640))

    def faces(self, frame: np.ndarray) -> list[Face]:
        return [
            Face(
                box=tuple(float(v) for v in f.bbox),
                score=float(f.det_score),
                landmarks=np.asarray(f.kps, dtype=np.float32),
                embedding=_normalise(f.normed_embedding),
            )
            for f in self.app.get(frame)
        ]


def get_embedder(cfg: dict):
    backend = cfg["recognition"]["backend"]
    if backend == "insightface":
        return InsightFaceEmbedder(cfg)
    return OpenCVEmbedder(cfg)
