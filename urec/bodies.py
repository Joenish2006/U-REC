"""Whole-body person detection with YOLOX (Apache-2.0), run through ONNX Runtime.

Owner: Sathish
"""
from __future__ import annotations

import cv2
import numpy as np

from . import models
from .runtime import providers

PERSON = 0  # COCO class id


class BodyDetector:
    def __init__(self, cfg: dict):
        import onnxruntime as ort

        b = cfg["bodies"]
        self.score = float(b["score"])
        self.nms = float(b["nms"])
        self.session = ort.InferenceSession(
            str(models.get(b["model"])), providers=providers(cfg["recognition"]["device"]))
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        h, w = inp.shape[2], inp.shape[3]
        self.size = (h if isinstance(h, int) else 640, w if isinstance(w, int) else 640)
        self._grids, self._strides = self._make_grids(*self.size)

    @staticmethod
    def _make_grids(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
        grids, strides = [], []
        for s in (8, 16, 32):
            xv, yv = np.meshgrid(np.arange(w // s), np.arange(h // s))
            grid = np.stack((xv, yv), axis=2).reshape(-1, 2)
            grids.append(grid)
            strides.append(np.full((grid.shape[0], 1), s))
        return np.concatenate(grids).astype(np.float32), np.concatenate(strides).astype(np.float32)

    def _preprocess(self, frame: np.ndarray) -> tuple[np.ndarray, float]:
        h, w = self.size
        ratio = min(h / frame.shape[0], w / frame.shape[1])
        resized = cv2.resize(frame, (int(frame.shape[1] * ratio), int(frame.shape[0] * ratio)),
                             interpolation=cv2.INTER_LINEAR)
        padded = np.full((h, w, 3), 114, dtype=np.uint8)
        padded[:resized.shape[0], :resized.shape[1]] = resized
        return np.ascontiguousarray(padded.transpose(2, 0, 1), dtype=np.float32)[None], ratio

    def detect(self, frame: np.ndarray) -> list[tuple[np.ndarray, float]]:
        """Return [(box x1,y1,x2,y2, score)] for every person in the frame."""
        blob, ratio = self._preprocess(frame)
        out = self.session.run(None, {self.input_name: blob})[0][0]
        xy = (out[:, :2] + self._grids) * self._strides
        wh = np.exp(out[:, 2:4]) * self._strides
        scores = out[:, 4] * out[:, 5 + PERSON]
        keep = scores >= self.score
        if not np.any(keep):
            return []
        xy, wh, scores = xy[keep], wh[keep], scores[keep]
        boxes = np.concatenate([xy - wh / 2, xy + wh / 2], axis=1) / ratio
        fh, fw = frame.shape[:2]
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, fw - 1)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, fh - 1)
        xywh = [[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])] for b in boxes]
        idx = cv2.dnn.NMSBoxes(xywh, scores.tolist(), self.score, self.nms)
        return [(boxes[i], float(scores[i])) for i in np.array(idx).ravel()]
