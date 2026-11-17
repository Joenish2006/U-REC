"""Live identification: green box + name for known people, red "Unknown" box otherwise.

Owner: Sathish

Each frame:
  1. YOLOX finds every person's whole body.
  2. Every few frames, the face model finds faces and turns each into an embedding.
  3. Each face is matched to the body it belongs to (its centre sits in the top part
     of the body box). A face with no body around it gets a body box estimated from it.
  4. The tracker follows bodies between frames; each recognised face votes for a name.
  5. Boxes are drawn: green with the name once the votes agree, red "Unknown" otherwise.
"""
from __future__ import annotations

import threading
import time
from typing import Callable

import cv2
import numpy as np

from . import camera, config, term, ui
from .bodies import BodyDetector
from .database import Database
from .embedder import get_embedder
from .matcher import Matcher
from .tracker import Tracker

MIN_INSIDE = 0.6  # at least 60% of the face box must lie inside the body box
UPPER_BODY = 0.6  # faces in the top 60% of a body box are preferred (far-away, full-body view)


def body_from_face(face_box, shape) -> np.ndarray:
    """Estimate a body box from a face when the body detector found none around it."""
    x1, y1, x2, y2 = face_box
    w, h = x2 - x1, y2 - y1
    fh, fw = shape[:2]
    return np.array([max(0, x1 - 0.9 * w), max(0, y1 - 0.4 * h),
                     min(fw - 1, x2 + 0.9 * w), min(fh - 1, y2 + 4.0 * h)], dtype=np.float32)


def _inside(face_box, body) -> float:
    """Fraction of the face box that lies inside the body box."""
    fx1, fy1, fx2, fy2 = face_box
    w = max(0.0, min(fx2, body[2]) - max(fx1, body[0]))
    h = max(0.0, min(fy2, body[3]) - max(fy1, body[1]))
    area = max((fx2 - fx1) * (fy2 - fy1), 1e-6)
    return w * h / area


def assign_faces(faces, boxes: list, shape) -> list[int]:
    """For each face, the index of the body box it belongs to (appending boxes if needed).

    A face belongs to a body when it lies mostly inside it. Close to a webcam the face
    can fill the lower half of the visible body, so position alone isn't required; when
    several bodies qualify, one with the face in its upper part wins, then the smallest.
    """
    owners: list[int] = []
    for f in faces:
        cy = (f.box[1] + f.box[3]) / 2
        best, best_rank = None, None
        for i, b in enumerate(boxes):
            if _inside(f.box, b) < MIN_INSIDE:
                continue
            upper = cy <= b[1] + UPPER_BODY * (b[3] - b[1])
            rank = (not upper, (b[2] - b[0]) * (b[3] - b[1]))
            if best_rank is None or rank < best_rank:
                best, best_rank = i, rank
        if best is None:
            boxes.append(body_from_face(f.box, shape))
            best = len(boxes) - 1
        owners.append(best)
    return owners


def run(cfg: dict, camera_index: int | None = None, show_window: bool = True,
        on_frame: Callable[[np.ndarray], None] | None = None,
        stop: threading.Event | None = None) -> None:
    term.info("Loading models ...")
    embedder = get_embedder(cfg)
    bodies = BodyDetector(cfg)
    with Database(config.database_path(cfg)) as db:
        matcher = Matcher(db.gallery(embedder.name), config.threshold(cfg))
    if matcher.people == 0:
        term.warn(f"Nobody is registered for the '{embedder.name}' backend yet: everyone will be Unknown.")

    ic = cfg["identify"]
    tracker = Tracker(window=int(ic["vote_window"]), max_misses=int(ic["max_misses"]))
    every = max(1, int(ic["recognize_every"]))
    min_votes = int(ic["min_votes"])
    mirror = bool(cfg["camera"]["mirror"])
    cap = camera.open_camera(cfg, camera_index)
    window = "U-REC - identify"
    frame_i, fps, last = 0, 0.0, time.monotonic()

    try:
        while not (stop and stop.is_set()):
            frame = camera.read(cap, mirror)
            boxes = [b for b, _ in bodies.detect(frame)]
            faces = embedder.faces(frame) if frame_i % every == 0 else []
            owners = assign_faces(faces, boxes, frame.shape)
            tracks = tracker.update(boxes)
            for face, owner in zip(faces, owners):
                tracks[owner].vote(*matcher.match(face.embedding))

            for track in tracks:
                name, score = track.label(min_votes)
                if name:
                    ui.label_box(frame, track.box, ui.GREEN, f"{name} {score:.2f}")
                else:
                    ui.label_box(frame, track.box, ui.RED, "Unknown")

            now = time.monotonic()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-6))
            last = now
            ui.banner(frame, [f"U-REC  |  {embedder.name}  |  {matcher.people} registered  |  "
                              f"{fps:4.1f} fps", "q / Esc: quit"])
            frame_i += 1

            if on_frame:
                on_frame(frame)
            if show_window:
                cv2.imshow(window, frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    finally:
        cap.release()
        if show_window:
            cv2.destroyAllWindows()
