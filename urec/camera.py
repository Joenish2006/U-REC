"""Opens the webcam the same way on Windows and Linux.

Owner: Sathish
"""
from __future__ import annotations

import sys

import cv2


def open_camera(cfg: dict, index: int | None = None) -> cv2.VideoCapture:
    cam = cfg["camera"]
    idx = cam["index"] if index is None else index
    api = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    cap = cv2.VideoCapture(idx, api)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {idx}. Is it in use? Try --camera 1.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam["height"])
    return cap


def read(cap: cv2.VideoCapture, mirror: bool):
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("The camera stopped sending frames.")
    return cv2.flip(frame, 1) if mirror else frame
