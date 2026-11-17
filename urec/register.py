"""Face ID-style registration: a guided scan from several angles.

Owner: Nishanth

The person follows five prompts (straight, turn one way, the other way, tilt one way,
the other way). A frame is only kept when exactly one face is visible, it is close
enough, sharp, and turned the way the prompt asks. Only the embeddings are returned,
never the images.

Head pose is estimated from the 5 face landmarks. Turning and tilting are checked
relative to the person's own "straight" pose, in either direction, so it works the
same with a mirrored or unmirrored camera.
"""
from __future__ import annotations

import time

import cv2
import numpy as np

from collections import Counter

from . import camera, term, ui

STEPS = [
    ("straight", "Look straight at the camera"),
    ("yaw_a", "Slowly turn your head to one side"),
    ("yaw_b", "Now turn to the other side"),
    ("pitch_a", "Tilt your head up or down"),
    ("pitch_b", "Now tilt it the other way"),
]
STRAIGHT_YAW = 0.12  # |yaw| below this counts as facing the camera
TURN_YAW = 0.18      # how far a turn must go from straight
TILT_PITCH = 0.08    # how far a tilt must go from straight
CAPTURE_GAP = 0.15   # seconds between kept frames


def head_pose(landmarks) -> tuple[float, float]:
    """(yaw, pitch) from 5 landmarks. Works with either landmark order."""
    lm = np.asarray(landmarks, dtype=np.float32)
    eye_mid = (lm[0] + lm[1]) / 2
    mouth_mid = (lm[3] + lm[4]) / 2
    eye_dist = max(float(np.linalg.norm(lm[0] - lm[1])), 1e-6)
    yaw = (lm[2, 0] - eye_mid[0]) / eye_dist
    pitch = (lm[2, 1] - eye_mid[1]) / max(float(mouth_mid[1] - eye_mid[1]), 1e-6)
    return float(yaw), float(pitch)


def sharpness(frame: np.ndarray, box, size: int = 112) -> float:
    """Blur score of the face. The face is resized to a fixed size first, so the score
    doesn't depend on how close the person is. Typical webcam: sharp 45-65, blurred < 20."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in box)
    crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
    if crop.size == 0:
        return 0.0
    crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


class PoseGuide:
    """Checks whether the current head pose satisfies the current step."""

    def __init__(self):
        self.yaws: list[float] = []
        self.pitches: list[float] = []
        self.yaw_sign = 0.0
        self.pitch_sign = 0.0

    @property
    def base_yaw(self) -> float:
        return float(np.median(self.yaws)) if self.yaws else 0.0

    @property
    def base_pitch(self) -> float | None:
        return float(np.median(self.pitches)) if self.pitches else None

    def ok(self, step: str, yaw: float, pitch: float) -> bool:
        if step == "straight":
            if abs(yaw) < STRAIGHT_YAW:
                self.yaws.append(yaw)
                self.pitches.append(pitch)
                return True
            return False
        if step.startswith("yaw"):
            d = yaw - self.base_yaw
            if step == "yaw_a":
                if abs(d) > TURN_YAW:
                    self.yaw_sign = self.yaw_sign or float(np.sign(d))
                    return True
                return False
            sign = self.yaw_sign or 1.0
            return d * sign < -TURN_YAW
        if self.base_pitch is None:  # straight step was skipped: use the first pitch seen
            self.pitches.append(pitch)
        d = pitch - self.base_pitch  # type: ignore[operator]
        if step == "pitch_a":
            if abs(d) > TILT_PITCH:
                self.pitch_sign = self.pitch_sign or float(np.sign(d))
                return True
            return False
        sign = self.pitch_sign or 1.0
        return d * sign < -TILT_PITCH


def scan(cfg: dict, embedder, name: str, camera_index: int | None = None) -> np.ndarray | None:
    """Run the guided scan. Returns an (n, d) array of embeddings, or None if cancelled."""
    rc = cfg["register"]
    per_step = int(rc["frames_per_step"])
    mirror = bool(cfg["camera"]["mirror"])
    cap = camera.open_camera(cfg, camera_index)
    window = f"U-REC - register {name}"
    guide = PoseGuide()
    vectors: list[np.ndarray] = []
    skipped: list[str] = []
    rejected: Counter[str] = Counter()
    step_i, step_count = 0, 0
    step_start = last_capture = time.monotonic()

    try:
        while step_i < len(STEPS):
            frame = camera.read(cap, mirror)
            view = frame.copy()
            step, instruction = STEPS[step_i]
            now = time.monotonic()
            faces = embedder.faces(frame)
            status = instruction

            if not faces:
                status = "No face found - face the camera, in good light"
                rejected["no face"] += 1
            elif len(faces) > 1:
                status = "Only one person in view, please"
                rejected["more than one face"] += 1
                for f in faces:
                    ui.label_box(view, f.box, ui.RED, "?")
            else:
                f = faces[0]
                yaw, pitch = head_pose(f.landmarks)
                color = ui.YELLOW
                if f.box[2] - f.box[0] < rc["min_face_px"]:
                    status = "Come a little closer"
                    rejected["face too small"] += 1
                elif sharpness(frame, f.box) < rc["min_sharpness"]:
                    status = "Hold still"
                    rejected["blurry"] += 1
                elif guide.ok(step, yaw, pitch) and now - last_capture >= CAPTURE_GAP:
                    vectors.append(f.embedding)
                    step_count += 1
                    last_capture = now
                    color = ui.GREEN
                    status = "Good - keep going"
                ui.label_box(view, f.box, color, name)

            if step_count >= per_step or now - step_start > rc["step_timeout"]:
                if step_count < per_step:
                    skipped.append(instruction)
                step_i, step_count = step_i + 1, 0
                step_start = now

            done = min(step_i, len(STEPS)) + step_count / per_step
            ui.banner(view, [f"Registering {name}  -  step {min(step_i + 1, len(STEPS))}/{len(STEPS)}",
                             status, "q / Esc: cancel"])
            ui.progress(view, done / len(STEPS))
            cv2.imshow(window, view)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                term.info("Registration cancelled.")
                return None
    finally:
        cap.release()
        cv2.destroyAllWindows()

    for s in skipped:
        term.warn(f"Timed out on '{s}'")
    if len(vectors) < int(rc["min_total"]):
        term.error(f"Only {len(vectors)} good frames (need {rc['min_total']}).")
        if rejected:
            reasons = ", ".join(f"{why}: {n}" for why, n in rejected.most_common())
            term.info(f"Frames rejected · {reasons}")
            if rejected.most_common(1)[0][0] == "blurry":
                term.info("Hold still in good light, or lower register.min_sharpness in config.toml.")
        return None
    return np.vstack(vectors).astype(np.float32)
