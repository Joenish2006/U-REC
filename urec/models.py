"""Downloads model files on first use into ~/.urec/models (never into the repo).

Owner: Joenish

Every model here has a permissive licence:
  YuNet (face detection)       MIT          opencv/opencv_zoo
  SFace (face recognition)     Apache-2.0   opencv/opencv_zoo
  YOLOX (body detection)       Apache-2.0   Megvii-BaseDetection/YOLOX
InsightFace models are NOT downloaded here: the insightface package fetches them
itself into ~/.insightface (non-commercial research licence).
"""
from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path

_ZOO = "https://github.com/opencv/opencv_zoo/raw/main/models"
_YOLOX = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0"

MODELS: dict[str, tuple[str, str]] = {
    "yunet": ("face_detection_yunet_2023mar.onnx",
              f"{_ZOO}/face_detection_yunet/face_detection_yunet_2023mar.onnx"),
    "sface": ("face_recognition_sface_2021dec.onnx",
              f"{_ZOO}/face_recognition_sface/face_recognition_sface_2021dec.onnx"),
    "yolox_nano": ("yolox_nano.onnx", f"{_YOLOX}/yolox_nano.onnx"),
    "yolox_tiny": ("yolox_tiny.onnx", f"{_YOLOX}/yolox_tiny.onnx"),
    "yolox_s": ("yolox_s.onnx", f"{_YOLOX}/yolox_s.onnx"),
    "yolox_m": ("yolox_m.onnx", f"{_YOLOX}/yolox_m.onnx"),
    "yolox_l": ("yolox_l.onnx", f"{_YOLOX}/yolox_l.onnx"),
}


def model_dir() -> Path:
    return Path(os.environ.get("UREC_MODEL_DIR", Path.home() / ".urec" / "models"))


def get(key: str) -> Path:
    """Return the local path of a model, downloading it the first time."""
    if key not in MODELS:
        raise KeyError(f"Unknown model {key!r}. Choose from: {', '.join(MODELS)}")
    filename, url = MODELS[key]
    path = model_dir() / filename
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    print(f"Downloading {filename} ...", flush=True)

    def progress(blocks: int, block_size: int, total: int) -> None:
        if total > 0:
            pct = min(100, blocks * block_size * 100 // total)
            sys.stdout.write(f"\r  {pct:3d}% of {total / 1e6:.1f} MB")
            sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, part, progress)
    except Exception as exc:
        part.unlink(missing_ok=True)
        hint = ""
        if "CERTIFICATE_VERIFY_FAILED" in str(exc):
            hint = ("\n  Your network is intercepting HTTPS (common on campus/office Wi-Fi). "
                    "Run setup again on a mobile hotspot or home network, or download the file "
                    f"in a browser and save it as {path}")
        raise RuntimeError(f"Could not download {url}: {exc}{hint}") from exc
    print()
    part.replace(path)
    return path
