"""Settings. Defaults live here; config.toml (gitignored) overrides them.

Owner: Joenish
"""
from __future__ import annotations

import copy
import re
import shutil
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILE = ROOT / "config.toml"
EXAMPLE_FILE = ROOT / "config.example.toml"
BACKENDS = ("opencv", "insightface")

DEFAULTS: dict = {
    "recognition": {
        "backend": "opencv",
        "insightface_model": "buffalo_l",
        "threshold_opencv": 0.42,
        "threshold_insightface": 0.45,
        "min_face_score": 0.6,
        "device": "auto",
    },
    "camera": {"index": 0, "width": 1280, "height": 720, "mirror": True},
    "bodies": {"model": "yolox_s", "score": 0.5, "nms": 0.45},
    "identify": {"recognize_every": 2, "vote_window": 8, "min_votes": 2, "max_misses": 15},
    "register": {
        "frames_per_step": 6,
        "step_timeout": 12.0,
        "min_total": 10,
        "min_face_px": 80,
        "min_sharpness": 25.0,
    },
    "web": {"host": "127.0.0.1", "port": 5000},
    "paths": {"database": "data/urec.db"},
}


def _merge(base: dict, override: dict) -> None:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value


def load(path: str | Path | None = None) -> dict:
    """Return the settings: DEFAULTS merged with config.toml (or the given file)."""
    cfg = copy.deepcopy(DEFAULTS)
    file = Path(path) if path else CONFIG_FILE
    if file.exists():
        with open(file, "rb") as fh:
            _merge(cfg, tomllib.load(fh))
    elif path:
        raise FileNotFoundError(f"Config file not found: {file}")
    backend = cfg["recognition"]["backend"]
    if backend not in BACKENDS:
        raise ValueError(f"recognition.backend must be one of {BACKENDS}, got {backend!r}")
    return cfg


def threshold(cfg: dict) -> float:
    rec = cfg["recognition"]
    return float(rec[f"threshold_{rec['backend']}"])


def database_path(cfg: dict) -> Path:
    p = Path(cfg["paths"]["database"])
    return p if p.is_absolute() else ROOT / p


def init_file() -> bool:
    """Create config.toml from the example if it is missing. Returns True if created."""
    if CONFIG_FILE.exists():
        return False
    shutil.copyfile(EXAMPLE_FILE, CONFIG_FILE)
    return True


def set_backend(name: str) -> None:
    """Switch recognition.backend in config.toml (creating it if needed)."""
    if name not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}")
    init_file()
    text = CONFIG_FILE.read_text(encoding="utf-8")
    new, count = re.subn(r'(?m)^backend\s*=\s*"[^"]*"', f'backend = "{name}"', text, count=1)
    if count == 0:
        raise ValueError("No 'backend = ...' line found in config.toml")
    CONFIG_FILE.write_text(new, encoding="utf-8")
