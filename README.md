# U-REC

Real-time face recognition from a webcam, run from the command line.
Known people get a **green box around their whole body with their name**. Everyone
else gets a **red box labelled "Unknown"**. New people are added with a guided,
Face ID-style scan.

> University project (ENEMY RECOGNITION) by **Team Rocognance**, B.Tech Robotics and
> Automation, Lovely Professional University.

## How it works

```
camera frame
  ├─ YOLOX ───────────────► whole-body boxes
  └─ face model ──────────► faces → embeddings (lists of numbers)
                                   │
            matcher: closest registered embedding ≥ threshold?  yes → name, no → Unknown
                                   │
     tracker: follows each body between frames, votes on the name so it doesn't flicker
                                   │
               green box + name   /   red box + "Unknown"
```

The face model is swappable in `config.toml`:

| Backend | Models | Accuracy | Licence |
|---|---|---|---|
| `opencv` (default) | YuNet + SFace, 128 numbers | Good | MIT / Apache-2.0: free for any use |
| `insightface` | SCRFD + ArcFace (`buffalo_l`), 512 numbers | Best | **Non-commercial research only** |

## Setup

Needs Python 3.11+ and a webcam. Models (about 30–300 MB) download on first run into
`~/.urec/models` and `~/.insightface`, never into this folder.

| | Linux / macOS | Windows |
|---|---|---|
| Free backend | `./setup.sh` | `setup.bat` |
| Best accuracy | `./setup.sh --insightface` | `setup.bat --insightface` ¹ |
| Run | `./run.sh` | `run.bat` |

¹ InsightFace on Windows needs [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
with **Desktop development with C++**.

The setup scripts use the NVIDIA GPU automatically when `nvidia-smi` is found.

**The long way (any OS):**
```bash
python -m venv .venv
# activate it: source .venv/bin/activate   or   .venv\Scripts\activate
pip install -r requirements.txt            # + requirements-insightface.txt for the best backend
python cli.py init                         # creates config.toml
python cli.py download-models
python cli.py
```

## Quick start

```bash
git clone https://github.com/Joenish2006/U-REC
cd U-REC
./setup.sh --insightface        # Windows: setup.bat --insightface   (one time)
./run.sh register YourName      # Windows: run.bat register YourName
./run.sh identify               # Windows: run.bat identify   (q to quit)
```

**Registering:** sit in good light with only you in view, then follow the prompts:
look straight, turn your head one way and then the other, tilt it up or down and then
the other way. The box around your face turns green each time a frame is kept. It
takes about 10–20 seconds and saves around 30 samples.

**Identifying:** everyone starts as a red "Unknown" box. A registered person turns
green with their name after about a second. Press `q` or Esc to close.

## Usage

Run with no arguments for the menu:

```
=== U-REC ===   backend: insightface   people: 5
 [1] Identify (camera window)
 [2] Identify in browser (localhost)
 [3] Register a person
 [4] List people
 [5] Delete a person
 [6] Reset database
 [q] Quit
```

Or call a command directly (`run.sh` / `run.bat` / `python cli.py`):

| Command | What it does |
|---|---|
| `identify` | live recognition in a window (`q` or Esc to quit) |
| `identify --web` | same view in the browser at http://localhost:5000 |
| `register NAME` | guided face scan: straight, turn both ways, tilt both ways |
| `list` | registered people |
| `delete NAME` | remove one person |
| `reset` | remove everyone (type `YES` to confirm) |
| `info` | backend, threshold, GPU/CPU, file locations |
| `--camera 1` | use another camera (put it before the command) |

## Troubleshooting

| Problem | Fix |
|---|---|
| `Could not open camera` | Close other apps using the webcam (Zoom, Meet, browser), or try `--camera 1` |
| `Permission denied` on `./run.sh` | Run `chmod +x run.sh setup.sh` once |
| Registration says `Only N good frames` | It prints why frames were rejected. `blurry`: hold still in better light, or lower `min_sharpness` in `config.toml` |
| Strangers get a name | Raise `threshold_*` in `config.toml` |
| A registered person stays "Unknown" | Lower `threshold_*` a little, or register them again in better light |
| Slow (low fps) | Set `model = "yolox_tiny"`, `insightface_model = "buffalo_s"`, `recognize_every = 3` |
| Model download fails with `CERTIFICATE_VERIFY_FAILED` | The network is intercepting HTTPS (campus/office Wi-Fi). Run setup once on a mobile hotspot or home network |
| Switched backend and everyone is "Unknown" | Each backend stores its own data: register people again (`list` shows who needs it) |

## Project layout

| File | Job | Owner |
|---|---|---|
| `urec/embedder.py` | face detection + embeddings, OpenCV / InsightFace switch | Joenish |
| `urec/matcher.py` | name or Unknown, using the threshold | Joenish |
| `urec/config.py`, `models.py`, `runtime.py` | settings, model downloads, GPU choice | Joenish |
| `urec/register.py` | Face ID-style guided scan with quality checks | Nishanth |
| `urec/bodies.py` | YOLOX whole-body detection | Sathish |
| `urec/tracker.py` | follows people between frames, vote smoothing | Sathish |
| `urec/identify.py`, `ui.py`, `camera.py` | live loop and drawing | Sathish |
| `cli.py`, `urec/web.py`, `setup.*`, `run.*` | menu, commands, launchers, localhost stream | Omkar |
| `urec/database.py`, `tests/` | SQLite store, unit tests | Thanigai |

Run the tests with `python -m unittest` (no camera or models needed).

## Privacy

- Only **embeddings** (numbers) are stored, in `data/urec.db`. No photos are saved.
- `data/`, model files and `config.toml` are in `.gitignore` and never uploaded.
- Register only people who have agreed to it. Everyone else stays "Unknown".
- The web view listens on `127.0.0.1` only, so other devices can't see the camera.
- Limitation: a normal webcam is 2D, so a printed photo could fool it. Blink-based
  liveness detection is planned future work.

## Licences

- **U-REC source code:** MIT (see `LICENSE`).
- **YuNet** face detector: MIT · **SFace** face recogniser: Apache-2.0 (OpenCV Zoo).
- **YOLOX** body detector: Apache-2.0 (Megvii).
- **InsightFace** library: MIT. Its **pretrained models are for non-commercial research
  use only**. They are not included here; the optional `insightface` backend downloads
  them on first run. Use the `opencv` backend for anything commercial.
