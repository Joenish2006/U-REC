#!/usr/bin/env bash
# One-time setup for Linux / macOS.
#   ./setup.sh                 free OpenCV backend
#   ./setup.sh --insightface   also install InsightFace (best accuracy, non-commercial models)
set -euo pipefail
cd "$(dirname "$0")"

WANT_INSIGHTFACE=0
[[ "${1:-}" == "--insightface" ]] && WANT_INSIGHTFACE=1
VPY=.venv/bin/python

if command -v uv >/dev/null 2>&1; then
  uv venv --python 3.12 .venv
  install()   { uv pip install --python "$VPY" "$@"; }
  uninstall() { uv pip uninstall --python "$VPY" "$@"; }
  reinstall() { uv pip install --python "$VPY" --reinstall --no-deps "$@"; }
else
  PY=""
  for c in python3.12 python3.11 python3; do
    if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(sys.version_info < (3, 11))'; then
      PY=$c; break
    fi
  done
  [[ -n "$PY" ]] || { echo "Python 3.11 or newer is required."; exit 1; }
  "$PY" -m venv .venv
  "$VPY" -m pip install --upgrade pip
  install()   { "$VPY" -m pip install "$@"; }
  uninstall() { "$VPY" -m pip uninstall -y "$@"; }
  reinstall() { "$VPY" -m pip install --force-reinstall --no-deps "$@"; }
fi

install -r requirements.txt

if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  echo "NVIDIA GPU found: installing onnxruntime-gpu"
  uninstall onnxruntime
  install onnxruntime-gpu
fi

"$VPY" cli.py init
if [[ $WANT_INSIGHTFACE == 1 ]]; then
  install -r requirements-insightface.txt
  # insightface pulls in opencv-python-headless, which has no camera window: put the full one back
  uninstall opencv-python-headless || true
  reinstall opencv-python
  "$VPY" cli.py set-backend insightface
fi

"$VPY" cli.py download-models
echo
echo "Setup complete. Start with: ./run.sh"
