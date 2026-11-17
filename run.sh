#!/usr/bin/env bash
# Short way on Linux / macOS:  ./run.sh   ./run.sh identify   ./run.sh register Omkar
cd "$(dirname "$0")"
[[ -x .venv/bin/python ]] || { echo "Run ./setup.sh first."; exit 1; }
exec .venv/bin/python cli.py "$@"
