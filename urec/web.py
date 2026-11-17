"""Streams the identify view to a browser at http://localhost:5000.

Owner: Omkar

Binds to 127.0.0.1 by default, so only this computer can see the camera feed.
"""
from __future__ import annotations

import threading
import time

import cv2
from flask import Flask, Response

from . import identify

PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>U-REC</title>
<style>
  body { margin: 0; background: #11161f; color: #eef1f5; font-family: system-ui, sans-serif;
         display: flex; flex-direction: column; align-items: center; }
  h1 { font-size: 20px; font-weight: 600; margin: 16px; }
  img { max-width: 100vw; max-height: calc(100vh - 70px); }
</style></head>
<body><h1>U-REC live</h1><img src="/stream" alt="Live camera with recognition boxes"></body></html>"""


def serve(cfg: dict, camera_index: int | None = None) -> None:
    latest: dict[str, bytes | None] = {"jpg": None}
    lock = threading.Lock()
    stop = threading.Event()
    failure: list[BaseException] = []

    def on_frame(frame) -> None:
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with lock:
                latest["jpg"] = buf.tobytes()

    def worker() -> None:
        try:
            identify.run(cfg, camera_index, show_window=False, on_frame=on_frame, stop=stop)
        except BaseException as exc:  # shown to the user and in the browser log
            failure.append(exc)
            print(f"Camera loop stopped: {exc}")
        finally:
            stop.set()

    threading.Thread(target=worker, daemon=True).start()
    app = Flask(__name__)

    @app.route("/")
    def index():
        return PAGE

    @app.route("/stream")
    def stream():
        def frames():
            while not stop.is_set():
                with lock:
                    jpg = latest["jpg"]
                if jpg:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
                time.sleep(0.04)
        return Response(frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

    host, port = cfg["web"]["host"], int(cfg["web"]["port"])
    print(f"Open http://{'localhost' if host == '127.0.0.1' else host}:{port}  (Ctrl+C to stop)")
    try:
        app.run(host=host, port=port, threaded=True, use_reloader=False)
    finally:
        stop.set()
