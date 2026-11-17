"""Picks the fastest ONNX Runtime device available on this machine.

Owner: Joenish
"""
from __future__ import annotations

_PREFERRED = [
    "CUDAExecutionProvider",    # NVIDIA (onnxruntime-gpu)
    "DmlExecutionProvider",     # any DirectX 12 GPU on Windows (onnxruntime-directml)
    "CoreMLExecutionProvider",  # Apple
    "CPUExecutionProvider",
]


def providers(device: str = "auto") -> list[str]:
    import onnxruntime as ort

    if device == "cpu":
        return ["CPUExecutionProvider"]
    available = ort.get_available_providers()
    chosen = [p for p in _PREFERRED if p in available]
    return chosen or ["CPUExecutionProvider"]
