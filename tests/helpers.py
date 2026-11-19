"""Small helpers shared by the test files."""
import numpy as np

from urec.embedder import Face


def unit(*v):
    """A unit-length vector, like a real embedding."""
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def face(box):
    """A fake detected face with the given box."""
    return Face(box=box, score=0.9, landmarks=np.zeros((5, 2), np.float32), embedding=unit(1, 0))
