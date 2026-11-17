"""Local SQLite store of registered people. Stores embeddings (numbers) only, never photos.

Owner: Thanigai

Embeddings are kept per backend, because OpenCV (128 numbers) and InsightFace
(512 numbers) can't be compared with each other. Switching backend means re-registering.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_SCHEMA = """
CREATE TABLE IF NOT EXISTS people (
    id      INTEGER PRIMARY KEY,
    name    TEXT NOT NULL UNIQUE COLLATE NOCASE,
    created TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS embeddings (
    id        INTEGER PRIMARY KEY,
    person_id INTEGER NOT NULL REFERENCES people(id) ON DELETE CASCADE,
    backend   TEXT NOT NULL,
    vector    BLOB NOT NULL
);
"""

MAX_NAME = 40


def clean_name(name: str) -> str:
    name = " ".join(name.split())
    if not name:
        raise ValueError("Name can't be empty.")
    if len(name) > MAX_NAME:
        raise ValueError(f"Name must be at most {MAX_NAME} characters.")
    return name


class Database:
    def __init__(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(_SCHEMA)

    def close(self) -> None:
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def has(self, name: str) -> bool:
        row = self.conn.execute("SELECT 1 FROM people WHERE name = ?", (name,)).fetchone()
        return row is not None

    def save_person(self, name: str, backend: str, vectors: np.ndarray) -> None:
        """Add a person, or replace their embeddings for this backend."""
        name = clean_name(name)
        vectors = np.atleast_2d(np.asarray(vectors, dtype=np.float32))
        with self.conn:
            self.conn.execute(
                "INSERT OR IGNORE INTO people (name, created) VALUES (?, ?)",
                (name, datetime.now(timezone.utc).isoformat(timespec="seconds")))
            (pid,) = self.conn.execute("SELECT id FROM people WHERE name = ?", (name,)).fetchone()
            self.conn.execute("DELETE FROM embeddings WHERE person_id = ? AND backend = ?",
                              (pid, backend))
            self.conn.executemany(
                "INSERT INTO embeddings (person_id, backend, vector) VALUES (?, ?, ?)",
                [(pid, backend, v.tobytes()) for v in vectors])

    def people(self) -> list[tuple[str, str, dict[str, int]]]:
        """[(name, created, {backend: sample count})], sorted by name."""
        out: dict[str, tuple[str, dict[str, int]]] = {}
        for name, created in self.conn.execute(
                "SELECT name, created FROM people ORDER BY name COLLATE NOCASE"):
            out[name] = (created, {})
        for name, backend, count in self.conn.execute(
                "SELECT p.name, e.backend, COUNT(*) FROM embeddings e "
                "JOIN people p ON p.id = e.person_id GROUP BY p.name, e.backend"):
            out[name][1][backend] = count
        return [(name, created, counts) for name, (created, counts) in out.items()]

    def delete(self, name: str) -> bool:
        with self.conn:
            cur = self.conn.execute("DELETE FROM people WHERE name = ?", (name,))
        return cur.rowcount > 0

    def reset(self) -> int:
        """Delete everyone. Returns how many people were removed."""
        with self.conn:
            cur = self.conn.execute("DELETE FROM people")
            self.conn.execute("DELETE FROM embeddings")
        return cur.rowcount

    def gallery(self, backend: str) -> dict[str, np.ndarray]:
        """{name: (n, d) array} of every stored embedding for one backend."""
        result: dict[str, list[np.ndarray]] = {}
        for name, blob in self.conn.execute(
                "SELECT p.name, e.vector FROM embeddings e JOIN people p ON p.id = e.person_id "
                "WHERE e.backend = ? ORDER BY e.id", (backend,)):
            result.setdefault(name, []).append(np.frombuffer(blob, dtype=np.float32))
        return {name: np.vstack(vs) for name, vs in result.items()}
