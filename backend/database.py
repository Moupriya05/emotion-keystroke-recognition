"""
database.py
------------
Tiny SQLite wrapper for storing typing sessions (raw keystroke events +
emotion label). No ORM - just stdlib sqlite3, so the whole project has
zero infra dependencies beyond Python itself.

Schema:
    sessions(id, user_id, emotion_label, text_type, prompt_text,
             typed_text, created_at)
    keystrokes(id, session_id, code, event_type, t_ms)
"""

from __future__ import annotations
import json
import sqlite3
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "emotion_keystrokes.db"


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            emotion_label TEXT,
            text_type TEXT,
            prompt_text TEXT,
            typed_text TEXT,
            created_at REAL NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS keystrokes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            code TEXT NOT NULL,
            event_type TEXT NOT NULL,
            t_ms REAL NOT NULL,
            FOREIGN KEY(session_id) REFERENCES sessions(id)
        )
    """)
    conn.commit()
    conn.close()


def insert_session(
    user_id: str,
    events: List[Dict[str, Any]],
    emotion_label: Optional[str] = None,
    text_type: str = "free",
    prompt_text: str = "",
    typed_text: str = "",
) -> int:
    """Stores one typing session and its raw keystroke log. Returns session id."""
    conn = get_connection()
    cur = conn.execute(
        """INSERT INTO sessions (user_id, emotion_label, text_type, prompt_text,
                                  typed_text, created_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (user_id, emotion_label, text_type, prompt_text, typed_text, time.time()),
    )
    session_id = cur.lastrowid
    conn.executemany(
        "INSERT INTO keystrokes (session_id, code, event_type, t_ms) VALUES (?, ?, ?, ?)",
        [(session_id, e.get("code") or e.get("key"), e["type"], float(e["t"])) for e in events],
    )
    conn.commit()
    conn.close()
    return session_id


def get_all_sessions() -> List[Dict[str, Any]]:
    """
    Returns every stored session as a dict with its event log attached,
    ready to be fed straight into ml.features.extract_features().
    """
    conn = get_connection()
    sessions = conn.execute("SELECT * FROM sessions").fetchall()
    out = []
    for s in sessions:
        events = conn.execute(
            "SELECT code, event_type as type, t_ms as t FROM keystrokes WHERE session_id = ? ORDER BY t_ms",
            (s["id"],),
        ).fetchall()
        out.append({
            "session_id": s["id"],
            "user_id": s["user_id"],
            "emotion_label": s["emotion_label"],
            "text_type": s["text_type"],
            "prompt_text": s["prompt_text"],
            "typed_text": s["typed_text"],
            "events": [dict(e) for e in events],
        })
    conn.close()
    return out



if __name__ == "__main__":
    init_db()
    print(f"Database initialised at {DB_PATH}")
