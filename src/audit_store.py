from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from .config import PATHS

DEFAULT_DB = PATHS.artifacts / "case_audit.sqlite3"
ALLOWED_STATUSES = {"NEW", "REVIEWING", "ESCALATED", "CLOSED_NO_FRAUD", "CLOSED_CONFIRMED"}


def _connect(path: Path = DEFAULT_DB) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.executescript("""
        CREATE TABLE IF NOT EXISTS cases (
            case_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            analyst_notes TEXT NOT NULL DEFAULT '',
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS audit_events (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            case_id TEXT NOT NULL,
            event_time TEXT NOT NULL,
            event_type TEXT NOT NULL,
            actor TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            dataset_sha256 TEXT,
            model_version TEXT
        );
    """)
    return connection


def get_case(case_id: str, path: Path = DEFAULT_DB) -> dict:
    with _connect(path) as connection:
        row = connection.execute("SELECT * FROM cases WHERE case_id = ?", (case_id,)).fetchone()
    return dict(row) if row else {"case_id": case_id, "status": "NEW", "analyst_notes": ""}


def save_disposition(case_id: str, status: str, notes: str, actor: str,
                     dataset_sha256: str, model_version: str,
                     path: Path = DEFAULT_DB) -> None:
    if status not in ALLOWED_STATUSES:
        raise ValueError(f"Unsupported case status: {status}")
    now = datetime.now(timezone.utc).isoformat()
    payload = {"status": status, "notes": notes}
    with _connect(path) as connection:
        connection.execute("""
            INSERT INTO cases(case_id, status, analyst_notes, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(case_id) DO UPDATE SET
              status=excluded.status, analyst_notes=excluded.analyst_notes,
              updated_at=excluded.updated_at
        """, (case_id, status, notes, now))
        connection.execute("""
            INSERT INTO audit_events(case_id, event_time, event_type, actor, payload_json,
                                     dataset_sha256, model_version)
            VALUES (?, ?, 'DISPOSITION_UPDATED', ?, ?, ?, ?)
        """, (case_id, now, actor, json.dumps(payload), dataset_sha256, model_version))


def list_events(case_id: str, path: Path = DEFAULT_DB) -> list[dict]:
    with _connect(path) as connection:
        rows = connection.execute(
            "SELECT * FROM audit_events WHERE case_id = ? ORDER BY event_id DESC", (case_id,)
        ).fetchall()
    output = []
    for row in rows:
        item = dict(row)
        item["payload"] = json.loads(item.pop("payload_json"))
        output.append(item)
    return output

