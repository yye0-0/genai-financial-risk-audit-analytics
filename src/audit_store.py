from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from .config import PATHS

DEFAULT_DB = PATHS.artifacts / "case_audit.sqlite3"
ALLOWED_STATUSES = {"NEW", "REVIEWING", "ESCALATED", "CLOSED_NO_FRAUD", "CLOSED_CONFIRMED"}
FEEDBACK_CATEGORIES = {"none", "factual_error", "missing_evidence", "omission", "unclear"}


class ConcurrentReviewError(ValueError):
    """Another reviewer saved this case after it was opened."""


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
    columns = {row["name"] for row in connection.execute("PRAGMA table_info(cases)")}
    if "reviewed_summary" not in columns:
        connection.execute("ALTER TABLE cases ADD COLUMN reviewed_summary TEXT NOT NULL DEFAULT ''")
    return connection


def get_case(case_id: str, path: Path = DEFAULT_DB) -> dict:
    with _connect(path) as connection:
        row = connection.execute("SELECT * FROM cases WHERE case_id = ?", (case_id,)).fetchone()
    return dict(row) if row else {"case_id": case_id, "status": "NEW", "analyst_notes": "",
                                 "reviewed_summary": "", "updated_at": ""}


def save_disposition(case_id: str, status: str, notes: str, actor: str,
                     dataset_sha256: str, model_version: str,
                     path: Path = DEFAULT_DB, *, reviewed_summary: str = "",
                     generation: dict | None = None, feedback: str = "none",
                     expected_updated_at: str | None = None) -> None:
    if status not in ALLOWED_STATUSES:
        raise ValueError(f"Unsupported case status: {status}")
    if feedback not in FEEDBACK_CATEGORIES:
        raise ValueError("Unsupported feedback category")
    if not actor.strip():
        raise ValueError("A reviewer name is required")
    if status in {"ESCALATED", "CLOSED_NO_FRAUD", "CLOSED_CONFIRMED"} and not notes.strip():
        raise ValueError("Record a rationale before escalating or closing a case")
    now = datetime.now(timezone.utc).isoformat()
    with _connect(path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        previous = connection.execute(
            "SELECT status, updated_at FROM cases WHERE case_id = ?", (case_id,)).fetchone()
        if expected_updated_at is not None and expected_updated_at != (
                previous["updated_at"] if previous else ""):
            raise ConcurrentReviewError("This case changed. Reload it before saving your review.")
        payload = {"previous_status": previous["status"] if previous else "NEW",
                   "status": status, "notes": notes, "reviewed_summary": reviewed_summary,
                   "generation": generation, "feedback": feedback}
        connection.execute("""
            INSERT INTO cases(case_id, status, analyst_notes, updated_at, reviewed_summary)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(case_id) DO UPDATE SET
              status=excluded.status, analyst_notes=excluded.analyst_notes,
              updated_at=excluded.updated_at, reviewed_summary=excluded.reviewed_summary
        """, (case_id, status, notes, now, reviewed_summary))
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

