"""SQLite persistence for analysis jobs.

The API is stateless apart from this store: uploads, progress and completed
results survive a server restart.  JSON is stored as a TEXT column so the
schema stays simple and report links remain reproducible.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .schemas import AnalysisResult, JobResponse


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class JobStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path), timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def init_schema(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    stage TEXT NOT NULL DEFAULT '',
                    progress REAL NOT NULL DEFAULT 0,
                    message TEXT NOT NULL DEFAULT '',
                    error TEXT,
                    source_name TEXT NOT NULL DEFAULT '',
                    source_path TEXT NOT NULL DEFAULT '',
                    result_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def create(self, job_id: str, source_name: str, source_path: str = "") -> JobResponse:
        now = _now()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO jobs
                (job_id, status, stage, progress, message, error, source_name, source_path, result_json, created_at, updated_at)
                VALUES (?, 'queued', 'queued', 0, 'Queued', NULL, ?, ?, NULL, ?, ?)
                """,
                (job_id, source_name, source_path, now, now),
            )
            conn.commit()
        return self.get(job_id)  # type: ignore[return-value]

    def update(
        self,
        job_id: str,
        *,
        status: str | None = None,
        stage: str | None = None,
        progress: float | None = None,
        message: str | None = None,
        error: str | None = None,
    ) -> None:
        fields: dict[str, Any] = {}
        if status is not None:
            fields["status"] = status
        if stage is not None:
            fields["stage"] = stage
        if progress is not None:
            fields["progress"] = max(0.0, min(1.0, float(progress)))
        if message is not None:
            fields["message"] = message
        if error is not None:
            fields["error"] = error
        if not fields:
            return
        fields["updated_at"] = _now()
        assignments = ", ".join(f"{key} = ?" for key in fields)
        with self._lock, self._connect() as conn:
            conn.execute(
                f"UPDATE jobs SET {assignments} WHERE job_id = ?",
                (*fields.values(), job_id),
            )
            conn.commit()

    def complete(self, job_id: str, result: AnalysisResult) -> None:
        now = _now()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET status='completed', stage='done', progress=1.0, message='Completed',
                    error=NULL, result_json=?, updated_at=?
                WHERE job_id=?
                """,
                (result.model_dump_json(), now, job_id),
            )
            conn.commit()

    def fail(self, job_id: str, error: str) -> None:
        self.update(job_id, status="failed", stage="failed", progress=0.0, message="Failed", error=error)

    def get(self, job_id: str) -> JobResponse | None:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return self._row_to_response(row) if row else None

    def list(self, limit: int = 20, include_result: bool = False) -> list[JobResponse]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (int(limit),)
            ).fetchall()
        return [self._row_to_response(row, include_result=include_result) for row in rows]

    def _row_to_response(self, row: sqlite3.Row, include_result: bool = True) -> JobResponse:
        result = None
        if include_result and row["result_json"]:
            try:
                result = AnalysisResult.model_validate_json(row["result_json"])
            except Exception:
                result = None
        return JobResponse(
            job_id=row["job_id"],
            status=row["status"],
            stage=row["stage"],
            progress=float(row["progress"]),
            message=row["message"],
            error=row["error"],
            source_name=row["source_name"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            result=result,
        )

    def delete(self, job_id: str) -> bool:
        with self._lock, self._connect() as conn:
            cur = conn.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))
            conn.commit()
            return cur.rowcount > 0

    def stats(self) -> dict[str, Any]:
        with self._lock, self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
            by_status = {
                row["status"]: row["n"]
                for row in conn.execute("SELECT status, COUNT(*) AS n FROM jobs GROUP BY status")
            }
        return {"total": int(total), "by_status": by_status, "database": str(self.path)}


def result_to_json(result: AnalysisResult) -> str:
    return json.dumps(json.loads(result.model_dump_json()), indent=2, ensure_ascii=False)
