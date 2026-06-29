"""Filesystem-backed job store."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


class JobStatus(str, Enum):
    PENDING = "pending"
    EXTRACTING = "extracting"
    TRANSCRIBING = "transcribing"
    ANALYZING = "analyzing"
    COMPOSING = "composing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class JobRecord:
    job_id: str
    status: JobStatus
    progress: float = 0.0
    message: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    document_path: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    source_name: str | None = None
    source_type: str | None = None
    title: str | None = None
    youtube_video_id: str | None = None
    thumbnail_asset: str | None = None
    preview_asset: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> JobRecord:
        data = dict(data)
        data["status"] = JobStatus(data["status"])
        known = {f.name for f in fields(cls)}
        return cls(**{k: data[k] for k in known if k in data})


def default_jobs_root() -> Path:
    return Path("/tmp/vid2doc")


class JobStore:
    def __init__(self, root: Path | None = None):
        self.root = root or default_jobs_root()
        self.root.mkdir(parents=True, exist_ok=True)

    def create(self) -> JobRecord:
        job_id = str(uuid.uuid4())
        job_dir = self.root / job_id
        job_dir.mkdir(parents=True)
        (job_dir / "assets").mkdir(exist_ok=True)
        record = JobRecord(job_id=job_id, status=JobStatus.PENDING)
        self.save(record)
        return record

    def path(self, job_id: str) -> Path:
        return self.root / job_id

    def meta_path(self, job_id: str) -> Path:
        return self.path(job_id) / "job.json"

    def metrics_path(self, job_id: str) -> Path:
        return self.path(job_id) / "metrics.json"

    def save(self, record: JobRecord) -> None:
        self.meta_path(record.job_id).write_text(
            json.dumps(record.to_dict(), indent=2), encoding="utf-8"
        )

    def load(self, job_id: str) -> JobRecord:
        return JobRecord.from_dict(
            json.loads(self.meta_path(job_id).read_text(encoding="utf-8"))
        )

    def list_jobs(self, *, limit: int = 50) -> list[JobRecord]:
        records: list[JobRecord] = []
        for job_dir in self.root.iterdir():
            if not job_dir.is_dir():
                continue
            meta = job_dir / "job.json"
            if not meta.exists():
                continue
            try:
                records.append(self.load(job_dir.name))
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
        records.sort(key=lambda r: r.created_at, reverse=True)
        return records[:limit]

    def delete(self, job_id: str) -> None:
        import shutil

        path = self.path(job_id)
        if not path.exists():
            raise FileNotFoundError(job_id)
        shutil.rmtree(path)

    def update(
        self,
        job_id: str,
        *,
        status: JobStatus | None = None,
        progress: float | None = None,
        message: str | None = None,
        document_path: str | None = None,
        metrics: dict[str, Any] | None = None,
        error: str | None = None,
        source_name: str | None = None,
        source_type: str | None = None,
        title: str | None = None,
        youtube_video_id: str | None = None,
        thumbnail_asset: str | None = None,
        preview_asset: str | None = None,
    ) -> JobRecord:
        record = self.load(job_id)
        if status is not None:
            record.status = status
        if progress is not None:
            record.progress = progress
        if message is not None:
            record.message = message
        if document_path is not None:
            record.document_path = document_path
        if metrics is not None:
            record.metrics.update(metrics)
        if error is not None:
            record.error = error
        if source_name is not None:
            record.source_name = source_name
        if source_type is not None:
            record.source_type = source_type
        if title is not None:
            record.title = title
        if youtube_video_id is not None:
            record.youtube_video_id = youtube_video_id
        if thumbnail_asset is not None:
            record.thumbnail_asset = thumbnail_asset
        if preview_asset is not None:
            record.preview_asset = preview_asset
        self.save(record)
        return record