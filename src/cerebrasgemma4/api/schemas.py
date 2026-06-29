from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class DocumentStyle(str, Enum):
    AUTO = "auto"
    TUTORIAL = "tutorial"
    MEETING = "meeting"


class ConvertOptionsSchema(BaseModel):
    max_frames: int = Field(default=8, ge=1, le=30)
    max_duration_sec: float | None = Field(default=120.0, ge=10, le=600)
    language: str = "auto"
    document_style: DocumentStyle = DocumentStyle.AUTO
    include_transcript: bool = True


class ConvertResponse(BaseModel):
    job_id: str
    status: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: float
    message: str
    created_at: str | None = None
    document_path: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    source_name: str | None = None
    source_type: str | None = None
    title: str | None = None
    youtube_video_id: str | None = None
    thumbnail_asset: str | None = None
    preview_asset: str | None = None


class JobSummary(BaseModel):
    job_id: str
    status: str
    created_at: str
    source_type: str | None = None
    source_name: str | None = None
    title: str | None = None
    youtube_video_id: str | None = None
    thumbnail_url: str | None = None
    preview_url: str | None = None
    elapsed_sec: float | None = None
    cerebras_calls: int | None = None


class JobHistoryResponse(BaseModel):
    jobs: list[JobSummary]
    total: int


class DocumentResponse(BaseModel):
    job_id: str
    markdown: str
    assets: list[str] = Field(default_factory=list)


class ProbeResponse(BaseModel):
    duration_sec: float
    max_duration_sec: float
    max_frames: int
    width: int = 0
    height: int = 0
    fps: float = 0.0
    estimated_scout_calls: int
    estimated_analyze_calls: int
    estimated_total_api_calls: int
    within_hackathon_rpm: bool
    hackathon_capped: bool = False
    estimated_pipeline_minutes: float = 0.0
    hackathon_rpm: int
    hackathon_tpm: int
    note: str = ""