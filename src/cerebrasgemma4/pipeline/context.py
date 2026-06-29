"""Persisted video context for post-generation chat."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from cerebrasgemma4.pipeline.gemma.analyze import FrameAnalysis
from cerebrasgemma4.pipeline.transcript import TranscriptResult

CONTEXT_FILENAME = "context.json"


@dataclass
class VideoContext:
    source_name: str
    duration_sec: float
    transcript_source: str
    transcript_full_text: str
    transcript_segments: list[dict[str, Any]] = field(default_factory=list)
    analyses: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VideoContext:
        return cls(
            source_name=data["source_name"],
            duration_sec=float(data["duration_sec"]),
            transcript_source=data.get("transcript_source", "unknown"),
            transcript_full_text=data.get("transcript_full_text", ""),
            transcript_segments=list(data.get("transcript_segments") or []),
            analyses=list(data.get("analyses") or []),
        )

    @classmethod
    def from_pipeline(
        cls,
        *,
        source_name: str,
        duration_sec: float,
        transcript: TranscriptResult,
        analyses: list[FrameAnalysis],
    ) -> VideoContext:
        return cls(
            source_name=source_name,
            duration_sec=duration_sec,
            transcript_source=transcript.source,
            transcript_full_text=transcript.full_text,
            transcript_segments=[
                {
                    "start_sec": s.start_sec,
                    "end_sec": s.end_sec,
                    "text": s.text,
                }
                for s in transcript.segments
            ],
            analyses=[
                {
                    "frame_id": a.frame_id,
                    "timestamp_sec": a.timestamp_sec,
                    "title": a.title,
                    "body": a.body,
                    "quoted_text": a.quoted_text,
                    "asset_name": a.asset_name,
                }
                for a in analyses
            ],
        )


def context_path(job_dir: Path) -> Path:
    return job_dir / CONTEXT_FILENAME


def save_context(job_dir: Path, ctx: VideoContext) -> Path:
    path = context_path(job_dir)
    path.write_text(json.dumps(ctx.to_dict(), indent=2), encoding="utf-8")
    return path


def load_context(job_dir: Path) -> VideoContext | None:
    path = context_path(job_dir)
    if not path.exists():
        return None
    return VideoContext.from_dict(json.loads(path.read_text(encoding="utf-8")))