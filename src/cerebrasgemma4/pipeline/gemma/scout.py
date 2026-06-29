"""Scout stage: score frame chunks for document relevance."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from cerebrasgemma4.llm import build_multimodal_message, complete
from cerebrasgemma4.pipeline.frames import FrameChunk


@dataclass
class FrameScore:
    frame_id: str
    timestamp_sec: float
    relevance: float
    brief: str
    has_readable_text: bool


SCOUT_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "frame_scores",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "frames": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "frame_id": {"type": "string"},
                            "timestamp_sec": {"type": "number"},
                            "relevance": {"type": "number"},
                            "has_readable_text": {"type": "boolean"},
                            "brief": {"type": "string"},
                        },
                        "required": [
                            "frame_id",
                            "timestamp_sec",
                            "relevance",
                            "has_readable_text",
                            "brief",
                        ],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["frames"],
            "additionalProperties": False,
        },
    },
}


def _parse_json(content: str) -> dict:
    content = content.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise


def scout_chunk(
    chunk: FrameChunk,
    transcript_excerpt: str,
) -> tuple[list[FrameScore], dict]:
    frame_list = "\n".join(
        f"- {f.frame_id} at {f.timestamp_sec:.1f}s" for f in chunk.frames
    )
    prompt = (
        "You are a video documentation scout. These images are consecutive video "
        f"frames from {chunk.start_sec:.1f}s to {chunk.end_sec:.1f}s (1 fps).\n\n"
        f"Transcript for this window:\n{transcript_excerpt or '(no speech)'}\n\n"
        f"Frames:\n{frame_list}\n\n"
        "Score each frame for usefulness in a written document (0-1). "
        "Flag frames with readable on-screen text."
    )
    msg = build_multimodal_message(
        prompt,
        [f.path for f in chunk.frames],
        detail=False,
    )
    result = complete(
        [msg],
        response_format=SCOUT_SCHEMA,
        temperature=0.3,
        max_completion_tokens=2048,
    )
    data = _parse_json(result.content)
    scores = [
        FrameScore(
            frame_id=item["frame_id"],
            timestamp_sec=float(item["timestamp_sec"]),
            relevance=float(item["relevance"]),
            brief=item["brief"],
            has_readable_text=bool(item["has_readable_text"]),
        )
        for item in data.get("frames", [])
    ]
    metrics = {"usage": result.usage, "time_info": result.time_info}
    return scores, metrics


def select_top_frames(
    all_scores: list[FrameScore],
    frame_paths: dict[str, Path],
    *,
    max_frames: int,
) -> list[tuple[FrameScore, Path]]:
    ranked = sorted(all_scores, key=lambda s: s.relevance, reverse=True)
    selected: list[tuple[FrameScore, Path]] = []
    seen_ts: set[int] = set()
    for score in ranked:
        bucket = int(score.timestamp_sec)
        if bucket in seen_ts:
            continue
        path = frame_paths.get(score.frame_id)
        if path is None:
            continue
        selected.append((score, path))
        seen_ts.add(bucket)
        if len(selected) >= max_frames:
            break
    selected.sort(key=lambda x: x[0].timestamp_sec)
    return selected