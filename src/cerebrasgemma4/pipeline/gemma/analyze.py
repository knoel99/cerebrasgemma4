"""Analyst stage: deep per-frame description."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from cerebrasgemma4.llm import build_multimodal_message, complete
from cerebrasgemma4.pipeline.gemma.scout import FrameScore


@dataclass
class FrameAnalysis:
    frame_id: str
    timestamp_sec: float
    title: str
    body: str
    quoted_text: list[str]
    asset_name: str


ANALYZE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "frame_analysis",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "body": {"type": "string"},
                "quoted_text": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["title", "body", "quoted_text"],
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


def analyze_frame(
    score: FrameScore,
    path: Path,
    transcript_excerpt: str,
    *,
    asset_name: str,
) -> tuple[FrameAnalysis, dict]:
    prompt = (
        f"Analyze this video frame at {score.timestamp_sec:.1f}s for documentation.\n"
        f"Scout note: {score.brief}\n"
        f"Transcript nearby: {transcript_excerpt or '(no speech)'}\n\n"
        "Extract visible text (OCR), describe UI/diagrams/actions shown. "
        "Write for a technical document reader."
    )
    msg = build_multimodal_message(prompt, [path], detail=True)
    result = complete(
        [msg],
        response_format=ANALYZE_SCHEMA,
        temperature=0.4,
        max_completion_tokens=4096,
    )
    data = _parse_json(result.content)
    analysis = FrameAnalysis(
        frame_id=score.frame_id,
        timestamp_sec=score.timestamp_sec,
        title=data.get("title", f"Frame at {score.timestamp_sec:.0f}s"),
        body=data.get("body", ""),
        quoted_text=data.get("quoted_text", []),
        asset_name=asset_name,
    )
    metrics = {"usage": result.usage, "time_info": result.time_info}
    return analysis, metrics