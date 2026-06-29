"""Writer stage: stream final Markdown document."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from cerebrasgemma4.llm import stream, stream_collect
from cerebrasgemma4.pipeline.gemma.analyze import FrameAnalysis
from cerebrasgemma4.pipeline.transcript import TranscriptResult


@dataclass
class ComposeInput:
    source_name: str
    duration_sec: float
    transcript: TranscriptResult
    analyses: list[FrameAnalysis]


def _format_timestamp(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def build_compose_prompt(data: ComposeInput) -> str:
    frame_sections = []
    for a in data.analyses:
        quotes = "\n".join(f'> "{q}"' for q in a.quoted_text) if a.quoted_text else ""
        frame_sections.append(
            f"### [{_format_timestamp(a.timestamp_sec)}] {a.title}\n"
            f"Image: assets/{a.asset_name}\n"
            f"{a.body}\n{quotes}"
        )
    frames_block = "\n\n".join(frame_sections) or "(no key frames selected)"
    return (
        "Write a clear Markdown document from this video analysis.\n\n"
        f"Source: {data.source_name}\n"
        f"Duration: {_format_timestamp(data.duration_sec)}\n"
        f"Transcript source: {data.transcript.source}\n\n"
        "## Full transcript (reference)\n"
        f"{data.transcript.full_text[:8000]}\n\n"
        "## Key frame analyses\n"
        f"{frames_block}\n\n"
        "Output Markdown with:\n"
        "- # Title\n"
        "- ## Executive summary (3-5 bullets)\n"
        "- ## Detailed content (sections per key moment with ![caption](assets/...) images)\n"
        "- ## Key takeaways\n"
        "Use timestamps in headings. Write in the same language as the transcript."
    )


def compose_stream(data: ComposeInput) -> Iterator[str]:
    prompt = build_compose_prompt(data)
    messages = [{"role": "user", "content": prompt}]
    yield from stream(messages, temperature=0.7, max_completion_tokens=8192)


def compose_complete(data: ComposeInput) -> tuple[str, dict]:
    prompt = build_compose_prompt(data)
    messages = [{"role": "user", "content": prompt}]
    result = stream_collect(messages, temperature=0.7, max_completion_tokens=8192)
    return result.content, {"usage": result.usage, "time_info": result.time_info}