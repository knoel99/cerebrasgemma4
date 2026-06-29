"""Writer stage: stream final Markdown document."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from cerebrasgemma4.llm import stream, stream_collect
from cerebrasgemma4.pipeline.gemma.analyze import FrameAnalysis
from cerebrasgemma4.pipeline.transcript import TranscriptResult


COMPOSE_DEFAULT_INSTRUCTIONS = (
    "<role>\n"
    "You are a magazine photo editor writing the final layout from a video analysis. "
    "Each key frame is a figure in a photo-essay: pair it with a cutline and prose "
    "that ties the image to the narrative — same editorial taste as the scout contact "
    "sheet pass.\n"
    "</role>\n\n"
    "<output_format>\n"
    "Markdown with:\n"
    "- # Title — specific and evocative, not generic\n"
    "- ## Executive summary — 3-5 bullets; lead with the story\n"
    "- ## Detailed content — one section per key moment:\n"
    "  - ## [MM:SS] Section title\n"
    "  - ![Editorial cutline — what the reader sees and why it matters](assets/...)\n"
    "  - 1-3 short paragraphs weaving transcript claims with visual evidence\n"
    "- ## Key takeaways — 3-5 concrete bullets\n"
    "</output_format>\n\n"
    "<editorial_rules>\n"
    "- Captions read like photo cutlines, not filenames or placeholders\n"
    "- Use only frames listed in key frame analyses; do not invent visuals\n"
    "- Quote on-screen text verbatim when analyses include it\n"
    "- Prefer decisive, sharp moments; skip redundant talking-head-only beats\n"
    "- Timestamps in every section heading\n"
    "- Write in the same language as the transcript\n"
    "</editorial_rules>"
)


@dataclass
class ComposeInput:
    source_name: str
    duration_sec: float
    transcript: TranscriptResult
    analyses: list[FrameAnalysis]
    custom_prompt: str | None = None


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
    parts = [COMPOSE_DEFAULT_INSTRUCTIONS]
    custom = (data.custom_prompt or "").strip()
    if custom:
        parts.append(f"\n\n## Additional instructions from the user\n{custom}")
    parts.extend(
        [
            f"\n\nSource: {data.source_name}",
            f"Duration: {_format_timestamp(data.duration_sec)}",
            f"Transcript source: {data.transcript.source}",
            "\n## Full transcript (reference)",
            data.transcript.full_text[:8000],
            "\n## Key frame analyses",
            frames_block,
        ]
    )
    return "\n".join(parts)


def compose_stream(data: ComposeInput) -> Iterator[str]:
    prompt = build_compose_prompt(data)
    messages = [{"role": "user", "content": prompt}]
    yield from stream(messages, temperature=0.7, max_completion_tokens=8192)


def compose_complete(data: ComposeInput) -> tuple[str, dict]:
    prompt = build_compose_prompt(data)
    messages = [{"role": "user", "content": prompt}]
    result = stream_collect(messages, temperature=0.7, max_completion_tokens=8192)
    return result.content, {"usage": result.usage, "time_info": result.time_info}