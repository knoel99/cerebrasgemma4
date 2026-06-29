"""Chat with extracted video context to Q&A and propose report enrichments."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from cerebrasgemma4.llm import complete
from cerebrasgemma4.pipeline.context import VideoContext

MAX_TRANSCRIPT_CHARS = 12_000
MAX_DOCUMENT_CHARS = 16_000
MAX_HISTORY_TURNS = 10

CHAT_RESPONSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "video_chat_response",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "reply": {"type": "string"},
                "enrichment": {
                    "anyOf": [
                        {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "markdown": {"type": "string"},
                            },
                            "required": ["title", "markdown"],
                            "additionalProperties": False,
                        },
                        {"type": "null"},
                    ],
                },
            },
            "required": ["reply", "enrichment"],
            "additionalProperties": False,
        },
    },
}

SYSTEM_PROMPT = (
    "You are a video documentation assistant. Answer only from the provided context: "
    "current report, transcript, key-frame analyses, and observation tables when present. "
    "If the user asks to add, expand, or enrich the report, set enrichment with a new "
    "section title and Markdown body (no leading ## — the app adds the heading). "
    "For chart or metric requests, prefer observation tables over sparse frame notes; "
    "do not invent numeric values. "
    "For simple questions, set enrichment to null. "
    "Write in the same language as the report/transcript unless the user asks otherwise."
)


@dataclass
class ChatEnrichment:
    title: str
    markdown: str


@dataclass
class ChatTurnResult:
    reply: str
    enrichment: ChatEnrichment | None
    usage: dict[str, Any]
    time_info: dict[str, Any]


def _format_timestamp(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 20] + "\n… [truncated]"


def _format_observations(ctx: VideoContext) -> str:
    from cerebrasgemma4.pipeline.charts import format_table
    from cerebrasgemma4.pipeline.gemma.series import DataObservation

    if not ctx.observations:
        return "(no observations — ask for charts to rescan frames and transcript)"
    rows = [DataObservation.from_dict(item) for item in ctx.observations]
    return format_table(rows)


def _format_analyses(ctx: VideoContext) -> str:
    if not ctx.analyses:
        return "(no key frame analyses)"
    blocks = []
    for a in ctx.analyses:
        quotes = a.get("quoted_text") or []
        quote_block = "\n".join(f'> "{q}"' for q in quotes) if quotes else ""
        blocks.append(
            f"### [{_format_timestamp(a['timestamp_sec'])}] {a.get('title', '')}\n"
            f"Asset: assets/{a.get('asset_name', '')}\n"
            f"{a.get('body', '')}\n{quote_block}".strip()
        )
    return "\n\n".join(blocks)


def build_context_block(ctx: VideoContext | None, document_md: str) -> str:
    doc = _truncate(document_md, MAX_DOCUMENT_CHARS)
    if ctx is None:
        return f"## Current report\n{doc}\n\n(Transcript and frame analyses unavailable for this job.)"

    transcript = _truncate(ctx.transcript_full_text, MAX_TRANSCRIPT_CHARS)
    return (
        f"## Current report\n{doc}\n\n"
        f"## Source\n{ctx.source_name}\n"
        f"Duration: {_format_timestamp(ctx.duration_sec)}\n"
        f"Transcript source: {ctx.transcript_source}\n\n"
        f"## Full transcript\n{transcript}\n\n"
        f"## Key frame analyses\n{_format_analyses(ctx)}\n\n"
        f"## Observations\n{_format_observations(ctx)}"
    )


def build_chat_messages(
    *,
    ctx: VideoContext | None,
    document_md: str,
    chat_history: list[dict[str, str]],
    user_message: str,
) -> list[dict[str, Any]]:
    context_block = build_context_block(ctx, document_md)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Video context for this conversation:\n\n{context_block}",
        },
        {
            "role": "assistant",
            "content": "Understood. I will answer from this context and propose report sections when asked.",
        },
    ]
    for turn in chat_history[-MAX_HISTORY_TURNS:]:
        role = turn.get("role")
        content = turn.get("content", "")
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_message})
    return messages


def _parse_json(content: str) -> dict:
    content = content.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise


def run_chat_turn(
    *,
    ctx: VideoContext | None,
    document_md: str,
    chat_history: list[dict[str, str]],
    user_message: str,
) -> ChatTurnResult:
    messages = build_chat_messages(
        ctx=ctx,
        document_md=document_md,
        chat_history=chat_history,
        user_message=user_message,
    )
    result = complete(
        messages,
        response_format=CHAT_RESPONSE_SCHEMA,
        temperature=0.5,
        max_completion_tokens=4096,
    )
    data = _parse_json(result.content)
    enrichment_raw = data.get("enrichment")
    enrichment = None
    if enrichment_raw and isinstance(enrichment_raw, dict):
        title = (enrichment_raw.get("title") or "").strip()
        markdown = (enrichment_raw.get("markdown") or "").strip()
        if title and markdown:
            enrichment = ChatEnrichment(title=title, markdown=markdown)
    return ChatTurnResult(
        reply=(data.get("reply") or "").strip(),
        enrichment=enrichment,
        usage=result.usage,
        time_info=result.time_info,
    )