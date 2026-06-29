"""Cerebras Gemma 4 LLM client."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from dotenv import load_dotenv
from cerebras.cloud.sdk import Cerebras

from cerebrasgemma4.images import build_image_parts

load_dotenv()

MODEL = "gemma-4-31b"
_client = Cerebras(api_key=os.environ["CEREBRAS_API_KEY"])

_DEFAULTS = {
    "max_completion_tokens": 32_768,
    "temperature": 1.0,
    "top_p": 0.95,
}


@dataclass
class CompletionResult:
    content: str
    usage: dict[str, Any] = field(default_factory=dict)
    time_info: dict[str, Any] = field(default_factory=dict)


def stream(messages: list[dict], **kwargs) -> Iterator[str]:
    params = {**_DEFAULTS, **kwargs}
    for chunk in _client.chat.completions.create(
        messages=messages,
        model=MODEL,
        stream=True,
        **params,
    ):
        yield chunk.choices[0].delta.content or ""


def stream_collect(messages: list[dict], **kwargs) -> CompletionResult:
    """Stream completion and aggregate text + final usage/time_info."""
    params = {**_DEFAULTS, **kwargs}
    parts: list[str] = []
    usage: dict[str, Any] = {}
    time_info: dict[str, Any] = {}
    for chunk in _client.chat.completions.create(
        messages=messages,
        model=MODEL,
        stream=True,
        **params,
    ):
        delta = chunk.choices[0].delta.content or ""
        if delta:
            parts.append(delta)
        if getattr(chunk, "usage", None):
            usage = (
                chunk.usage.model_dump()
                if hasattr(chunk.usage, "model_dump")
                else dict(chunk.usage)
            )
        if getattr(chunk, "time_info", None):
            time_info = (
                chunk.time_info.model_dump()
                if hasattr(chunk.time_info, "model_dump")
                else dict(chunk.time_info)
            )
    return CompletionResult(content="".join(parts), usage=usage, time_info=time_info)


def complete(messages: list[dict], **kwargs) -> CompletionResult:
    params = {**_DEFAULTS, **kwargs}
    response = _client.chat.completions.create(
        messages=messages,
        model=MODEL,
        stream=False,
        **params,
    )
    usage = {}
    time_info = {}
    if response.usage:
        usage = response.usage.model_dump() if hasattr(response.usage, "model_dump") else dict(response.usage)
    if hasattr(response, "time_info") and response.time_info:
        time_info = (
            response.time_info.model_dump()
            if hasattr(response.time_info, "model_dump")
            else dict(response.time_info)
        )
    return CompletionResult(
        content=response.choices[0].message.content or "",
        usage=usage,
        time_info=time_info,
    )


def build_multimodal_message(
    text: str,
    image_paths: list[Path | str],
    *,
    detail: bool = False,
) -> dict:
    paths = [Path(p) for p in image_paths]
    parts: list[dict] = build_image_parts(paths, detail=detail)
    parts.append({"type": "text", "text": text})
    return {"role": "user", "content": parts}