"""Per-job chat history persistence."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CHAT_FILENAME = "chat.json"


def chat_path(job_dir: Path) -> Path:
    return job_dir / CHAT_FILENAME


def load_chat_history(job_dir: Path) -> list[dict[str, Any]]:
    path = chat_path(job_dir)
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data.get("messages") or [])


def save_chat_history(job_dir: Path, messages: list[dict[str, Any]]) -> None:
    path = chat_path(job_dir)
    path.write_text(
        json.dumps({"messages": messages}, indent=2),
        encoding="utf-8",
    )


def append_chat_turn(
    job_dir: Path,
    *,
    user_message: str,
    assistant_reply: str,
    enrichment: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    messages = load_chat_history(job_dir)
    now = datetime.now(timezone.utc).isoformat()
    messages.append({"role": "user", "content": user_message, "created_at": now})
    assistant: dict[str, Any] = {
        "role": "assistant",
        "content": assistant_reply,
        "created_at": now,
    }
    if enrichment:
        assistant["enrichment"] = enrichment
    messages.append(assistant)
    save_chat_history(job_dir, messages)
    return messages


def append_section_to_document(document_path: Path, title: str, markdown: str) -> str:
    existing = document_path.read_text(encoding="utf-8").rstrip()
    section = f"\n\n## {title.strip()}\n\n{markdown.strip()}\n"
    updated = existing + section
    document_path.write_text(updated, encoding="utf-8")
    return updated