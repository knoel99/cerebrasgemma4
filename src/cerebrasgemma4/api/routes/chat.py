from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from cerebrasgemma4.api.schemas import (
    ApplyEnrichmentRequest,
    ApplyEnrichmentResponse,
    ChatHistoryResponse,
    ChatMessageSchema,
    ChatRequest,
)
from cerebrasgemma4.pipeline.chat_store import (
    append_chat_turn,
    append_section_to_document,
    load_chat_history,
)
from cerebrasgemma4.pipeline.context import load_context
from cerebrasgemma4.pipeline.gemma.chat import run_chat_turn
from cerebrasgemma4.pipeline.jobs import JobStatus, JobStore

router = APIRouter(prefix="/api/jobs", tags=["chat"])


def _get_store() -> JobStore:
    return JobStore()


def _require_completed_job(store: JobStore, job_id: str):
    try:
        record = store.load(job_id)
    except FileNotFoundError:
        raise HTTPException(404, "Job not found") from None
    if record.status != JobStatus.COMPLETED or not record.document_path:
        raise HTTPException(409, f"Job not ready: {record.status.value}")
    doc_path = Path(record.document_path)
    job_dir = store.path(job_id)
    return record, doc_path, job_dir


def _history_to_schema(messages: list) -> list[ChatMessageSchema]:
    out: list[ChatMessageSchema] = []
    for msg in messages:
        enrichment = msg.get("enrichment")
        out.append(
            ChatMessageSchema(
                role=msg["role"],
                content=msg.get("content", ""),
                created_at=msg.get("created_at"),
                enrichment=enrichment,
            )
        )
    return out


@router.get("/{job_id}/chat", response_model=ChatHistoryResponse)
def get_chat_history(job_id: str):
    store = _get_store()
    _require_completed_job(store, job_id)
    job_dir = store.path(job_id)
    messages = load_chat_history(job_dir)
    return ChatHistoryResponse(
        job_id=job_id,
        messages=_history_to_schema(messages),
        has_context=load_context(job_dir) is not None,
    )


@router.post("/{job_id}/chat")
def post_chat(job_id: str, body: ChatRequest):
    store = _get_store()
    record, doc_path, job_dir = _require_completed_job(store, job_id)
    document_md = doc_path.read_text(encoding="utf-8")
    ctx = load_context(job_dir)
    history = load_chat_history(job_dir)

    try:
        turn = run_chat_turn(
            ctx=ctx,
            document_md=document_md,
            chat_history=history,
            user_message=body.message.strip(),
        )
    except Exception as exc:
        raise HTTPException(502, f"Chat failed: {exc}") from exc

    enrichment_dict = None
    if turn.enrichment:
        enrichment_dict = {
            "title": turn.enrichment.title,
            "markdown": turn.enrichment.markdown,
        }

    append_chat_turn(
        job_dir,
        user_message=body.message.strip(),
        assistant_reply=turn.reply,
        enrichment=enrichment_dict,
    )

    reply = turn.reply

    def stream_events():
        chunk_size = 24
        for i in range(0, len(reply), chunk_size):
            yield f"data: {json.dumps({'token': reply[i:i + chunk_size]})}\n\n"
        done_payload: dict = {"done": True}
        if enrichment_dict:
            done_payload["enrichment"] = enrichment_dict
        if turn.time_info:
            done_payload["time_info"] = turn.time_info
        if turn.usage:
            done_payload["usage"] = turn.usage
        yield f"data: {json.dumps(done_payload)}\n\n"

    return StreamingResponse(stream_events(), media_type="text/event-stream")


@router.post("/{job_id}/chat/apply", response_model=ApplyEnrichmentResponse)
def apply_enrichment(job_id: str, body: ApplyEnrichmentRequest):
    store = _get_store()
    _record, doc_path, _job_dir = _require_completed_job(store, job_id)
    updated = append_section_to_document(doc_path, body.title, body.markdown)
    revision = updated.count("## ")
    return ApplyEnrichmentResponse(
        job_id=job_id,
        markdown=updated,
        revision=revision,
    )