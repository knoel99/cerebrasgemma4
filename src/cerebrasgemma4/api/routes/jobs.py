from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response, StreamingResponse

from cerebrasgemma4.api.schemas import (
    DocumentResponse,
    JobHistoryResponse,
    JobStatusResponse,
    JobSummary,
)
from cerebrasgemma4.export import export_filename, render_html_document, render_pdf_bytes
from cerebrasgemma4.pipeline.jobs import JobRecord, JobStatus, JobStore
from cerebrasgemma4.pipeline.perf import apply_wall_elapsed

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


def _get_store() -> JobStore:
    return JobStore()


def _require_completed_job(store: JobStore, job_id: str) -> tuple[JobRecord, Path, Path]:
    try:
        record = store.load(job_id)
    except FileNotFoundError:
        raise HTTPException(404, "Job not found") from None
    if record.status != JobStatus.COMPLETED or not record.document_path:
        raise HTTPException(409, f"Job not ready: {record.status.value}")
    doc_path = Path(record.document_path)
    assets_dir = store.path(job_id) / "assets"
    return record, doc_path, assets_dir


def _asset_url(job_id: str, asset_name: str | None) -> str | None:
    if not asset_name:
        return None
    return f"/api/jobs/{job_id}/assets/{asset_name}"


def _job_to_summary(record) -> JobSummary:
    metrics = record.metrics or {}
    cerebras = metrics.get("cerebras") or {}
    return JobSummary(
        job_id=record.job_id,
        status=record.status.value,
        created_at=record.created_at,
        source_type=record.source_type,
        source_name=record.source_name,
        title=record.title,
        youtube_video_id=record.youtube_video_id,
        thumbnail_url=_asset_url(record.job_id, record.thumbnail_asset),
        preview_url=_asset_url(record.job_id, record.preview_asset),
        elapsed_sec=metrics.get("elapsed_sec"),
        cerebras_calls=cerebras.get("calls"),
    )


def _estimate_remaining_sec(record: JobRecord) -> float | None:
    metrics = record.metrics or {}
    if record.status in {JobStatus.COMPLETED, JobStatus.FAILED}:
        return 0.0 if record.status == JobStatus.COMPLETED else None

    elapsed = metrics.get("wall_elapsed_sec") or metrics.get("elapsed_sec")
    if elapsed is None:
        try:
            created = datetime.fromisoformat(record.created_at)
        except ValueError:
            created = None
        if created is not None:
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            elapsed = (datetime.now(timezone.utc) - created).total_seconds()

    progress = record.progress or 0
    if elapsed is not None and progress >= 8:
        # Progress-weighted ETA tracks real pipeline speed better than prep-time guess.
        remaining = elapsed * (100.0 - progress) / max(progress, 1.0)
        return max(0.0, remaining)

    total_min = metrics.get("estimated_total_minutes")
    if total_min is None or elapsed is None:
        return None
    return max(0.0, float(total_min) * 60.0 - elapsed)


def _job_to_status(record) -> JobStatusResponse:
    metrics = dict(record.metrics or {})
    if record.status not in {JobStatus.COMPLETED, JobStatus.FAILED} and metrics:
        apply_wall_elapsed(metrics, record.created_at)
    total_min = metrics.get("estimated_total_minutes")
    remaining = _estimate_remaining_sec(record)
    return JobStatusResponse(
        job_id=record.job_id,
        status=record.status.value,
        progress=record.progress,
        message=record.message,
        created_at=record.created_at,
        document_path=record.document_path,
        metrics=metrics,
        error=record.error,
        source_name=record.source_name,
        source_type=record.source_type,
        title=record.title,
        youtube_video_id=record.youtube_video_id,
        thumbnail_asset=record.thumbnail_asset,
        preview_asset=record.preview_asset,
        estimated_total_minutes=total_min,
        estimated_remaining_sec=round(remaining, 1) if remaining is not None else None,
        language=record.language,
        custom_prompt=record.custom_prompt,
    )


@router.get("", response_model=JobHistoryResponse)
def list_jobs(limit: int = 50):
    store = _get_store()
    records = store.list_jobs(limit=limit)
    jobs = [_job_to_summary(r) for r in records]
    return JobHistoryResponse(jobs=jobs, total=len(jobs))


@router.get("/{job_id}", response_model=JobStatusResponse)
def get_job(job_id: str):
    store = _get_store()
    try:
        record = store.load(job_id)
    except FileNotFoundError:
        raise HTTPException(404, "Job not found") from None
    return _job_to_status(record)


@router.delete("/{job_id}")
def delete_job(job_id: str):
    store = _get_store()
    try:
        store.delete(job_id)
    except FileNotFoundError:
        raise HTTPException(404, "Job not found") from None
    return {"deleted": True, "job_id": job_id}


@router.get("/{job_id}/document", response_model=DocumentResponse)
def get_document(job_id: str):
    store = _get_store()
    record, doc_path, _ = _require_completed_job(store, job_id)
    assets = sorted(p.name for p in (store.path(job_id) / "assets").glob("*"))
    return DocumentResponse(
        job_id=job_id,
        markdown=doc_path.read_text(encoding="utf-8"),
        assets=assets,
    )


@router.get("/{job_id}/export/html")
def export_html(job_id: str):
    store = _get_store()
    record, doc_path, assets_dir = _require_completed_job(store, job_id)
    markdown_text = doc_path.read_text(encoding="utf-8")
    title = record.title or "FastYoutubeReport"
    html_doc = render_html_document(markdown_text, assets_dir=assets_dir, title=title)
    filename = export_filename(record.title, job_id, "html")
    return Response(
        content=html_doc,
        media_type="text/html; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/{job_id}/export/pdf")
def export_pdf(job_id: str):
    store = _get_store()
    record, doc_path, assets_dir = _require_completed_job(store, job_id)
    markdown_text = doc_path.read_text(encoding="utf-8")
    title = record.title or "FastYoutubeReport"
    html_doc = render_html_document(markdown_text, assets_dir=assets_dir, title=title)
    try:
        pdf_bytes = render_pdf_bytes(html_doc)
    except RuntimeError as exc:
        raise HTTPException(500, str(exc)) from exc
    filename = export_filename(record.title, job_id, "pdf")
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/{job_id}/metrics")
def get_metrics(job_id: str):
    store = _get_store()
    try:
        record = store.load(job_id)
    except FileNotFoundError:
        raise HTTPException(404, "Job not found") from None
    metrics_path = store.metrics_path(job_id)
    return {
        "job_id": job_id,
        "metrics": record.metrics,
        "metrics_file": str(metrics_path) if metrics_path.exists() else None,
    }


@router.get("/{job_id}/metrics/file")
def download_metrics_file(job_id: str):
    store = _get_store()
    try:
        store.load(job_id)
    except FileNotFoundError:
        raise HTTPException(404, "Job not found") from None
    path = store.metrics_path(job_id)
    if not path.exists():
        raise HTTPException(404, "Metrics file not found")
    return FileResponse(
        path,
        media_type="application/json",
        filename=f"fastyoutubereport-metrics-{job_id[:8]}.json",
    )


@router.get("/{job_id}/stream")
def stream_document(job_id: str):
    store = _get_store()
    try:
        record = store.load(job_id)
    except FileNotFoundError:
        raise HTTPException(404, "Job not found") from None

    if record.document_path and record.status == JobStatus.COMPLETED:
        text = Path(record.document_path).read_text(encoding="utf-8")

        def replay():
            chunk_size = 80
            for i in range(0, len(text), chunk_size):
                yield f"data: {json.dumps({'token': text[i:i + chunk_size]})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"

        return StreamingResponse(replay(), media_type="text/event-stream")

    raise HTTPException(409, "Document not ready; poll /api/jobs/{id} until completed")


@router.get("/{job_id}/assets/{name}")
def get_asset(job_id: str, name: str):
    store = _get_store()
    path = store.path(job_id) / "assets" / name
    if not path.exists():
        raise HTTPException(404, "Asset not found")
    return FileResponse(path)