from __future__ import annotations

import asyncio
import traceback
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile

from cerebrasgemma4.api.schemas import (
    ConvertOptionsSchema,
    ConvertResponse,
    DefaultsResponse,
    ProbeResponse,
)
from cerebrasgemma4.pipeline.chapters import fetch_youtube_chapters
from cerebrasgemma4.pipeline.gemma.compose import COMPOSE_DEFAULT_INSTRUCTIONS
from cerebrasgemma4.pipeline.demo import is_demo_mode
from cerebrasgemma4.pipeline.ingest import (
    download_url_to_file,
    download_youtube,
    download_youtube_processing,
    fetch_youtube_metadata,
    probe_video,
    probe_youtube,
    save_upload,
    youtube_video_id,
)
from cerebrasgemma4.pipeline.options import HACKATHON_RPM, HACKATHON_TPM, suggest_convert_options
from cerebrasgemma4.pipeline.jobs import JobStatus, JobStore
from cerebrasgemma4.pipeline.orchestrator import ConvertOptions, run_pipeline
from cerebrasgemma4.pipeline.perf import PerfTracker, apply_wall_elapsed, save_metrics_file

router = APIRouter(prefix="/api", tags=["convert"])

_PROBE_NOTE = (
    f"Hackathon limits: {HACKATHON_RPM} req/min, {HACKATHON_TPM // 1000}K tok/min. "
    "Full video is processed; calls are paced automatically when limits are approached."
)


def _get_store() -> JobStore:
    return JobStore()


def _mb(path: Path) -> float:
    try:
        return round(path.stat().st_size / (1024 * 1024), 2)
    except OSError:
        return 0.0


def _run_job(
    store: JobStore,
    job_id: str,
    video_path: Path,
    source_name: str,
    options: ConvertOptions,
    youtube_url: str | None,
) -> None:
    try:
        opts = ConvertOptions(
            max_frames=options.max_frames,
            max_duration_sec=options.max_duration_sec,
            language=options.language,
            youtube_url=youtube_url,
            custom_prompt=options.custom_prompt,
            chapters=options.chapters,
        )
        run_pipeline(store, job_id, video_path, source_name, opts)
    except Exception:
        record = store.load(job_id)
        if record.metrics:
            save_metrics_file(store.metrics_path(job_id), record.metrics)
        store.update(
            job_id,
            status=JobStatus.FAILED,
            error=traceback.format_exc(),
            message="Pipeline failed",
        )


@router.get("/defaults", response_model=DefaultsResponse)
def get_defaults():
    return DefaultsResponse(compose_prompt=COMPOSE_DEFAULT_INSTRUCTIONS)


@router.post("/probe", response_model=ProbeResponse)
async def probe_video_source(
    file: UploadFile | None = File(default=None),
    youtube_url: str | None = Form(default=None),
):
    """Probe video metadata and suggest max_duration / max_frames."""
    if not file and not youtube_url:
        raise HTTPException(400, "Provide a video file or youtube_url")

    import tempfile

    tmp = Path(tempfile.mkdtemp())
    try:
        if youtube_url:
            if not youtube_video_id(youtube_url):
                raise HTTPException(400, "Invalid YouTube URL")
            try:
                meta = await asyncio.to_thread(probe_youtube, youtube_url)
            except RuntimeError as exc:
                raise HTTPException(502, str(exc)) from exc
        else:
            assert file is not None
            suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
            video_path = tmp / f"probe{suffix}"
            data = await file.read()
            await asyncio.to_thread(save_upload, video_path, data)
            meta = await asyncio.to_thread(probe_video, video_path)
        chapter_count = 0
        if youtube_url:
            chapters = await asyncio.to_thread(fetch_youtube_chapters, youtube_url)
            chapter_count = len(chapters)
        s = suggest_convert_options(
            meta.duration_sec,
            width=meta.width,
            height=meta.height,
            fps=meta.fps,
            chapter_count=chapter_count,
            youtube=bool(youtube_url),
        )
        return ProbeResponse(
            duration_sec=s.duration_sec,
            max_duration_sec=s.max_duration_sec,
            max_frames=s.max_frames,
            width=s.width,
            height=s.height,
            fps=s.fps,
            estimated_scout_calls=s.estimated_scout_calls,
            estimated_analyze_calls=s.estimated_analyze_calls,
            estimated_total_api_calls=s.estimated_total_api_calls,
            within_hackathon_rpm=s.within_hackathon_rpm,
            hackathon_capped=s.hackathon_capped,
            estimated_pipeline_minutes=s.estimated_pipeline_minutes,
            estimated_total_minutes=s.estimated_total_minutes,
            hackathon_rpm=s.hackathon_rpm,
            hackathon_tpm=s.hackathon_tpm,
            note=_PROBE_NOTE,
        )
    finally:
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)


@router.post("/convert", response_model=ConvertResponse)
async def convert_video(
    background_tasks: BackgroundTasks,
    file: UploadFile | None = File(default=None),
    youtube_url: str | None = Form(default=None),
    language: str = Form(default="auto"),
    custom_prompt: str | None = Form(default=None),
):
    if not file and not youtube_url:
        raise HTTPException(400, "Provide a video file or youtube_url")

    opts = ConvertOptionsSchema(language=language, custom_prompt=custom_prompt)
    store = _get_store()
    record = store.create()
    job_id = record.job_id
    job_dir = store.path(job_id)

    perf = PerfTracker()
    job_title: str | None = None
    yt_video_id: str | None = None
    thumbnail_asset: str | None = None

    try:
        if youtube_url:
            if not youtube_video_id(youtube_url):
                raise HTTPException(400, "Invalid YouTube URL")

            with perf.step("youtube_metadata", "YouTube metadata", kind="local") as step:
                yt_meta = await asyncio.to_thread(fetch_youtube_metadata, youtube_url)
                if yt_meta:
                    job_title = yt_meta.title
                    yt_video_id = yt_meta.video_id
                    step["detail"] = {"video_id": yt_video_id, "title": job_title}

            if yt_meta:
                thumb_path = job_dir / "assets" / "youtube_thumbnail.jpg"
                with perf.step("thumbnail_download", "YouTube thumbnail", kind="local") as step:
                    await asyncio.to_thread(download_url_to_file, yt_meta.thumbnail_url, thumb_path)
                    thumbnail_asset = "youtube_thumbnail.jpg"
                    step["detail"] = {"size_kb": round(thumb_path.stat().st_size / 1024, 1)}

            video_path = job_dir / "source.mp4"
            download_label = (
                "YouTube video download (720p)"
                if is_demo_mode()
                else "YouTube video download"
            )
            with perf.step("youtube_download", download_label, kind="local") as step:
                try:
                    if is_demo_mode():
                        await asyncio.to_thread(
                            download_youtube_processing, youtube_url, video_path
                        )
                    else:
                        await asyncio.to_thread(download_youtube, youtube_url, video_path)
                except RuntimeError as exc:
                    raise HTTPException(502, str(exc)) from exc
                step["detail"] = {
                    "size_mb": _mb(video_path),
                    "demo_mode": is_demo_mode(),
                }
            source_name = youtube_url
            yt_url = youtube_url

        else:
            assert file is not None
            suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
            video_path = job_dir / f"source{suffix}"
            data = await file.read()
            with perf.step("file_upload", "File upload", kind="local") as step:
                await asyncio.to_thread(save_upload, video_path, data)
                step["detail"] = {"filename": file.filename, "size_mb": _mb(video_path)}
            source_name = file.filename or "upload.mp4"
            yt_url = None

        with perf.step("video_probe", "Video probe (ingest)", kind="local") as step:
            meta = await asyncio.to_thread(probe_video, video_path)
            step["detail"] = {
                "duration_sec": round(meta.duration_sec, 2),
                "resolution": f"{meta.width}x{meta.height}",
            }

        chapters = None
        if yt_url:
            chapters = await asyncio.to_thread(fetch_youtube_chapters, yt_url)
        suggestions = suggest_convert_options(
            meta.duration_sec,
            width=meta.width,
            height=meta.height,
            fps=meta.fps,
            chapter_count=len(chapters or []),
            youtube=bool(yt_url),
        )

        convert_opts = ConvertOptions(
            max_frames=suggestions.max_frames,
            max_duration_sec=suggestions.max_duration_sec,
            language=opts.language,
            youtube_url=yt_url,
            custom_prompt=opts.custom_prompt,
            chapters=chapters or None,
        )

        prep_metrics = perf.snapshot()
        prep_metrics["estimated_total_minutes"] = suggestions.estimated_total_minutes
        prep_metrics["estimated_pipeline_minutes"] = suggestions.estimated_pipeline_minutes
        if chapters:
            prep_metrics["chapter_count"] = len(chapters)
        apply_wall_elapsed(prep_metrics, store.load(job_id).created_at)
        store.update(
            job_id,
            status=JobStatus.PENDING,
            message="Queued",
            source_name=source_name,
            source_type="youtube" if yt_url else "file",
            title=job_title,
            youtube_video_id=yt_video_id,
            thumbnail_asset=thumbnail_asset,
            language=opts.language,
            custom_prompt=opts.custom_prompt,
            metrics=prep_metrics,
        )
        save_metrics_file(store.metrics_path(job_id), prep_metrics)

        background_tasks.add_task(
            _run_job,
            store,
            job_id,
            video_path,
            source_name,
            convert_opts,
            yt_url,
        )
        return ConvertResponse(job_id=job_id, status=JobStatus.PENDING.value)
    except HTTPException:
        prep_metrics = perf.snapshot()
        if prep_metrics.get("steps"):
            store.update(job_id, status=JobStatus.FAILED, metrics=prep_metrics, message="Ingest failed")
            save_metrics_file(store.metrics_path(job_id), prep_metrics)
        raise