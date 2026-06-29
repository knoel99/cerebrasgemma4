"""End-to-end video-to-document pipeline."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from cerebrasgemma4.pipeline.context import VideoContext, save_context
from cerebrasgemma4.pipeline.demo import (
    DEMO_MAX_EXTRACT_FRAMES,
    DEMO_SAMPLES_PER_REGION,
    DEMO_SCOUT_MAX_HEIGHT,
    is_demo_mode,
)
from cerebrasgemma4.pipeline.frames import (
    chunk_frames,
    extract_frames,
    extract_frames_sparse,
    extract_report_frames_batch,
    plan_demo_timestamps,
)
from cerebrasgemma4.pipeline.gemma.analyze import analyze_frame
from cerebrasgemma4.pipeline.gemma.compose import ComposeInput, compose_complete
from cerebrasgemma4.pipeline.chapters import VideoChapter
from cerebrasgemma4.pipeline.gemma.scout import apply_duplicate_penalties, select_top_frames
from cerebrasgemma4.pipeline.gemma.scout_strategy import run_hierarchical_scout
from cerebrasgemma4.pipeline.ingest import probe_video
from cerebrasgemma4.pipeline.jobs import JobStatus, JobStore
from cerebrasgemma4.pipeline.perf import (
    PerfTracker,
    apply_wall_elapsed,
    normalize_api_call,
    save_metrics_file,
)
from cerebrasgemma4.pipeline.rate_limit import CerebrasRateLimiter
from cerebrasgemma4.pipeline.transcript import TranscriptResult, get_transcript, segments_in_range


@dataclass
class ConvertOptions:
    max_frames: int = 8
    max_duration_sec: float | None = 120.0
    language: str = "auto"
    youtube_url: str | None = None
    custom_prompt: str | None = None
    chapters: list[VideoChapter] | None = None


def _document_title(markdown: str, fallback: str) -> str:
    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()[:200]
    return fallback.rsplit("/", 1)[-1][:200]


def _wait_rate_limit(
    limiter: CerebrasRateLimiter,
    store: JobStore,
    job_id: str,
    *,
    token_estimate: int | None = None,
) -> None:
    pause = limiter.wait_for_slot(token_estimate=token_estimate)
    if pause >= 0.5:
        snap = limiter.snapshot()
        store.update(
            job_id,
            message=(
                f"Rate limit pause ({pause:.0f}s) · "
                f"{snap['requests_in_window']}/{snap['rpm_cap']} req/min · "
                f"{snap['tokens_in_window']:,}/{snap['tpm_cap']:,} tok/min"
            ),
        )


def _push_metrics(
    store: JobStore,
    job_id: str,
    perf: PerfTracker,
    *,
    limiter: CerebrasRateLimiter | None = None,
    **extra,
) -> None:
    snap = perf.snapshot()
    snap.update(extra)
    if limiter is not None:
        snap["rate_limit"] = limiter.snapshot()
    record = store.load(job_id)
    apply_wall_elapsed(snap, record.created_at)
    store.update(job_id, metrics=snap)
    save_metrics_file(store.metrics_path(job_id), snap)


def _cleanup_source_video(video_path: Path) -> None:
    """Remove the local source video once the document is ready."""
    try:
        if video_path.is_file():
            video_path.unlink()
    except OSError:
        pass





def run_pipeline(
    store: JobStore,
    job_id: str,
    video_path: Path,
    source_name: str,
    options: ConvertOptions,
) -> Path:
    perf = PerfTracker()
    existing = store.load(job_id)
    if existing.metrics:
        perf.merge_prep(existing.metrics)
    limiter = CerebrasRateLimiter()
    job_dir = store.path(job_id)
    assets_dir = job_dir / "assets"

    store.update(job_id, status=JobStatus.EXTRACTING, progress=5, message="Probing video")
    with perf.step("probe", "Video probe (ffprobe)", kind="local") as step:
        meta = probe_video(video_path)
        step["detail"] = {
            "duration_sec": round(meta.duration_sec, 2),
            "resolution": f"{meta.width}x{meta.height}",
            "fps": round(meta.fps, 2),
        }
    _push_metrics(store, job_id, perf, limiter=limiter)

    store.update(job_id, status=JobStatus.EXTRACTING, progress=15, message="Extracting frames")
    extract_label = (
        "Sparse frame extraction (demo)"
        if is_demo_mode()
        else "Frame extraction (1 fps)"
    )
    with perf.step("extract_frames", extract_label, kind="local") as step:
        _push_metrics(store, job_id, perf, limiter=limiter)
        frames_dir = job_dir / "frames"

        def _extract_progress(done: int, total: int) -> None:
            step["detail"] = {"sparse": True, "extracted": done, "planned": total}
            _push_metrics(store, job_id, perf, limiter=limiter)

        if is_demo_mode():
            timestamps = plan_demo_timestamps(
                meta.duration_sec,
                options.chapters,
                samples_per_region=DEMO_SAMPLES_PER_REGION,
                max_frames=DEMO_MAX_EXTRACT_FRAMES,
            )
            step["detail"] = {"sparse": True, "extracted": 0, "planned": len(timestamps)}
            _push_metrics(store, job_id, perf, limiter=limiter)
            frames = extract_frames_sparse(
                video_path,
                frames_dir,
                timestamps,
                max_height=DEMO_SCOUT_MAX_HEIGHT,
                on_progress=_extract_progress,
            )
            step["detail"] = {
                "frame_count": len(frames),
                "sparse": True,
                "timestamps": len(timestamps),
            }
        else:
            frames = extract_frames(
                video_path,
                frames_dir,
                fps=1.0,
                max_duration_sec=options.max_duration_sec,
            )
            step["detail"] = {"frame_count": len(frames)}
        chunks = chunk_frames(frames, chunk_size=5)
        frame_paths = {f.frame_id: f.path for f in frames}
        step["detail"]["chunk_count"] = len(chunks)
    _push_metrics(store, job_id, perf, limiter=limiter)

    store.update(job_id, status=JobStatus.TRANSCRIBING, progress=25, message="Transcribing")
    with perf.step("transcript", "Transcription", kind="local") as step:
        _push_metrics(store, job_id, perf, limiter=limiter)
        try:
            transcript = get_transcript(
                video_path=video_path,
                youtube_url=options.youtube_url,
                language=options.language,
            )
        except Exception as exc:
            transcript = TranscriptResult(segments=[], source="none", full_text="")
            step["detail"] = {"source": "none", "error": str(exc)}
        else:
            step["detail"] = {
                "source": transcript.source,
                "segment_count": len(transcript.segments),
            }
    _push_metrics(store, job_id, perf, limiter=limiter)

    chapter_count = len(options.chapters or [])
    store.update(
        job_id,
        status=JobStatus.ANALYZING,
        progress=35,
        message="Global scout (overview)"
        if chapter_count or len(chunks) > 3
        else "Scouting frames",
    )
    scout_meta: dict = {}
    with perf.step(
        "scout",
        "Scout frames (Gemma 4 · Cerebras)",
        kind="cerebras",
        detail={"chapter_count": chapter_count},
    ) as scout_step:
        _push_metrics(store, job_id, perf, limiter=limiter)
        scout_calls_done = 0

        def on_global_batch(done: int, total: int) -> None:
            scout_step["detail"]["phase"] = "global"
            scout_step["detail"]["planned_calls"] = total
            pct = 35 + (12 * (done - 1) / max(total, 1))
            store.update(
                job_id,
                progress=pct,
                message=f"Global overview {done}/{total}",
            )

        def on_detail_chunk(done: int, total: int, mode: str) -> None:
            scout_step["detail"]["phase"] = mode
            scout_step["detail"]["planned_calls"] = total
            pct = 50 + (15 * done / max(total, 1)) if mode == "detail" else 35 + (
                30 * done / max(total, 1)
            )
            label = (
                f"Detail scout {done}/{total}"
                if mode == "detail"
                else f"Scouting chunk {done}/{total}"
            )
            store.update(job_id, progress=pct, message=label)

        def after_scout_call(record: dict) -> None:
            nonlocal scout_calls_done
            limiter.record(record.get("usage"))
            perf.record_api_call(normalize_api_call(**record))
            scout_calls_done += 1
            scout_step["detail"]["completed_calls"] = scout_calls_done
            _push_metrics(store, job_id, perf, limiter=limiter, scout_calls=scout_calls_done)

        all_scores, scout_meta = run_hierarchical_scout(
            frames=frames,
            chunks=chunks,
            duration_sec=meta.duration_sec,
            transcript_segments=transcript.segments,
            max_frames=options.max_frames,
            chapters=options.chapters,
            on_global_batch=on_global_batch,
            on_detail_chunk=on_detail_chunk,
            before_call=lambda: _wait_rate_limit(limiter, store, job_id),
            after_call=after_scout_call,
        )
        scout_step["detail"].update(scout_meta)

    all_scores = apply_duplicate_penalties(all_scores, frame_paths)
    selected = select_top_frames(all_scores, frame_paths, max_frames=options.max_frames)
    preview_asset: str | None = None
    asset_jobs: list[tuple[float, Path]] = []
    if selected:
        best_score, _best_path = max(selected, key=lambda item: item[0].relevance)
        preview_asset = "preview.jpg"
        asset_jobs.append((best_score.timestamp_sec, assets_dir / preview_asset))

    store.update(job_id, progress=70, message="Analyzing key frames")
    analyses = []
    with perf.step(
        "analyze",
        "Frame analysis (Gemma 4 · Cerebras)",
        kind="cerebras",
        detail={"selected_frames": len(selected)},
    ) as analyze_step:
        _push_metrics(store, job_id, perf, limiter=limiter)
        for i, (score, path) in enumerate(selected):
            asset_name = f"frame_{int(score.timestamp_sec):04d}.jpg"
            asset_jobs.append((score.timestamp_sec, assets_dir / asset_name))
            excerpt = segments_in_range(
                transcript.segments,
                max(0, score.timestamp_sec - 2),
                score.timestamp_sec + 3,
            )
            _wait_rate_limit(limiter, store, job_id)
            call_t0 = time.perf_counter()
            analysis, raw = analyze_frame(score, path, excerpt, asset_name=asset_name)
            call_wall = time.perf_counter() - call_t0
            limiter.record(raw.get("usage"))
            analyses.append(analysis)
            perf.record_api_call(
                normalize_api_call(
                    stage="analyze",
                    label=f"Analyze {score.frame_id}",
                    wall_sec=call_wall,
                    usage=raw.get("usage"),
                    time_info=raw.get("time_info"),
                    extra={
                        "frame_id": score.frame_id,
                        "timestamp_sec": score.timestamp_sec,
                    },
                )
            )
            analyze_step["detail"]["completed_calls"] = i + 1
            pct = 70 + (20 * (i + 1) / max(len(selected), 1))
            store.update(
                job_id, progress=pct, message=f"Analyzing frame {i + 1}/{len(selected)}"
            )
            _push_metrics(store, job_id, perf, limiter=limiter, analyze_calls=i + 1)

    if asset_jobs:
        report_video = video_path
        with perf.step("hd_assets", "Report screenshots (720p)", kind="local") as hd_step:
            _push_metrics(store, job_id, perf, limiter=limiter)
            extract_report_frames_batch(report_video, asset_jobs)
            hd_step["detail"] = {
                "count": len(asset_jobs),
                "resolution": "720p",
                "source": report_video.name,
            }
        _push_metrics(store, job_id, perf, limiter=limiter)

    store.update(job_id, status=JobStatus.COMPOSING, progress=92, message="Writing document")
    with perf.step("compose", "Document writing (Gemma 4 · Cerebras)", kind="cerebras") as step:
        _push_metrics(store, job_id, perf, limiter=limiter)
        compose_data = ComposeInput(
            source_name=source_name,
            duration_sec=meta.duration_sec,
            transcript=transcript,
            analyses=analyses,
            custom_prompt=options.custom_prompt,
        )
        _wait_rate_limit(limiter, store, job_id, token_estimate=16_000)
        call_t0 = time.perf_counter()
        markdown, raw = compose_complete(compose_data)
        call_wall = time.perf_counter() - call_t0
        limiter.record(raw.get("usage"))
        perf.record_api_call(
            normalize_api_call(
                stage="compose",
                label="Compose Markdown",
                wall_sec=call_wall,
                usage=raw.get("usage"),
                time_info=raw.get("time_info"),
            )
        )
        step["detail"] = {
            "completion_tokens": perf.api_calls[-1].get("completion_tokens"),
            "output_tokens_per_sec": perf.api_calls[-1].get("output_tokens_per_sec"),
        }

    doc_path = job_dir / "document.md"
    doc_path.write_text(markdown, encoding="utf-8")

    save_context(
        job_dir,
        VideoContext.from_pipeline(
            source_name=source_name,
            duration_sec=meta.duration_sec,
            transcript=transcript,
            analyses=analyses,
        ),
    )

    final = perf.snapshot()
    final.update(
        {
            "scout_calls": sum(1 for c in perf.api_calls if c["stage"] == "scout"),
            "analyze_calls": sum(1 for c in perf.api_calls if c["stage"] == "analyze"),
            "frame_count": len(frames),
            "selected_frames": len(analyses),
            "transcript_source": transcript.source,
            "rate_limit": limiter.snapshot(),
            "scout_strategy": scout_meta.get("strategy"),
            "chapter_count": scout_meta.get("chapter_count", chapter_count),
        }
    )
    existing = store.load(job_id)
    final_title = existing.title or _document_title(markdown, source_name)
    store.update(
        job_id,
        status=JobStatus.COMPLETED,
        progress=100,
        message="Done",
        document_path=str(doc_path),
        metrics=final,
        title=final_title,
        preview_asset=preview_asset,
    )
    _cleanup_source_video(video_path)
    return doc_path