"""End-to-end video-to-document pipeline."""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from cerebrasgemma4.pipeline.frames import chunk_frames, extract_frames
from cerebrasgemma4.pipeline.gemma.analyze import analyze_frame
from cerebrasgemma4.pipeline.gemma.compose import ComposeInput, compose_complete
from cerebrasgemma4.pipeline.gemma.scout import scout_chunk, select_top_frames
from cerebrasgemma4.pipeline.ingest import probe_video
from cerebrasgemma4.pipeline.jobs import JobStatus, JobStore
from cerebrasgemma4.pipeline.perf import PerfTracker, normalize_api_call, save_metrics_file
from cerebrasgemma4.pipeline.rate_limit import CerebrasRateLimiter
from cerebrasgemma4.pipeline.transcript import TranscriptResult, get_transcript, segments_in_range


@dataclass
class ConvertOptions:
    max_frames: int = 8
    max_duration_sec: float | None = 120.0
    language: str = "auto"
    youtube_url: str | None = None


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


def _push_metrics(store: JobStore, job_id: str, perf: PerfTracker, **extra) -> None:
    snap = perf.snapshot()
    snap.update(extra)
    store.update(job_id, metrics=snap)
    save_metrics_file(store.metrics_path(job_id), snap)


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
    _push_metrics(store, job_id, perf)

    store.update(job_id, status=JobStatus.EXTRACTING, progress=15, message="Extracting frames")
    with perf.step("extract_frames", "Frame extraction (1 fps)", kind="local") as step:
        frames_dir = job_dir / "frames"
        frames = extract_frames(
            video_path,
            frames_dir,
            fps=1.0,
            max_duration_sec=options.max_duration_sec,
        )
        chunks = chunk_frames(frames, chunk_size=5)
        frame_paths = {f.frame_id: f.path for f in frames}
        step["detail"] = {"frame_count": len(frames), "chunk_count": len(chunks)}
    _push_metrics(store, job_id, perf)

    store.update(job_id, status=JobStatus.TRANSCRIBING, progress=25, message="Transcribing")
    with perf.step("transcript", "Transcription", kind="local") as step:
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
    _push_metrics(store, job_id, perf)

    store.update(job_id, status=JobStatus.ANALYZING, progress=35, message="Scouting frames")
    all_scores = []
    with perf.step(
        "scout",
        "Scout frames (Gemma 4 · Cerebras)",
        kind="cerebras",
        detail={"calls": len(chunks)},
    ) as scout_step:
        for i, chunk in enumerate(chunks):
            excerpt = segments_in_range(
                transcript.segments, chunk.start_sec, chunk.end_sec + 1
            )
            _wait_rate_limit(limiter, store, job_id)
            call_t0 = time.perf_counter()
            scores, raw = scout_chunk(chunk, excerpt)
            call_wall = time.perf_counter() - call_t0
            limiter.record(raw.get("usage"))
            all_scores.extend(scores)
            perf.record_api_call(
                normalize_api_call(
                    stage="scout",
                    label=f"Scout chunk {i + 1}/{len(chunks)}",
                    wall_sec=call_wall,
                    usage=raw.get("usage"),
                    time_info=raw.get("time_info"),
                    extra={"chunk": i, "frames": len(chunk.frames)},
                )
            )
            scout_step["detail"]["completed_calls"] = i + 1
            pct = 35 + (30 * (i + 1) / max(len(chunks), 1))
            store.update(
                job_id, progress=pct, message=f"Scouting chunk {i + 1}/{len(chunks)}"
            )
            _push_metrics(store, job_id, perf, scout_calls=i + 1)

    selected = select_top_frames(all_scores, frame_paths, max_frames=options.max_frames)
    preview_asset: str | None = None
    if selected:
        best_score, best_path = max(selected, key=lambda item: item[0].relevance)
        preview_asset = "preview.jpg"
        shutil.copy2(best_path, assets_dir / preview_asset)

    store.update(job_id, progress=70, message="Analyzing key frames")
    analyses = []
    with perf.step(
        "analyze",
        "Frame analysis (Gemma 4 · Cerebras)",
        kind="cerebras",
        detail={"selected_frames": len(selected)},
    ) as analyze_step:
        for i, (score, path) in enumerate(selected):
            asset_name = f"frame_{int(score.timestamp_sec):04d}.jpg"
            shutil.copy2(path, assets_dir / asset_name)
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
            _push_metrics(store, job_id, perf, analyze_calls=i + 1)

    store.update(job_id, status=JobStatus.COMPOSING, progress=92, message="Writing document")
    with perf.step("compose", "Document writing (Gemma 4 · Cerebras)", kind="cerebras") as step:
        compose_data = ComposeInput(
            source_name=source_name,
            duration_sec=meta.duration_sec,
            transcript=transcript,
            analyses=analyses,
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

    final = perf.snapshot()
    final.update(
        {
            "scout_calls": sum(1 for c in perf.api_calls if c["stage"] == "scout"),
            "analyze_calls": sum(1 for c in perf.api_calls if c["stage"] == "analyze"),
            "frame_count": len(frames),
            "selected_frames": len(analyses),
            "transcript_source": transcript.source,
            "rate_limit": limiter.snapshot(),
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
    return doc_path