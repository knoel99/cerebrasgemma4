"""Frame extraction aligned with Gemma 4 native video mode (1 fps)."""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from cerebrasgemma4.pipeline.chapters import plan_scout_regions

REPORT_MIN_HEIGHT = 720
SCOUT_MAX_HEIGHT = 720
REPORT_JPEG_QSCALE = 2


@dataclass
class Frame:
    frame_id: str
    timestamp_sec: float
    path: Path


@dataclass
class FrameChunk:
    chunk_id: int
    start_sec: float
    end_sec: float
    frames: list[Frame]


def _video_filters(*, fps: float | None = None, max_height: int | None = None) -> str:
    parts: list[str] = []
    if fps is not None:
        parts.append(f"fps={fps}")
    if max_height is not None:
        parts.append(f"scale=-2:'min({max_height},ih)'")
    return ",".join(parts)


def extract_frames(
    video_path: Path,
    output_dir: Path,
    *,
    fps: float = 1.0,
    max_duration_sec: float | None = None,
    max_height: int | None = SCOUT_MAX_HEIGHT,
) -> list[Frame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(output_dir / "frame_%06d.jpg")
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vf",
        _video_filters(fps=fps, max_height=max_height),
    ]
    if max_duration_sec is not None:
        cmd.extend(["-t", str(max_duration_sec)])
    cmd.extend(["-q:v", "3", pattern, "-y"])
    subprocess.run(cmd, check=True)

    frames: list[Frame] = []
    for i, path in enumerate(sorted(output_dir.glob("frame_*.jpg"))):
        ts = i / fps
        frames.append(Frame(frame_id=_frame_id(ts), timestamp_sec=ts, path=path))
    return frames


def _frame_id(timestamp_sec: float) -> str:
    return f"f_{int(timestamp_sec):04d}"


def plan_demo_timestamps(
    duration_sec: float,
    chapters: list | None,
    *,
    samples_per_region: int = 8,
    max_frames: int = 72,
) -> list[float]:
    """Sparse timestamps for demo scout (seek-based, no full-video decode)."""
    regions = plan_scout_regions(duration_sec, chapters)
    stamps: set[float] = set()
    for region in regions:
        span = max(0.5, region.end_sec - region.start_sec)
        stamps.add(round((region.start_sec + region.end_sec) / 2, 2))
        count = max(2, samples_per_region)
        for i in range(count):
            t = region.start_sec + span * i / max(1, count - 1)
            stamps.add(round(min(duration_sec, t), 2))
    ordered = sorted(stamps)
    if len(ordered) <= max_frames:
        return ordered
    step = (len(ordered) - 1) / (max_frames - 1)
    return [ordered[round(i * step)] for i in range(max_frames)]


def _extract_single_scout_frame(
    video_path: Path,
    output_dir: Path,
    timestamp_sec: float,
    *,
    max_height: int,
) -> Frame:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_id = _frame_id(timestamp_sec)
    out_path = output_dir / f"{frame_id}.jpg"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{max(0.0, timestamp_sec):.3f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-vf",
        f"scale=-2:'min({max_height},ih)'",
        "-q:v",
        "4",
        str(out_path),
        "-y",
    ]
    subprocess.run(cmd, check=True)
    return Frame(frame_id=frame_id, timestamp_sec=timestamp_sec, path=out_path)


def extract_frames_sparse(
    video_path: Path,
    output_dir: Path,
    timestamps: list[float],
    *,
    max_height: int = SCOUT_MAX_HEIGHT,
    max_workers: int = 8,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[Frame]:
    """Extract scout frames via parallel seek (fast on long videos)."""
    if not timestamps:
        raise ValueError("sparse extraction requires timestamps")
    unique = sorted({round(t, 2) for t in timestamps})
    frames: list[Frame] = []
    workers = min(max_workers, max(1, len(unique)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _extract_single_scout_frame,
                video_path,
                output_dir,
                ts,
                max_height=max_height,
            ): ts
            for ts in unique
        }
        for future in as_completed(futures):
            frames.append(future.result())
            if on_progress:
                on_progress(len(frames), len(unique))
    return sorted(frames, key=lambda f: f.timestamp_sec)


def extract_report_frames_batch(
    video_path: Path,
    items: list[tuple[float, Path]],
    *,
    min_height: int = REPORT_MIN_HEIGHT,
    max_workers: int = 6,
) -> None:
    """Extract multiple report screenshots in parallel."""
    if not items:
        return
    workers = min(max_workers, max(1, len(items)))

    def _one(ts: float, dest: Path) -> None:
        extract_report_frame(video_path, ts, dest, min_height=min_height)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_one, ts, dest) for ts, dest in items]
        for future in futures:
            future.result()


def extract_report_frame(
    video_path: Path,
    timestamp_sec: float,
    output_path: Path,
    *,
    min_height: int = REPORT_MIN_HEIGHT,
    qscale: int = REPORT_JPEG_QSCALE,
) -> Path:
    """Extract one high-quality frame for report assets (height ≥ min_height)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{max(0.0, timestamp_sec):.3f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-vf",
        f"scale=-2:{min_height}",
        "-q:v",
        str(qscale),
        str(output_path),
        "-y",
    ]
    subprocess.run(cmd, check=True)
    return output_path


def chunk_frames(frames: list[Frame], chunk_size: int = 5) -> list[FrameChunk]:
    chunks: list[FrameChunk] = []
    for i in range(0, len(frames), chunk_size):
        batch = frames[i : i + chunk_size]
        if not batch:
            continue
        chunks.append(
            FrameChunk(
                chunk_id=i // chunk_size,
                start_sec=batch[0].timestamp_sec,
                end_sec=batch[-1].timestamp_sec,
                frames=batch,
            )
        )
    return chunks


def segment_chunks(chunks: list[FrameChunk], segment_sec: float = 60.0) -> list[list[FrameChunk]]:
    if not chunks:
        return []
    segments: list[list[FrameChunk]] = []
    current: list[FrameChunk] = []
    seg_start = chunks[0].start_sec
    for chunk in chunks:
        if chunk.start_sec - seg_start >= segment_sec and current:
            segments.append(current)
            current = []
            seg_start = chunk.start_sec
        current.append(chunk)
    if current:
        segments.append(current)
    return segments