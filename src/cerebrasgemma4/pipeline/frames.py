"""Frame extraction aligned with Gemma 4 native video mode (1 fps)."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


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


def extract_frames(
    video_path: Path,
    output_dir: Path,
    *,
    fps: float = 1.0,
    max_duration_sec: float | None = None,
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
        f"fps={fps}",
    ]
    if max_duration_sec is not None:
        cmd.extend(["-t", str(max_duration_sec)])
    cmd.extend(["-q:v", "3", pattern, "-y"])
    subprocess.run(cmd, check=True)

    frames: list[Frame] = []
    for i, path in enumerate(sorted(output_dir.glob("frame_*.jpg"))):
        ts = i / fps
        frames.append(
            Frame(
                frame_id=f"f_{int(ts):04d}",
                timestamp_sec=ts,
                path=path,
            )
        )
    return frames


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