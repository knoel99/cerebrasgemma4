import subprocess
from pathlib import Path

import pytest

from cerebrasgemma4.pipeline.frames import (
    REPORT_MIN_HEIGHT,
    chunk_frames,
    extract_frames,
    extract_frames_sparse,
    extract_report_frame,
    plan_demo_timestamps,
    segment_chunks,
)


@pytest.fixture
def sample_video(tmp_path: Path) -> Path:
    out = tmp_path / "test.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=5:size=320x240:rate=10",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=5",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-shortest",
            str(out),
            "-y",
        ],
        check=True,
    )
    return out


def test_extract_frames_at_1fps(sample_video: Path, tmp_path: Path):
    frames = extract_frames(sample_video, tmp_path / "frames", fps=1.0)
    assert len(frames) == 5
    assert frames[0].timestamp_sec == 0.0
    assert frames[-1].timestamp_sec == 4.0
    assert all(f.path.exists() for f in frames)


def test_chunk_frames(sample_video: Path, tmp_path: Path):
    frames = extract_frames(sample_video, tmp_path / "frames", fps=1.0)
    chunks = chunk_frames(frames, chunk_size=5)
    assert len(chunks) == 1
    assert len(chunks[0].frames) == 5
    assert chunks[0].start_sec == 0.0
    assert chunks[0].end_sec == 4.0


def test_plan_demo_timestamps_caps_count():
    stamps = plan_demo_timestamps(600, None, samples_per_region=6, max_frames=20)
    assert len(stamps) <= 20
    assert stamps == sorted(stamps)


def test_extract_frames_sparse(sample_video: Path, tmp_path: Path):
    stamps = [0.0, 1.0, 2.0, 3.0]
    frames = extract_frames_sparse(sample_video, tmp_path / "sparse", stamps, max_height=240)
    assert len(frames) == 4
    assert all(f.path.exists() for f in frames)


def test_extract_report_frame_at_least_720p(sample_video: Path, tmp_path: Path):
    out = tmp_path / "report.jpg"
    extract_report_frame(sample_video, 2.0, out)
    assert out.exists()
    from PIL import Image

    with Image.open(out) as img:
        assert img.height >= REPORT_MIN_HEIGHT


def test_segment_chunks():
    from cerebrasgemma4.pipeline.frames import Frame, FrameChunk

    chunks = [
        FrameChunk(i, i * 5, i * 5 + 4, [Frame(f"f{i}", float(i * 5), Path(f"/tmp/{i}.jpg"))])
        for i in range(20)
    ]
    segments = segment_chunks(chunks, segment_sec=60.0)
    assert len(segments) == 2
    assert sum(len(s) for s in segments) == 20
    assert segments[0][0].start_sec == 0.0
    assert segments[1][0].start_sec == 60.0