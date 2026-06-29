import subprocess
from pathlib import Path

import pytest

from cerebrasgemma4.pipeline.frames import (
    REPORT_MIN_HEIGHT,
    Frame,
    _sparse_timestamps_by_frame_id,
    chunk_frames,
    clear_frame_images,
    ensure_frame_files,
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


def test_sparse_timestamps_dedupe_same_frame_id():
    stamps = _sparse_timestamps_by_frame_id([4088.0, 4088.4, 4088.9, 4090.0])
    assert stamps == [4088.0, 4090.0]


def test_extract_frames_sparse_dedupes_colliding_frame_ids(sample_video: Path, tmp_path: Path):
    frames = extract_frames_sparse(
        sample_video,
        tmp_path / "sparse",
        [0.0, 0.2, 0.8, 1.0],
        max_height=240,
    )
    assert len(frames) == 2
    assert {f.frame_id for f in frames} == {"f_0000", "f_0001"}
    assert all(f.path.exists() for f in frames)


def test_ensure_frame_files_repairs_missing(sample_video: Path, tmp_path: Path):
    out_dir = tmp_path / "frames"
    frames = extract_frames_sparse(sample_video, out_dir, [1.0, 2.0], max_height=240)
    missing = Frame(
        frame_id=frames[0].frame_id,
        timestamp_sec=frames[0].timestamp_sec,
        path=out_dir / "missing.jpg",
    )
    frames[0].path.unlink()
    repaired = ensure_frame_files([missing, frames[1]], sample_video, out_dir, max_height=240)
    assert len(repaired) == 2
    assert all(f.path.exists() for f in repaired)


def test_clear_frame_images_removes_stale_jpegs(tmp_path: Path):
    out_dir = tmp_path / "frames"
    out_dir.mkdir()
    stale = out_dir / "f_0001.jpg"
    stale.write_bytes(b"stale")
    clear_frame_images(out_dir)
    assert not stale.exists()


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