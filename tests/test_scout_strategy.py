from unittest.mock import patch

from cerebrasgemma4.pipeline.chapters import VideoChapter
from cerebrasgemma4.pipeline.frames import Frame, FrameChunk, chunk_frames
from cerebrasgemma4.pipeline.gemma.scout import FrameScore, RegionScore
from cerebrasgemma4.pipeline.gemma.scout_strategy import (
    estimate_hierarchical_scout_calls,
    run_hierarchical_scout,
    should_use_hierarchical_scout,
)
from cerebrasgemma4.pipeline.chapters import ScoutRegion


def _frames(count: int) -> list[Frame]:
    return [
        Frame(frame_id=f"f_{i:04d}", timestamp_sec=float(i), path=__import__("pathlib").Path(f"/tmp/f_{i}.jpg"))
        for i in range(count)
    ]


def _mock_global(batch, _excerpt):
    scores = [
        RegionScore(
            region_id=region.region_id,
            region=region,
            relevance=0.9 if i == 0 else 0.2,
            needs_detail=i == 0,
            brief=region.label,
        )
        for i, (region, _frame) in enumerate(batch)
    ]
    return scores, {"usage": {}, "time_info": {}}


def _mock_scout(chunk, _excerpt):
    scores = [
        FrameScore(
            frame_id=f.frame_id,
            timestamp_sec=f.timestamp_sec,
            relevance=0.8,
            brief="ok",
            has_readable_text=False,
        )
        for f in chunk.frames
    ]
    return scores, {"usage": {}, "time_info": {}}


def _mock_region_mosaic(_region, frames, _excerpt):
    scores = [
        FrameScore(
            frame_id=f.frame_id,
            timestamp_sec=f.timestamp_sec,
            relevance=0.8,
            brief="ok",
            has_readable_text=False,
        )
        for f in frames
    ]
    return scores, {"usage": {}, "time_info": {}}


def test_should_use_hierarchical_for_long_video():
    assert should_use_hierarchical_scout(120, 24) is True
    assert should_use_hierarchical_scout(30, 30) is False


def test_should_use_hierarchical_when_chapters_present():
    assert should_use_hierarchical_scout(30, 5, chapter_count=3) is True


def test_hierarchical_scout_runs_global_then_detail_only():
    frames = _frames(12)
    chunks = chunk_frames(frames, chunk_size=5)
    chapters = [
        VideoChapter("Intro", 0, 6),
        VideoChapter("Demo", 6, 12),
    ]
    calls: list[str] = []

    def after_call(record):
        calls.append(record["extra"]["mode"])

    with (
        patch(
            "cerebrasgemma4.pipeline.gemma.scout_strategy.scout_global_batch",
            side_effect=_mock_global,
        ),
        patch(
            "cerebrasgemma4.pipeline.gemma.scout_strategy.scout_region_mosaic",
            side_effect=_mock_region_mosaic,
        ),
    ):
        scores, meta = run_hierarchical_scout(
            frames=frames,
            chunks=chunks,
            duration_sec=12,
            transcript_segments=[],
            max_frames=4,
            chapters=chapters,
            after_call=after_call,
        )

    assert meta["strategy"] == "hierarchical"
    assert meta["global_calls"] == 1
    assert meta["detail_calls"] <= len(chunks)
    assert "global" in calls
    assert "detail" in calls
    assert scores


def test_short_video_uses_full_scout():
    frames = _frames(3)
    chunks = chunk_frames(frames, chunk_size=5)
    with patch(
        "cerebrasgemma4.pipeline.gemma.scout_strategy.scout_chunk",
        side_effect=_mock_scout,
    ) as mock_chunk:
        _scores, meta = run_hierarchical_scout(
            frames=frames,
            chunks=chunks,
            duration_sec=3,
            transcript_segments=[],
            max_frames=2,
            chapters=None,
        )
    assert meta["strategy"] == "full"
    assert mock_chunk.call_count == len(chunks)


def test_estimate_hierarchical_scout_calls_lower_than_naive():
    import math

    naive = math.ceil(600 / 5)
    smart = estimate_hierarchical_scout_calls(600, 600, 30)
    assert smart < naive