"""Hierarchical scout: global overview first, detail only where needed."""

from __future__ import annotations

import math
import time
from typing import Any, Callable

from cerebrasgemma4.pipeline.chapters import (
    ScoutRegion,
    VideoChapter,
    plan_scout_regions,
    select_regions_for_detail,
)
from cerebrasgemma4.pipeline.demo import (
    DEMO_MAX_DETAIL_REGIONS,
    DEMO_REGION_MOSAIC_MAX,
    is_demo_mode,
)
from cerebrasgemma4.pipeline.frames import Frame, FrameChunk
from cerebrasgemma4.images import MOSAIC_MAX_CELLS
from cerebrasgemma4.pipeline.gemma.scout import (
    FrameScore,
    RegionScore,
    scout_chunk,
    scout_global_batch,
    scout_region_mosaic,
)
from cerebrasgemma4.pipeline.transcript import segments_in_range

GLOBAL_BATCH_SIZE = 5
SHORT_VIDEO_SEC = 45.0
MAX_DETAIL_REGIONS_CAP = 8
REGION_MOSAIC_MAX_FRAMES = MOSAIC_MAX_CELLS


def _frame_near_timestamp(frames: list[Frame], timestamp_sec: float) -> Frame | None:
    if not frames:
        return None
    return min(frames, key=lambda f: abs(f.timestamp_sec - timestamp_sec))


def _region_midpoint(region: ScoutRegion) -> float:
    return (region.start_sec + region.end_sec) / 2


def _frames_in_region(frames: list[Frame], region: ScoutRegion) -> list[Frame]:
    return [
        f
        for f in frames
        if region.start_sec <= f.timestamp_sec <= region.end_sec + 0.5
    ]


def _sample_frames(frames: list[Frame], max_count: int) -> list[Frame]:
    if len(frames) <= max_count:
        return frames
    if max_count <= 1:
        return [frames[0]]
    step = (len(frames) - 1) / (max_count - 1)
    return [frames[round(i * step)] for i in range(max_count)]


def _detail_region_budget(max_frames: int) -> int:
    cap = DEMO_MAX_DETAIL_REGIONS if is_demo_mode() else MAX_DETAIL_REGIONS_CAP
    return min(cap, max(2, math.ceil(max_frames / 2)))


def _region_mosaic_max_frames() -> int:
    return DEMO_REGION_MOSAIC_MAX if is_demo_mode() else REGION_MOSAIC_MAX_FRAMES


def should_use_hierarchical_scout(
    duration_sec: float,
    chunk_count: int,
    *,
    chapter_count: int = 0,
) -> bool:
    if chapter_count > 0 and chunk_count > 2:
        return True
    if duration_sec <= SHORT_VIDEO_SEC or chunk_count <= 2:
        return False
    return chunk_count > 3


def run_hierarchical_scout(
    *,
    frames: list[Frame],
    chunks: list[FrameChunk],
    duration_sec: float,
    transcript_segments: list,
    max_frames: int,
    chapters: list[VideoChapter] | None,
    on_global_batch: Callable[[int, int], None] | None = None,
    on_detail_chunk: Callable[[int, int, str], None] | None = None,
    before_call: Callable[[], None] | None = None,
    after_call: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[list[FrameScore], dict]:
    """
    Run global scout then detail scout in selected regions only.

    Returns frame scores and strategy metadata for metrics.
    """
    if not should_use_hierarchical_scout(
        duration_sec, len(chunks), chapter_count=len(chapters or [])
    ):
        all_scores: list[FrameScore] = []
        for i, chunk in enumerate(chunks):
            if on_detail_chunk:
                on_detail_chunk(i + 1, len(chunks), "full")
            excerpt = segments_in_range(
                transcript_segments, chunk.start_sec, chunk.end_sec + 1
            )
            if before_call:
                before_call()
            t0 = time.perf_counter()
            scores, raw = scout_chunk(chunk, excerpt)
            record = {
                "stage": "scout",
                "label": f"Scout mosaic {i + 1}/{len(chunks)}",
                "wall_sec": time.perf_counter() - t0,
                "usage": raw.get("usage"),
                "time_info": raw.get("time_info"),
                "extra": {
                    "chunk": i,
                    "frames": len(chunk.frames),
                    "mode": "full",
                    "mosaic": True,
                },
            }
            if after_call:
                after_call(record)
            all_scores.extend(scores)
        return all_scores, {
            "strategy": "full",
            "global_calls": 0,
            "detail_calls": len(chunks),
            "regions_global": 0,
            "regions_detail": 0,
            "chapter_count": len(chapters or []),
            "mosaic_scout": True,
        }

    regions = plan_scout_regions(duration_sec, chapters)
    representatives: list[tuple[ScoutRegion, Frame]] = []
    for region in regions:
        frame = _frame_near_timestamp(frames, _region_midpoint(region))
        if frame is not None:
            representatives.append((region, frame))

    global_batches = [
        representatives[i : i + GLOBAL_BATCH_SIZE]
        for i in range(0, len(representatives), GLOBAL_BATCH_SIZE)
    ]
    region_scores: list[RegionScore] = []
    for i, batch in enumerate(global_batches):
        if on_global_batch:
            on_global_batch(i + 1, len(global_batches))
        start = batch[0][0].start_sec
        end = batch[-1][0].end_sec
        excerpt = segments_in_range(transcript_segments, start, end + 1)
        if before_call:
            before_call()
        t0 = time.perf_counter()
        scores, raw = scout_global_batch(batch, excerpt)
        record = {
            "stage": "scout",
            "label": f"Global scout {i + 1}/{len(global_batches)}",
            "wall_sec": time.perf_counter() - t0,
            "usage": raw.get("usage"),
            "time_info": raw.get("time_info"),
            "extra": {"mode": "global", "regions": len(batch)},
        }
        if after_call:
            after_call(record)
        region_scores.extend(scores)

    detail_regions = select_regions_for_detail(
        region_scores,
        max_regions=_detail_region_budget(max_frames),
    )
    detail_mosaics: list[tuple[ScoutRegion, list[Frame]]] = []
    for region in detail_regions:
        region_frames = _sample_frames(
            _frames_in_region(frames, region),
            _region_mosaic_max_frames(),
        )
        if region_frames:
            detail_mosaics.append((region, region_frames))

    all_scores = []
    for i, (region, region_frames) in enumerate(detail_mosaics):
        if on_detail_chunk:
            on_detail_chunk(i + 1, len(detail_mosaics), "detail")
        excerpt = segments_in_range(
            transcript_segments, region.start_sec, region.end_sec + 1
        )
        if before_call:
            before_call()
        t0 = time.perf_counter()
        scores, raw = scout_region_mosaic(region, region_frames, excerpt)
        record = {
            "stage": "scout",
            "label": f"Detail mosaic {i + 1}/{len(detail_mosaics)}",
            "wall_sec": time.perf_counter() - t0,
            "usage": raw.get("usage"),
            "time_info": raw.get("time_info"),
            "extra": {
                "mode": "detail",
                "mosaic": True,
                "region": region.region_id,
                "frames": len(region_frames),
            },
        }
        if after_call:
            after_call(record)
        all_scores.extend(scores)

    return all_scores, {
        "strategy": "hierarchical",
        "global_calls": len(global_batches),
        "detail_calls": len(detail_mosaics),
        "regions_global": len(regions),
        "regions_detail": len(detail_regions),
        "chapter_count": len(chapters or []),
        "region_labels": [r.label for r in detail_regions],
        "mosaic_scout": True,
    }


def estimate_hierarchical_scout_calls(
    duration_sec: float,
    max_duration_sec: float,
    max_frames: int,
    *,
    chapter_count: int = 0,
) -> int:
    effective = min(max(1.0, duration_sec), max_duration_sec)
    if effective <= SHORT_VIDEO_SEC:
        return math.ceil(effective / 5)

    if chapter_count > 0:
        num_regions = min(24, chapter_count)
    else:
        num_regions = min(24, max(1, math.ceil(effective / 60)))
    global_calls = math.ceil(num_regions / GLOBAL_BATCH_SIZE)
    detail_regions = min(_detail_region_budget(max_frames), num_regions)
    detail_calls = detail_regions
    return global_calls + detail_calls