"""Video chapter detection and scout region planning."""

from __future__ import annotations

import math
from dataclasses import dataclass

from cerebrasgemma4.pipeline.ingest import _ytdlp_json, youtube_video_id


@dataclass(frozen=True)
class VideoChapter:
    title: str
    start_sec: float
    end_sec: float


@dataclass(frozen=True)
class ScoutRegion:
    region_id: str
    label: str
    start_sec: float
    end_sec: float
    source: str  # "chapter" | "segment"


def fetch_youtube_chapters(url: str) -> list[VideoChapter]:
    """Return YouTube chapters when yt-dlp exposes them."""
    if not youtube_video_id(url):
        return []
    try:
        data = _ytdlp_json(url)
    except RuntimeError:
        return []

    duration = float(data.get("duration") or 0)
    raw = data.get("chapters") or []
    chapters: list[VideoChapter] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        start = float(item.get("start_time") or 0)
        end = float(item.get("end_time") or start)
        if end <= start:
            if i + 1 < len(raw):
                end = float(raw[i + 1].get("start_time") or start + 1)
            elif duration > start:
                end = duration
            else:
                end = start + 1
        title = str(item.get("title") or f"Chapter {i + 1}").strip()
        chapters.append(VideoChapter(title=title, start_sec=start, end_sec=end))
    return _normalize_chapters(chapters, duration)


def _normalize_chapters(chapters: list[VideoChapter], duration_sec: float) -> list[VideoChapter]:
    if not chapters:
        return []
    capped = max(duration_sec, chapters[-1].end_sec)
    out: list[VideoChapter] = []
    for ch in chapters:
        start = max(0.0, ch.start_sec)
        end = min(capped, max(ch.end_sec, start + 1.0))
        out.append(VideoChapter(title=ch.title, start_sec=start, end_sec=end))
    return out


def plan_scout_regions(
    duration_sec: float,
    chapters: list[VideoChapter] | None = None,
    *,
    segment_sec: float = 60.0,
    max_regions: int = 24,
) -> list[ScoutRegion]:
    """Build coarse regions for the global scout pass."""
    duration_sec = max(1.0, float(duration_sec))
    regions: list[ScoutRegion] = []

    if chapters:
        for i, ch in enumerate(chapters):
            end = min(duration_sec, ch.end_sec)
            start = min(ch.start_sec, end - 0.5)
            regions.append(
                ScoutRegion(
                    region_id=f"ch_{i}",
                    label=ch.title,
                    start_sec=start,
                    end_sec=end,
                    source="chapter",
                )
            )
    else:
        count = min(max_regions, max(1, math.ceil(duration_sec / segment_sec)))
        step = duration_sec / count
        for i in range(count):
            start = i * step
            end = duration_sec if i == count - 1 else (i + 1) * step
            regions.append(
                ScoutRegion(
                    region_id=f"seg_{i}",
                    label=f"Segment {int(start)}s–{int(end)}s",
                    start_sec=start,
                    end_sec=end,
                    source="segment",
                )
            )

    return regions[:max_regions]


def select_regions_for_detail(
    region_scores: list,
    *,
    max_regions: int,
    min_relevance: float = 0.35,
) -> list[ScoutRegion]:
    """Pick regions that merit a fine-grained scout pass."""
    from cerebrasgemma4.pipeline.gemma.scout import RegionScore

    ranked = sorted(region_scores, key=lambda r: r.relevance, reverse=True)
    selected: list[ScoutRegion] = []
    seen: set[str] = set()

    for score in ranked:
        if not isinstance(score, RegionScore):
            continue
        if score.region_id in seen:
            continue
        must_take = len(selected) < min(2, max_regions)
        if must_take or score.needs_detail or score.relevance >= min_relevance:
            selected.append(score.region)
            seen.add(score.region_id)
        if len(selected) >= max_regions:
            break

    if not selected and ranked:
        selected.append(ranked[0].region)
    return selected