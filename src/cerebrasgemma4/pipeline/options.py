"""Suggested conversion options from video duration."""

from __future__ import annotations

import math
from dataclasses import dataclass

from cerebrasgemma4.pipeline.demo import (
    DEMO_MAX_FRAMES,
    is_demo_mode,
)
from cerebrasgemma4.pipeline.gemma.scout_strategy import estimate_hierarchical_scout_calls
from cerebrasgemma4.pipeline.rate_limit import HACKATHON_RPM, HACKATHON_TPM

MAX_FRAMES_CAP = 30
MIN_FRAMES = 4
COMPOSE_CALLS = 1

# Empirical wall-clock per call after mosaic scout (hackathon runs on Cerebras).
SCOUT_WALL_SEC = 0.9
ANALYZE_WALL_SEC = 0.55
COMPOSE_WALL_SEC = 1.0
SCOUT_TOKENS = 2100
ANALYZE_TOKENS = 650
COMPOSE_TOKENS = 6400
RATE_LIMIT_BUFFER = 1.1


@dataclass
class ConvertSuggestions:
    duration_sec: float
    max_duration_sec: float
    max_frames: int
    estimated_scout_calls: int
    estimated_analyze_calls: int
    estimated_total_api_calls: int
    within_hackathon_rpm: bool
    hackathon_capped: bool = False
    estimated_pipeline_minutes: float = 0.0
    estimated_total_minutes: float = 0.0
    hackathon_rpm: int = HACKATHON_RPM
    hackathon_tpm: int = HACKATHON_TPM
    width: int = 0
    height: int = 0
    fps: float = 0.0


def _full_video_duration(duration_sec: float) -> float:
    return max(1.0, math.ceil(duration_sec))


def _ideal_max_frames(duration_sec: float) -> int:
    return min(MAX_FRAMES_CAP, max(MIN_FRAMES, round(duration_sec / 15)))


def estimate_local_prep_sec(
    duration_sec: float,
    *,
    youtube: bool = False,
) -> float:
    """Wall time for ingest, frame extraction, and transcription."""
    if is_demo_mode():
        ingest = 10.0 if youtube else 3.0
        frames = 6.0
        transcript = 2.0
        return ingest + frames + transcript

    d = min(max(1.0, duration_sec), 7200.0)
    ingest = 6.0 + (8.0 if youtube else 2.0) + min(d / 180.0, 20.0)
    frames = max(3.0, d * 0.012)
    transcript = 2.5
    return ingest + frames + transcript


def estimate_local_prep_minutes(duration_sec: float, *, youtube: bool = False) -> float:
    return estimate_local_prep_sec(duration_sec, youtube=youtube) / 60.0


def estimate_pipeline_sec(
    scout_calls: int,
    analyze_calls: int,
    *,
    compose_calls: int = COMPOSE_CALLS,
) -> float:
    """Estimate Cerebras LLM wall time (compute + light rate-limit buffer)."""
    compute = (
        scout_calls * SCOUT_WALL_SEC
        + analyze_calls * ANALYZE_WALL_SEC
        + compose_calls * COMPOSE_WALL_SEC
    )
    total_calls = scout_calls + analyze_calls + compose_calls
    total_tokens = (
        scout_calls * SCOUT_TOKENS
        + analyze_calls * ANALYZE_TOKENS
        + compose_calls * COMPOSE_TOKENS
    )
    rpm_cap = max(1, int(HACKATHON_RPM * 0.92))
    tpm_cap = max(1, int(HACKATHON_TPM * 0.92))

    rpm_floor_sec = (total_calls / rpm_cap) * 60.0
    # Worst-minute token burst if calls cluster at the RPM cap.
    peak_minute_tokens = min(
        total_tokens,
        rpm_cap * max(SCOUT_TOKENS, ANALYZE_TOKENS, COMPOSE_TOKENS),
    )
    tpm_delay_sec = 0.0
    if peak_minute_tokens > tpm_cap:
        tpm_delay_sec = ((peak_minute_tokens - tpm_cap) / tpm_cap) * 60.0

    return (max(compute, rpm_floor_sec) + tpm_delay_sec) * RATE_LIMIT_BUFFER


def estimate_pipeline_minutes(
    scout_calls: int,
    analyze_calls: int,
    *,
    compose_calls: int = COMPOSE_CALLS,
) -> float:
    return estimate_pipeline_sec(scout_calls, analyze_calls, compose_calls=compose_calls) / 60.0


def estimate_total_minutes(
    duration_sec: float,
    scout_calls: int,
    analyze_calls: int,
    *,
    youtube: bool = False,
) -> float:
    """Estimate end-to-end report generation time."""
    return estimate_local_prep_minutes(duration_sec, youtube=youtube) + estimate_pipeline_minutes(
        scout_calls, analyze_calls
    )


def _total_api_calls(
    duration_sec: float,
    max_duration_sec: float,
    max_frames: int,
    *,
    chapter_count: int = 0,
) -> tuple[int, int, int]:
    scout = estimate_hierarchical_scout_calls(
        duration_sec,
        max_duration_sec,
        max_frames,
        chapter_count=chapter_count,
    )
    analyze = max(1, min(max_frames, MAX_FRAMES_CAP))
    total = scout + analyze + COMPOSE_CALLS
    return scout, analyze, total


def suggest_convert_options(
    duration_sec: float,
    *,
    width: int = 0,
    height: int = 0,
    fps: float = 0.0,
    chapter_count: int = 0,
    youtube: bool = False,
) -> ConvertSuggestions:
    duration_sec = max(1.0, float(duration_sec))
    max_duration = _full_video_duration(duration_sec)
    max_frames = DEMO_MAX_FRAMES if is_demo_mode() else _ideal_max_frames(duration_sec)
    scout, analyze, total = _total_api_calls(
        duration_sec,
        max_duration,
        max_frames,
        chapter_count=chapter_count,
    )
    pipeline_min = estimate_pipeline_minutes(scout, analyze)
    total_min = estimate_total_minutes(
        duration_sec,
        scout,
        analyze,
        youtube=youtube,
    )

    return ConvertSuggestions(
        duration_sec=round(duration_sec, 2),
        max_duration_sec=float(max_duration),
        max_frames=int(max_frames),
        estimated_scout_calls=scout,
        estimated_analyze_calls=analyze,
        estimated_total_api_calls=total,
        within_hackathon_rpm=total <= HACKATHON_RPM,
        hackathon_capped=False,
        estimated_pipeline_minutes=round(pipeline_min, 1),
        estimated_total_minutes=round(total_min, 1),
        width=width,
        height=height,
        fps=round(fps, 2) if fps else 0.0,
    )


def estimate_api_calls(
    duration_sec: float,
    max_duration_sec: float,
    max_frames: int,
    *,
    chapter_count: int = 0,
    youtube: bool = False,
) -> dict:
    scout, analyze, total = _total_api_calls(
        duration_sec,
        max_duration_sec,
        max_frames,
        chapter_count=chapter_count,
    )
    return {
        "scout_calls": scout,
        "analyze_calls": analyze,
        "compose_calls": COMPOSE_CALLS,
        "total_api_calls": total,
        "within_hackathon_rpm": total <= HACKATHON_RPM,
        "estimated_pipeline_minutes": round(estimate_pipeline_minutes(scout, analyze), 1),
        "estimated_total_minutes": round(
            estimate_total_minutes(
                duration_sec,
                scout,
                analyze,
                youtube=youtube,
            ),
            1,
        ),
        "hackathon_rpm": HACKATHON_RPM,
        "hackathon_tpm": HACKATHON_TPM,
    }