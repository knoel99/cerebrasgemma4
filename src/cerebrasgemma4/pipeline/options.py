"""Suggested conversion options from video duration."""

from __future__ import annotations

import math
from dataclasses import dataclass

from cerebrasgemma4.pipeline.rate_limit import (
    HACKATHON_RPM,
    HACKATHON_TPM,
    estimate_pipeline_minutes,
)

MAX_FRAMES_CAP = 30
MIN_FRAMES = 4
COMPOSE_CALLS = 1


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
    hackathon_rpm: int = HACKATHON_RPM
    hackathon_tpm: int = HACKATHON_TPM
    width: int = 0
    height: int = 0
    fps: float = 0.0


def _full_video_duration(duration_sec: float) -> float:
    return max(1.0, math.ceil(duration_sec))


def _ideal_max_frames(duration_sec: float) -> int:
    return min(MAX_FRAMES_CAP, max(MIN_FRAMES, round(duration_sec / 15)))


def _total_api_calls(
    duration_sec: float, max_duration_sec: float, max_frames: int
) -> tuple[int, int, int]:
    effective = min(max(1.0, duration_sec), max_duration_sec)
    scout = math.ceil(effective / 5)  # 1 fps, batches of 5 frames
    analyze = max(1, min(max_frames, MAX_FRAMES_CAP))
    total = scout + analyze + COMPOSE_CALLS
    return scout, analyze, total


def suggest_convert_options(
    duration_sec: float,
    *,
    width: int = 0,
    height: int = 0,
    fps: float = 0.0,
) -> ConvertSuggestions:
    duration_sec = max(1.0, float(duration_sec))
    max_duration = _full_video_duration(duration_sec)
    max_frames = _ideal_max_frames(duration_sec)
    scout, analyze, total = _total_api_calls(duration_sec, max_duration, max_frames)
    pipeline_min = estimate_pipeline_minutes(total)

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
        width=width,
        height=height,
        fps=round(fps, 2) if fps else 0.0,
    )


def estimate_api_calls(duration_sec: float, max_duration_sec: float, max_frames: int) -> dict:
    scout, analyze, total = _total_api_calls(duration_sec, max_duration_sec, max_frames)
    return {
        "scout_calls": scout,
        "analyze_calls": analyze,
        "compose_calls": COMPOSE_CALLS,
        "total_api_calls": total,
        "within_hackathon_rpm": total <= HACKATHON_RPM,
        "estimated_pipeline_minutes": round(estimate_pipeline_minutes(total), 1),
        "hackathon_rpm": HACKATHON_RPM,
        "hackathon_tpm": HACKATHON_TPM,
    }