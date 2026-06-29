import pytest

from cerebrasgemma4.pipeline.options import (
    HACKATHON_RPM,
    estimate_api_calls,
    suggest_convert_options,
)


@pytest.fixture(autouse=True)
def _disable_demo_mode(monkeypatch):
    monkeypatch.setenv("SIGHTLINE_DEMO_MODE", "0")


def test_suggest_for_120s_video():
    s = suggest_convert_options(119.5, width=1920, height=1080, fps=30)
    assert s.max_duration_sec == 120
    assert s.max_frames == 8
    assert s.estimated_scout_calls == 3
    assert s.estimated_total_api_calls == 12
    assert s.hackathon_capped is False
    assert s.estimated_pipeline_minutes >= 0
    assert s.estimated_total_minutes >= s.estimated_pipeline_minutes


def test_suggest_processes_full_long_video():
    s = suggest_convert_options(600, width=1920, height=1080, fps=30)
    assert s.max_duration_sec == 600
    assert s.max_frames == 30
    assert s.estimated_scout_calls == 10
    assert s.estimated_total_api_calls == 41
    assert s.hackathon_capped is False
    assert s.estimated_pipeline_minutes < 2.0
    assert s.estimated_total_minutes < 3.0
    assert s.estimated_total_minutes > s.estimated_pipeline_minutes


def test_estimate_api_calls():
    est = estimate_api_calls(60, 60, 8)
    assert est["scout_calls"] == 2
    assert est["total_api_calls"] == 11
    assert est["estimated_total_minutes"] < 2.0
    assert est["hackathon_rpm"] == HACKATHON_RPM


def test_estimate_long_youtube_video_realistic():
    s = suggest_convert_options(847, chapter_count=11, youtube=True)
    assert s.estimated_scout_calls == 11
    assert s.estimated_total_minutes < 3.0


def test_demo_mode_targets_sub_minute(monkeypatch):
    monkeypatch.setenv("SIGHTLINE_DEMO_MODE", "1")
    s = suggest_convert_options(847, chapter_count=11, youtube=True)
    assert s.max_frames == 6
    assert s.estimated_total_minutes < 1.0