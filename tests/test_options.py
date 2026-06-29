from cerebrasgemma4.pipeline.options import (
    HACKATHON_RPM,
    estimate_api_calls,
    suggest_convert_options,
)


def test_suggest_for_120s_video():
    s = suggest_convert_options(119.5, width=1920, height=1080, fps=30)
    assert s.max_duration_sec == 120
    assert s.max_frames == 8
    assert s.estimated_scout_calls == 24
    assert s.estimated_total_api_calls == 33
    assert s.hackathon_capped is False
    assert s.estimated_pipeline_minutes >= 0


def test_suggest_processes_full_long_video():
    s = suggest_convert_options(600, width=1920, height=1080, fps=30)
    assert s.max_duration_sec == 600
    assert s.max_frames == 30
    assert s.estimated_scout_calls == 120
    assert s.estimated_total_api_calls == 151
    assert s.hackathon_capped is False
    assert s.estimated_pipeline_minutes > 1.0


def test_estimate_api_calls():
    est = estimate_api_calls(60, 60, 8)
    assert est["scout_calls"] == 12
    assert est["total_api_calls"] == 21
    assert est["hackathon_rpm"] == HACKATHON_RPM