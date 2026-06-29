import time
from unittest.mock import patch

from cerebrasgemma4.pipeline.rate_limit import (
    CerebrasRateLimiter,
    estimate_pipeline_minutes,
)


def test_rate_limiter_waits_when_rpm_cap_reached():
    limiter = CerebrasRateLimiter(rpm_limit=2, tpm_limit=1_000_000, rpm_headroom=1.0)
    clock = iter([10.0, 10.0, 10.0, 70.0])
    with patch("cerebrasgemma4.pipeline.rate_limit.time.monotonic", side_effect=lambda: next(clock)):
        with patch("cerebrasgemma4.pipeline.rate_limit.time.sleep") as sleep:
            limiter.record({"total_tokens": 100})
            limiter.record({"total_tokens": 100})
            limiter.wait_for_slot(token_estimate=100)
    sleep.assert_called_once()


def test_rate_limiter_records_tokens_and_updates_estimate():
    limiter = CerebrasRateLimiter(rpm_limit=100, tpm_limit=100_000)
    limiter.record({"prompt_tokens": 4000, "completion_tokens": 1000})
    snap = limiter.snapshot()
    assert snap["tokens_in_window"] == 5000
    assert limiter.default_token_estimate >= 5000


def test_estimate_pipeline_minutes_scales_with_calls():
    short = estimate_pipeline_minutes(20)
    long = estimate_pipeline_minutes(151)
    assert long > short