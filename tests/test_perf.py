from pathlib import Path

from cerebrasgemma4.pipeline.perf import (
    PerfTracker,
    extract_cerebras_timing,
    normalize_api_call,
    save_metrics_file,
    summarize_cerebras,
)


def test_extract_cerebras_timing():
    timing = extract_cerebras_timing(
        {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        {"time_to_first_token": 0.05, "total_time": 0.8, "tokens_per_second": 1200},
    )
    assert timing["prompt_tokens"] == 100
    assert timing["ttft_ms"] == 50.0
    assert timing["output_tokens_per_sec"] == 1200


def test_summarize_cerebras():
    calls = [
        normalize_api_call(
            stage="scout",
            label="s1",
            wall_sec=0.5,
            usage={"prompt_tokens": 10, "completion_tokens": 5},
            time_info={"time_to_first_token": 0.04},
        ),
        normalize_api_call(
            stage="compose",
            label="c1",
            wall_sec=1.2,
            usage={"prompt_tokens": 20, "completion_tokens": 100},
            time_info={"time_to_first_token": 0.06, "tokens_per_second": 900},
        ),
    ]
    summary = summarize_cerebras(calls)
    assert summary["calls"] == 2
    assert summary["prompt_tokens"] == 30
    assert summary["completion_tokens"] == 105
    assert "scout" in summary["by_stage"]


def test_perf_tracker_steps():
    perf = PerfTracker()
    with perf.step("probe", "Probe", kind="local"):
        pass
    snap = perf.snapshot()
    assert len(snap["steps"]) == 1
    assert snap["steps"][0]["status"] == "done"
    assert snap["steps"][0]["duration_sec"] is not None


def test_merge_prep_includes_download_steps():
    prep = PerfTracker()
    with prep.step("youtube_download", "YouTube video download", kind="local"):
        pass
    prep_snap = prep.snapshot()

    pipeline = PerfTracker()
    pipeline.merge_prep(prep_snap)
    with pipeline.step("probe", "Video probe (ffprobe)", kind="local"):
        pass
    snap = pipeline.snapshot()

    assert snap["prep_elapsed_sec"] >= 0
    assert snap["ingest_sec"] >= 0
    assert snap["elapsed_sec"] >= snap["pipeline_elapsed_sec"]
    assert snap["steps"][0]["id"] == "youtube_download"


def test_save_metrics_file(tmp_path: Path):
    path = tmp_path / "metrics.json"
    payload = {"elapsed_sec": 12.3, "steps": []}
    save_metrics_file(path, payload)
    assert path.exists()
    assert "12.3" in path.read_text(encoding="utf-8")