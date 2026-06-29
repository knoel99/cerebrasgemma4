from pathlib import Path
from unittest.mock import patch

from cerebrasgemma4.pipeline.frames import Frame, load_scout_frames_from_dir
from cerebrasgemma4.pipeline.gemma.compose import ComposeInput, build_compose_prompt
from cerebrasgemma4.pipeline.gemma.series import (
    DataObservation,
    MetricValue,
    collect_observations,
    merge_observations,
    prompt_requests_charts,
    sample_frames,
)
from cerebrasgemma4.pipeline.transcript import TranscriptResult


def test_prompt_requests_charts_detects_chart_request():
    assert prompt_requests_charts("Montre l'évolution avec des graphiques")
    assert prompt_requests_charts("Extract metrics from slides and plot trends")
    assert not prompt_requests_charts("Focus on the intro")


def test_sample_frames_evenly_samples():
    frames = [
        Frame(frame_id=f"f_{i:04d}", timestamp_sec=float(i), path=Path(f"/tmp/f_{i:04d}.jpg"))
        for i in range(10)
    ]
    with patch.object(Path, "is_file", return_value=True):
        picked = sample_frames(frames, max_count=5)
    assert len(picked) == 5
    assert picked[0].frame_id == "f_0000"
    assert picked[-1].frame_id == "f_0009"


def test_merge_observations_prefers_richer_values():
    merged = merge_observations(
        [
            DataObservation(1.0, "image", metrics=[], frame_id="f_0001"),
            DataObservation(
                1.0,
                "image",
                metrics=[MetricValue("price", "Price", 42.0, "USD")],
                frame_id="f_0001",
            ),
        ]
    )
    assert len(merged) == 1
    assert merged[0].metrics[0].value == 42.0


def test_collect_observations_merges_frames_and_transcript():
    frames = [
        Frame(frame_id=f"f_{i:04d}", timestamp_sec=float(i), path=Path(f"/tmp/f_{i:04d}.jpg"))
        for i in range(3)
    ]

    def fake_frames(batch, *, custom_prompt, time_range):
        return [
            DataObservation(
                f.timestamp_sec,
                "image",
                metrics=[MetricValue("count", "Count", float(i), "")],
                frame_id=f.frame_id,
            )
            for i, f in enumerate(batch)
        ], {"usage": {}, "time_info": {}}

    text_obs = [
        DataObservation(
            10.0,
            "transcript",
            metrics=[MetricValue("revenue", "Revenue", 1_000_000.0, "USD")],
            notes="we reached one million",
        )
    ]

    with (
        patch.object(Path, "is_file", return_value=True),
        patch(
            "cerebrasgemma4.pipeline.gemma.series.read_frame_batch",
            side_effect=fake_frames,
        ),
        patch(
            "cerebrasgemma4.pipeline.gemma.series.read_transcript",
            return_value=(text_obs, {"usage": {}, "time_info": {}}),
        ),
    ):
        observations = collect_observations(
            frames=frames,
            transcript_segments=[{"start_sec": 10, "end_sec": 12, "text": "one million"}],
            custom_prompt="charts please",
        )
    assert len(observations) == 4
    assert any(o.source == "transcript" for o in observations)


def test_build_compose_prompt_includes_charts_section():
    prompt = build_compose_prompt(
        ComposeInput(
            source_name="demo.mp4",
            duration_sec=600,
            transcript=TranscriptResult(segments=[], source="mock", full_text=""),
            analyses=[],
            charts_section="| Time | Source | Count |\n| --- | --- | ---: |",
        )
    )
    assert "Observations and charts" in prompt
    assert "| Time | Source | Count |" in prompt


def test_load_scout_frames_from_dir(tmp_path: Path):
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    (frames_dir / "f_0042.jpg").write_bytes(b"jpg")
    loaded = load_scout_frames_from_dir(frames_dir)
    assert len(loaded) == 1
    assert loaded[0].frame_id == "f_0042"
    assert loaded[0].timestamp_sec == 42.0


def test_legacy_observation_dict_migration():
    obs = DataObservation.from_dict(
        {
            "frame_id": "f_0001",
            "timestamp_sec": 1.0,
            "widgets_sold": 42.0,
            "margin_pct": 12.5,
        }
    )
    keys = {m.key for m in obs.metrics}
    assert "widgets_sold" in keys
    assert "margin_pct" in keys