from pathlib import Path

from cerebrasgemma4.pipeline.context import VideoContext, load_context, save_context
from cerebrasgemma4.pipeline.gemma.analyze import FrameAnalysis
from cerebrasgemma4.pipeline.transcript import TranscriptResult, TranscriptSegment


def test_context_round_trip(tmp_path: Path):
    transcript = TranscriptResult(
        segments=[TranscriptSegment(0.0, 2.0, "Hello")],
        source="mock",
        full_text="Hello world",
    )
    analyses = [
        FrameAnalysis(
            frame_id="f_0001",
            timestamp_sec=1.0,
            title="Intro",
            body="Speaker on stage.",
            quoted_text=["HELLO"],
            asset_name="frame_0001.jpg",
        )
    ]
    ctx = VideoContext.from_pipeline(
        source_name="demo.mp4",
        duration_sec=120.0,
        transcript=transcript,
        analyses=analyses,
    )
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    save_context(job_dir, ctx)
    loaded = load_context(job_dir)
    assert loaded is not None
    assert loaded.source_name == "demo.mp4"
    assert loaded.transcript_full_text == "Hello world"
    assert len(loaded.analyses) == 1
    assert loaded.analyses[0]["title"] == "Intro"