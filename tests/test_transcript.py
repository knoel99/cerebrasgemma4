import subprocess
from pathlib import Path

from cerebrasgemma4.pipeline.transcript import extract_audio, get_transcript


def test_no_audio_stream_returns_empty(tmp_path: Path):
    video = tmp_path / "silent.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=1:size=160x120:rate=5",
            "-c:v",
            "libx264",
            "-an",
            str(video),
            "-y",
        ],
        check=True,
    )
    assert extract_audio(video, tmp_path / "audio.wav") is None
    result = get_transcript(video_path=video)
    assert result.source == "none"
    assert result.full_text == ""