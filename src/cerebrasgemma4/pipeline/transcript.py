"""Transcription: YouTube captions → optional GCP → Whisper fallback."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from cerebrasgemma4.pipeline.ingest import youtube_video_id


@dataclass
class TranscriptSegment:
    start_sec: float
    end_sec: float
    text: str


@dataclass
class TranscriptResult:
    segments: list[TranscriptSegment]
    source: str
    full_text: str


def segments_in_range(
    segments: list[TranscriptSegment], start: float, end: float
) -> str:
    parts = [
        s.text.strip()
        for s in segments
        if s.end_sec >= start and s.start_sec <= end
    ]
    return " ".join(parts).strip()


def fetch_youtube_transcript(url: str, language: str = "auto") -> TranscriptResult | None:
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError:
        return None

    vid = youtube_video_id(url)
    if not vid:
        return None
    try:
        api = YouTubeTranscriptApi()
        if language == "auto":
            fetched = api.fetch(vid)
        else:
            fetched = api.fetch(vid, languages=[language, "en", "fr"])
    except Exception:
        try:
            fetched = api.fetch(vid)
        except Exception:
            return None

    segments = [
        TranscriptSegment(
            start_sec=float(item.start),
            end_sec=float(item.start) + float(item.duration),
            text=item.text,
        )
        for item in fetched
    ]
    full = " ".join(s.text for s in segments)
    return TranscriptResult(segments=segments, source="youtube_captions", full_text=full)


def extract_audio(video_path: Path, audio_path: Path) -> Path | None:
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            str(audio_path),
            "-y",
        ],
        capture_output=True,
    )
    if result.returncode != 0 or not audio_path.exists() or audio_path.stat().st_size == 0:
        return None
    return audio_path


def transcribe_whisper(audio_path: Path, model_name: str | None = None) -> TranscriptResult:
    model_name = model_name or os.getenv("WHISPER_MODEL", "base")
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError("faster-whisper not installed") from exc

    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    whisper_kwargs: dict = {"vad_filter": True}
    if language and language != "auto":
        whisper_kwargs["language"] = language
    segments_iter, _ = model.transcribe(str(audio_path), **whisper_kwargs)
    segments = [
        TranscriptSegment(
            start_sec=float(seg.start),
            end_sec=float(seg.end),
            text=seg.text.strip(),
        )
        for seg in segments_iter
        if seg.text.strip()
    ]
    full = " ".join(s.text for s in segments)
    return TranscriptResult(segments=segments, source="whisper", full_text=full)


def get_transcript(
    *,
    video_path: Path,
    youtube_url: str | None = None,
    language: str = "auto",
) -> TranscriptResult:
    if youtube_url:
        yt = fetch_youtube_transcript(youtube_url, language=language)
        if yt:
            return yt

    audio = video_path.parent / "audio.wav"
    extracted = extract_audio(video_path, audio)
    if extracted is None:
        return TranscriptResult(segments=[], source="none", full_text="")
    return transcribe_whisper(extracted)