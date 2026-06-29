"""Video ingestion and metadata extraction."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import httpx


@dataclass
class VideoMetadata:
    duration_sec: float
    width: int
    height: int
    fps: float
    source_path: Path


@dataclass
class YoutubeMetadata:
    video_id: str
    title: str
    thumbnail_url: str


def probe_video(path: Path) -> VideoMetadata:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(result.stdout)
    video_stream = next(
        (s for s in data.get("streams", []) if s.get("codec_type") == "video"),
        {},
    )
    duration = float(data.get("format", {}).get("duration", 0) or 0)
    fps_parts = (video_stream.get("r_frame_rate") or "0/1").split("/")
    fps = float(fps_parts[0]) / float(fps_parts[1] or 1)
    return VideoMetadata(
        duration_sec=duration,
        width=int(video_stream.get("width") or 0),
        height=int(video_stream.get("height") or 0),
        fps=fps,
        source_path=path,
    )


def save_upload(dest: Path, data: bytes) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return dest


def youtube_thumbnail_url(video_id: str, *, quality: str = "hqdefault") -> str:
    return f"https://i.ytimg.com/vi/{video_id}/{quality}.jpg"


def fetch_youtube_metadata(url: str) -> YoutubeMetadata | None:
    """Resolve YouTube title and thumbnail without downloading the video."""
    video_id = youtube_video_id(url)
    if not video_id:
        return None

    title: str | None = None
    thumbnail_url = youtube_thumbnail_url(video_id)

    ytdlp = shutil.which("yt-dlp")
    if ytdlp:
        try:
            result = subprocess.run(
                [ytdlp, "--dump-json", "--skip-download", url],
                capture_output=True,
                text=True,
                check=True,
            )
            data = json.loads(result.stdout)
            title = data.get("title") or title
            thumbnail_url = data.get("thumbnail") or thumbnail_url
        except (subprocess.CalledProcessError, json.JSONDecodeError, OSError):
            pass

    if not title:
        try:
            resp = httpx.get(
                "https://www.youtube.com/oembed",
                params={"url": url, "format": "json"},
                timeout=10.0,
            )
            if resp.status_code == 200:
                title = resp.json().get("title")
        except httpx.HTTPError:
            pass

    return YoutubeMetadata(
        video_id=video_id,
        title=title or f"YouTube video {video_id}",
        thumbnail_url=thumbnail_url,
    )


def download_url_to_file(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with httpx.Client(follow_redirects=True, timeout=30.0) as client:
        resp = client.get(url)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
    return dest


def youtube_video_id(url: str) -> str | None:
    parsed = urlparse(url)
    if parsed.hostname in {"youtu.be", "www.youtu.be"}:
        return parsed.path.lstrip("/").split("/")[0] or None
    if parsed.hostname in {"youtube.com", "www.youtube.com", "m.youtube.com"}:
        if parsed.path == "/watch":
            return parse_qs(parsed.query).get("v", [None])[0]
        if parsed.path.startswith("/shorts/"):
            return parsed.path.split("/")[2]
    return None


def _ytdlp_path() -> str:
    ytdlp = shutil.which("yt-dlp")
    if not ytdlp:
        raise RuntimeError("yt-dlp required for YouTube URLs (pip install yt-dlp)")
    return ytdlp


def _ytdlp_json(url: str) -> dict:
    ytdlp = _ytdlp_path()
    try:
        result = subprocess.run(
            [ytdlp, "--dump-json", "--skip-download", "--no-playlist", url],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or str(exc)).strip()
        raise RuntimeError(f"YouTube metadata failed: {detail[-500:]}") from exc
    return json.loads(result.stdout)


def probe_youtube(url: str) -> VideoMetadata:
    """Read YouTube duration/resolution without downloading the video."""
    data = _ytdlp_json(url)
    fps_raw = data.get("fps") or 0
    try:
        fps = float(fps_raw)
    except (TypeError, ValueError):
        fps = 0.0
    return VideoMetadata(
        duration_sec=float(data.get("duration") or 0),
        width=int(data.get("width") or 0),
        height=int(data.get("height") or 0),
        fps=fps,
        source_path=Path(url),
    )


YOUTUBE_FORMAT_PROCESSING = (
    "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/"
    "bestvideo[height<=720]+bestaudio/"
    "best[height<=720]/best"
)

YOUTUBE_FORMAT_HD = YOUTUBE_FORMAT_PROCESSING


def _download_youtube_with_format(url: str, dest: Path, fmt: str) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    ytdlp = _ytdlp_path()
    output = str(dest.with_suffix(".%(ext)s") if dest.suffix else dest)
    last_error = "unknown error"
    for candidate in (fmt, "mp4/best", "best[ext=mp4]/best", "best"):
        try:
            subprocess.run(
                [ytdlp, "--no-playlist", "-f", candidate, "-o", output, url],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            last_error = (exc.stderr or exc.stdout or str(exc)).strip()[-500:]
            continue
        if dest.exists():
            return dest
        candidates = sorted(dest.parent.glob(dest.stem + ".*"))
        if candidates:
            return candidates[0]
    raise RuntimeError(f"YouTube download failed: {last_error}")


def download_youtube_processing(url: str, dest: Path) -> Path:
    """Fast 720p download for pipeline processing."""
    return _download_youtube_with_format(url, dest, YOUTUBE_FORMAT_PROCESSING)


def download_youtube_hd(url: str, dest: Path) -> Path:
    """720p download for report screenshots (same cap as processing)."""
    return _download_youtube_with_format(url, dest, YOUTUBE_FORMAT_HD)


def download_youtube(url: str, dest: Path) -> Path:
    """Download YouTube video with yt-dlp if available, else raise."""
    return download_youtube_hd(url, dest)


def download_http_video(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with httpx.stream("GET", url, follow_redirects=True, timeout=120.0) as resp:
        resp.raise_for_status()
        with dest.open("wb") as f:
            for chunk in resp.iter_bytes():
                f.write(chunk)
    return dest