from pathlib import Path
from unittest.mock import MagicMock, patch

from cerebrasgemma4.pipeline.ingest import (
    fetch_youtube_metadata,
    probe_youtube,
    youtube_thumbnail_url,
    youtube_video_id,
)


def test_youtube_video_id_parses_watch_url():
    assert youtube_video_id("https://www.youtube.com/watch?v=sYhIjzs3Lwc") == "sYhIjzs3Lwc"


def test_youtube_thumbnail_url():
    assert "sYhIjzs3Lwc" in youtube_thumbnail_url("sYhIjzs3Lwc")


def test_fetch_youtube_metadata_oembed_fallback():
    url = "https://www.youtube.com/watch?v=sYhIjzs3Lwc"
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"title": "Demo Video Title"}

    with patch("cerebrasgemma4.pipeline.ingest.shutil.which", return_value=None):
        with patch("cerebrasgemma4.pipeline.ingest.httpx.get", return_value=mock_resp):
            meta = fetch_youtube_metadata(url)

    assert meta is not None
    assert meta.video_id == "sYhIjzs3Lwc"
    assert meta.title == "Demo Video Title"
    assert "ytimg.com" in meta.thumbnail_url


def test_probe_youtube_from_json_dump():
    url = "https://www.youtube.com/watch?v=sYhIjzs3Lwc"
    payload = {
        "duration": 2745,
        "width": 1920,
        "height": 1080,
        "fps": 50,
        "title": "Demo",
    }

    with patch("cerebrasgemma4.pipeline.ingest._ytdlp_json", return_value=payload):
        meta = probe_youtube(url)

    assert meta.duration_sec == 2745
    assert meta.width == 1920
    assert meta.height == 1080
    assert meta.fps == 50


def test_job_summary_includes_media_urls(tmp_path: Path):
    from fastapi.testclient import TestClient

    from cerebrasgemma4.api.main import app
    from cerebrasgemma4.pipeline.jobs import JobStatus, JobStore

    store = JobStore(root=tmp_path / "jobs")
    record = store.create()
    assets = store.path(record.job_id) / "assets"
    (assets / "youtube_thumbnail.jpg").write_bytes(b"thumb")
    (assets / "preview.jpg").write_bytes(b"preview")
    store.update(
        record.job_id,
        status=JobStatus.COMPLETED,
        title="My YouTube Video",
        source_type="youtube",
        youtube_video_id="abc123",
        thumbnail_asset="youtube_thumbnail.jpg",
        preview_asset="preview.jpg",
    )

    import cerebrasgemma4.api.routes.jobs as jobs_route

    jobs_route._get_store = lambda: store
    try:
        client = TestClient(app)
        resp = client.get("/api/jobs")
        assert resp.status_code == 200
        job = resp.json()["jobs"][0]
        assert job["title"] == "My YouTube Video"
        assert job["thumbnail_url"].endswith("/youtube_thumbnail.jpg")
        assert job["preview_url"].endswith("/preview.jpg")
    finally:
        jobs_route._get_store = JobStore