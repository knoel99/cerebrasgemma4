from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from cerebrasgemma4.api.main import app
from cerebrasgemma4.pipeline.ingest import VideoMetadata


def test_probe_youtube_without_download():
    meta = VideoMetadata(
        duration_sec=2745.0,
        width=1920,
        height=1080,
        fps=50.0,
        source_path=Path("https://youtu.be/sYhIjzs3Lwc"),
    )

    with patch(
        "cerebrasgemma4.api.routes.convert.probe_youtube",
        return_value=meta,
    ):
        client = TestClient(app)
        resp = client.post(
            "/api/probe",
            data={"youtube_url": "https://youtu.be/sYhIjzs3Lwc"},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["duration_sec"] == 2745.0
    assert data["max_duration_sec"] == 2745.0
    assert data["estimated_scout_calls"] == 549