from pathlib import Path

from fastapi.testclient import TestClient

from cerebrasgemma4.api.main import STATIC_DIR, app


def test_index():
    client = TestClient(app)
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers.get("content-type", "")
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    assert "Sightline" in html
    assert "Cerebras Inference" in html


def test_static_assets():
    client = TestClient(app)
    resp = client.get("/static/app.js")
    assert resp.status_code == 200


def test_health():
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["model"] == "gemma-4-31b"


def test_convert_requires_input():
    client = TestClient(app)
    resp = client.post("/api/convert")
    assert resp.status_code == 400


def test_probe_requires_input():
    client = TestClient(app)
    resp = client.post("/api/probe")
    assert resp.status_code == 400


def test_list_jobs_empty():
    client = TestClient(app)
    resp = client.get("/api/jobs")
    assert resp.status_code == 200
    data = resp.json()
    assert "jobs" in data
    assert isinstance(data["jobs"], list)