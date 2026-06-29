import json
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from cerebrasgemma4.api.main import app
from cerebrasgemma4.pipeline.chat_store import append_section_to_document, load_chat_history
from cerebrasgemma4.pipeline.context import VideoContext, save_context
from cerebrasgemma4.pipeline.gemma.chat import (
    ChatEnrichment,
    ChatTurnResult,
    build_chat_messages,
    build_context_block,
)
from cerebrasgemma4.pipeline.jobs import JobStatus, JobStore


def test_build_context_block_includes_transcript():
    ctx = VideoContext(
        source_name="demo.mp4",
        duration_sec=90.0,
        transcript_source="mock",
        transcript_full_text="Pricing discussed at minute two.",
        analyses=[],
    )
    block = build_context_block(ctx, "# Report\n\nBody.")
    assert "Pricing discussed" in block
    assert "# Report" in block


def test_build_chat_messages_includes_user_turn():
    messages = build_chat_messages(
        ctx=None,
        document_md="# Doc",
        chat_history=[{"role": "user", "content": "Earlier?"}],
        user_message="Add a FAQ",
    )
    assert messages[-1]["content"] == "Add a FAQ"
    assert any(m["role"] == "system" for m in messages)


def test_append_section_to_document(tmp_path: Path):
    doc = tmp_path / "document.md"
    doc.write_text("# Title\n\nIntro.\n", encoding="utf-8")
    updated = append_section_to_document(doc, "FAQ", "- Q1\n- A1")
    assert "## FAQ" in updated
    assert "- Q1" in updated


def _seed_completed_job(store: JobStore) -> str:
    record = store.create()
    job_dir = store.path(record.job_id)
    doc_path = job_dir / "document.md"
    doc_path.write_text("# Demo\n\nOriginal.\n", encoding="utf-8")
    save_context(
        job_dir,
        VideoContext(
            source_name="demo.mp4",
            duration_sec=60.0,
            transcript_source="mock",
            transcript_full_text="Hello transcript",
            analyses=[],
        ),
    )
    store.update(
        record.job_id,
        status=JobStatus.COMPLETED,
        document_path=str(doc_path),
        title="Demo",
    )
    return record.job_id


def test_chat_apply_endpoint(tmp_path: Path):
    store = JobStore(root=tmp_path)
    job_id = _seed_completed_job(store)

    import cerebrasgemma4.api.routes.chat as chat_route

    chat_route._get_store = lambda: store
    try:
        client = TestClient(app)
        resp = client.post(
            f"/api/jobs/{job_id}/chat/apply",
            json={"title": "Extra", "markdown": "More details here."},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "## Extra" in data["markdown"]
        assert "More details here." in data["markdown"]
    finally:
        chat_route._get_store = JobStore


def test_post_chat_sse_persists_history(tmp_path: Path):
    store = JobStore(root=tmp_path)
    job_id = _seed_completed_job(store)
    mock_turn = ChatTurnResult(
        reply="Here is a FAQ suggestion.",
        enrichment=ChatEnrichment(title="FAQ", markdown="- Q: What?\n- A: This."),
        usage={"prompt_tokens": 1, "completion_tokens": 2},
        time_info={"time_to_first_token": 0.01},
    )

    import cerebrasgemma4.api.routes.chat as chat_route

    chat_route._get_store = lambda: store
    try:
        with patch("cerebrasgemma4.api.routes.chat.run_chat_turn", return_value=mock_turn):
            client = TestClient(app)
            resp = client.post(
                f"/api/jobs/{job_id}/chat",
                json={"message": "Add FAQ"},
            )
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers.get("content-type", "")
            body = resp.text
            assert "data:" in body
            assert '"done": true' in body or '"done":true' in body

        history = load_chat_history(store.path(job_id))
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"
        assert history[1]["enrichment"]["title"] == "FAQ"
    finally:
        chat_route._get_store = JobStore