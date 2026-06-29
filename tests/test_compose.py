from cerebrasgemma4.pipeline.gemma.compose import (
    COMPOSE_DEFAULT_INSTRUCTIONS,
    ComposeInput,
    build_compose_prompt,
)
from cerebrasgemma4.pipeline.transcript import TranscriptResult


def _sample_input(**kwargs) -> ComposeInput:
    defaults = {
        "source_name": "demo.mp4",
        "duration_sec": 125.0,
        "transcript": TranscriptResult(
            segments=[],
            source="mock",
            full_text="Hello world transcript.",
        ),
        "analyses": [],
    }
    defaults.update(kwargs)
    return ComposeInput(**defaults)


def test_build_compose_prompt_includes_default_instructions():
    prompt = build_compose_prompt(_sample_input())
    assert COMPOSE_DEFAULT_INSTRUCTIONS in prompt
    assert "photo editor" in prompt
    assert "<editorial_rules>" in prompt
    assert "demo.mp4" in prompt
    assert "Hello world transcript." in prompt


def test_build_compose_prompt_appends_custom_instructions():
    prompt = build_compose_prompt(
        _sample_input(custom_prompt="Focus on pricing and use bullet lists.")
    )
    assert "## Additional instructions from the user" in prompt
    assert "Focus on pricing and use bullet lists." in prompt
    assert prompt.index(COMPOSE_DEFAULT_INSTRUCTIONS) < prompt.index("Focus on pricing")


def test_defaults_endpoint_exposes_compose_prompt():
    from fastapi.testclient import TestClient

    from cerebrasgemma4.api.main import app

    client = TestClient(app)
    resp = client.get("/api/defaults")
    assert resp.status_code == 200
    assert resp.json()["compose_prompt"] == COMPOSE_DEFAULT_INSTRUCTIONS