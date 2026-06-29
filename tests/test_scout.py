from pathlib import Path

from PIL import Image, ImageDraw

from cerebrasgemma4.pipeline.frames import Frame
from cerebrasgemma4.pipeline.chapters import ScoutRegion
from cerebrasgemma4.pipeline.gemma.scout import (
    FrameScore,
    _build_scout_global_prompt,
    _build_scout_mosaic_prompt,
    _cluster_frame_ids,
    _frame_dhash,
    _parse_frame_scores,
    _parse_json,
    _salvage_object_array,
    _scout_mosaic_max_tokens,
    apply_duplicate_penalties,
)


def test_scout_mosaic_max_tokens_scales_with_frames():
    assert _scout_mosaic_max_tokens(5) >= 768
    assert _scout_mosaic_max_tokens(25) > _scout_mosaic_max_tokens(5)
    assert _scout_mosaic_max_tokens(100) <= 8192


def test_salvage_truncated_frames_array():
    partial = (
        '{"frames":[{"frame_id":"f_0001","timestamp_sec":1.0,'
        '"relevance":0.9,"has_readable_text":true,"brief":"slide title"},'
        '{"frame_id":"f_0002","timestamp_sec":2.0,"relevance":0.4,'
        '"has_readable_text":false,"brief":"talking he'
    )
    salvaged = _salvage_object_array(partial, "frames")
    assert salvaged is not None
    assert len(salvaged["frames"]) == 1
    assert salvaged["frames"][0]["frame_id"] == "f_0001"


def test_parse_json_salvages_truncated_mosaic_response():
    partial = (
        '{"frames":[{"frame_id":"f_0010","timestamp_sec":10.0,'
        '"relevance":0.8,"has_readable_text":false,"brief":"server rack"},'
        '{"frame_id":"f_0011","timestamp_sec":11.0,"relevance":0.3,'
        '"has_readable_text":false,"brief":"unfin'
    )
    data = _parse_json(partial)
    assert len(data["frames"]) == 1


def test_parse_frame_scores_backfills_missing_frames():
    frames = [
        Frame("f_0001", 1.0, Path("/tmp/a.jpg")),
        Frame("f_0002", 2.0, Path("/tmp/b.jpg")),
    ]
    known = {f.frame_id: f for f in frames}
    data = {
        "frames": [
            {
                "frame_id": "f_0001",
                "timestamp_sec": 1.0,
                "relevance": 0.9,
                "has_readable_text": True,
                "brief": "chip close-up",
            }
        ]
    }
    scores = _parse_frame_scores(data, known)
    assert len(scores) == 2
    by_id = {s.frame_id: s for s in scores}
    assert by_id["f_0001"].relevance == 0.9
    assert by_id["f_0002"].relevance == 0.2


def test_build_scout_global_prompt_uses_photo_editor_criteria():
    region = ScoutRegion(
        region_id="r_01",
        label="Architecture overview",
        start_sec=0.0,
        end_sec=45.0,
        source="chapter",
    )
    prompt = _build_scout_global_prompt(
        transcript_excerpt="Today we deploy the model.",
        region_list="- r_01: Architecture overview (0.0s–45.0s)",
        single_region=region,
    )
    assert "photo editor" in prompt
    assert "needs_detail" in prompt
    assert "contact-sheet" in prompt

    mosaic_prompt = _build_scout_global_prompt(
        transcript_excerpt="",
        region_list="- r_01: Intro (0.0s–30.0s)\n- r_02: Demo (30.0s–90.0s)",
        cols=2,
        rows=1,
    )
    assert "MOSAIC grid" in mosaic_prompt
    assert "<scoring_guide>" in mosaic_prompt


def test_build_scout_mosaic_prompt_uses_photo_editor_criteria():
    prompt = _build_scout_mosaic_prompt(
        cols=4,
        rows=2,
        window_label="Architecture demo",
        time_range="12.0s-20.0s",
        transcript_excerpt="We deploy the model here.",
        frame_list="- Cell 1: f_0001 at 12.0s",
    )
    assert "<role>" in prompt
    assert "photo editor" in prompt
    assert "<editorial_criteria>" in prompt
    assert "<burst_rule>" in prompt
    assert "Decisive moment" in prompt


def _save_pattern(path: Path, *, offset: int) -> None:
    image = Image.new("RGB", (320, 180), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    for x in range(0, 320, 20):
        color = (offset + x) % 255
        draw.rectangle((x, 0, x + 10, 180), fill=(color, 40, 120))
    image.save(path)


def test_frame_dhash_matches_identical_images(tmp_path: Path):
    path_a = tmp_path / "a.jpg"
    path_b = tmp_path / "b.jpg"
    _save_pattern(path_a, offset=0)
    _save_pattern(path_b, offset=0)
    assert _frame_dhash(path_a) == _frame_dhash(path_b)


def test_cluster_frame_ids_groups_near_duplicates(tmp_path: Path):
    duplicate = tmp_path / "dup.jpg"
    unique = tmp_path / "unique.jpg"
    _save_pattern(duplicate, offset=0)
    _save_pattern(tmp_path / "dup_copy.jpg", offset=0)
    _save_pattern(unique, offset=120)

    clusters = _cluster_frame_ids(
        {
            "f_dup_a": tmp_path / "dup.jpg",
            "f_dup_b": tmp_path / "dup_copy.jpg",
            "f_unique": unique,
        }
    )
    cluster_sizes = sorted(len(cluster) for cluster in clusters)
    assert cluster_sizes == [1, 2]


def test_apply_duplicate_penalties_keeps_best_keeper(tmp_path: Path):
    duplicate = tmp_path / "dup.jpg"
    duplicate_copy = tmp_path / "dup_copy.jpg"
    unique = tmp_path / "unique.jpg"
    _save_pattern(duplicate, offset=0)
    _save_pattern(duplicate_copy, offset=0)
    _save_pattern(unique, offset=120)

    scores = [
        FrameScore("f_dup_a", 1.0, 0.85, "hero slide", True),
        FrameScore("f_dup_b", 2.0, 0.82, "same slide", True),
        FrameScore("f_unique", 3.0, 0.7, "diagram", False),
    ]
    frame_paths = {
        "f_dup_a": duplicate,
        "f_dup_b": duplicate_copy,
        "f_unique": unique,
    }

    adjusted = {s.frame_id: s for s in apply_duplicate_penalties(scores, frame_paths)}
    assert adjusted["f_dup_a"].relevance == 0.85
    assert adjusted["f_dup_b"].relevance == 0.52
    assert adjusted["f_unique"].relevance == 0.7