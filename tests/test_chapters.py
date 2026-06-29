from cerebrasgemma4.pipeline.chapters import (
    VideoChapter,
    fetch_youtube_chapters,
    plan_scout_regions,
)
from unittest.mock import patch


def test_plan_scout_regions_from_chapters():
    chapters = [
        VideoChapter("Intro", 0, 30),
        VideoChapter("Main", 30, 90),
    ]
    regions = plan_scout_regions(90, chapters)
    assert len(regions) == 2
    assert regions[0].source == "chapter"
    assert regions[0].label == "Intro"


def test_plan_scout_regions_without_chapters():
    regions = plan_scout_regions(125)
    assert len(regions) == 3
    assert all(r.source == "segment" for r in regions)


def test_fetch_youtube_chapters_from_ytdlp():
    payload = {
        "duration": 120,
        "chapters": [
            {"start_time": 0, "end_time": 45, "title": "Part A"},
            {"start_time": 45, "end_time": 120, "title": "Part B"},
        ],
    }
    with patch("cerebrasgemma4.pipeline.chapters._ytdlp_json", return_value=payload):
        chapters = fetch_youtube_chapters("https://www.youtube.com/watch?v=abc")

    assert len(chapters) == 2
    assert chapters[1].title == "Part B"
    assert chapters[1].end_sec == 120