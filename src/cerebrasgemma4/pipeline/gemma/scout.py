"""Scout stage: score frame chunks for document relevance."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from pathlib import Path

from PIL import Image

from cerebrasgemma4.images import MOSAIC_MAX_CELLS, build_frame_mosaic, mosaic_grid
from cerebrasgemma4.llm import (
    build_multimodal_message,
    build_multimodal_message_bytes,
    complete,
)
from cerebrasgemma4.pipeline.chapters import ScoutRegion
from cerebrasgemma4.pipeline.frames import Frame, FrameChunk


@dataclass
class FrameScore:
    frame_id: str
    timestamp_sec: float
    relevance: float
    brief: str
    has_readable_text: bool


@dataclass
class RegionScore:
    region_id: str
    region: ScoutRegion
    relevance: float
    needs_detail: bool
    brief: str


GLOBAL_SCOUT_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "region_scores",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "regions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "region_id": {"type": "string"},
                            "relevance": {"type": "number"},
                            "needs_detail": {"type": "boolean"},
                            "brief": {"type": "string"},
                        },
                        "required": ["region_id", "relevance", "needs_detail", "brief"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["regions"],
            "additionalProperties": False,
        },
    },
}


SCOUT_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "frame_scores",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "frames": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "frame_id": {"type": "string"},
                            "timestamp_sec": {"type": "number"},
                            "relevance": {"type": "number"},
                            "has_readable_text": {"type": "boolean"},
                            "brief": {"type": "string"},
                        },
                        "required": [
                            "frame_id",
                            "timestamp_sec",
                            "relevance",
                            "has_readable_text",
                            "brief",
                        ],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["frames"],
            "additionalProperties": False,
        },
    },
}

DUPLICATE_HASH_THRESHOLD = 6
DUPLICATE_RELEVANCE_PENALTY = 0.3
_DHASH_SIZE = 8


def _scout_mosaic_max_tokens(frame_count: int) -> int:
    # ~120-150 completion tokens per scored frame; 1024 truncates large mosaics.
    return min(8192, max(768, frame_count * 150))


def _salvage_object_array(content: str, key: str) -> dict | None:
    """Pull complete objects from a truncated JSON array."""
    marker = re.search(rf'"{re.escape(key)}"\s*:\s*\[', content)
    if not marker:
        return None
    items: list[dict] = []
    depth = 0
    obj_start: int | None = None
    for i in range(marker.end(), len(content)):
        ch = content[i]
        if ch == "{":
            if depth == 0:
                obj_start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and obj_start is not None:
                try:
                    items.append(json.loads(content[obj_start : i + 1]))
                except json.JSONDecodeError:
                    pass
                obj_start = None
    return {key: items} if items else None


def _parse_json(content: str) -> dict:
    content = content.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*", content, re.DOTALL)
    if not match:
        raise json.JSONDecodeError("No JSON object found", content, 0)
    blob = match.group()

    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        for key in ("frames", "regions"):
            salvaged = _salvage_object_array(blob, key)
            if salvaged:
                return salvaged
        raise


def _frame_dhash(path: Path) -> int | None:
    """Difference hash (64-bit) for near-duplicate frame clustering."""
    try:
        with Image.open(path) as img:
            gray = img.convert("L").resize(
                (_DHASH_SIZE + 1, _DHASH_SIZE),
                Image.Resampling.LANCZOS,
            )
            pixels = list(gray.get_flattened_data())
    except OSError:
        return None

    bits = 0
    width = _DHASH_SIZE + 1
    for row in range(_DHASH_SIZE):
        row_offset = row * width
        for col in range(_DHASH_SIZE):
            left = pixels[row_offset + col]
            right = pixels[row_offset + col + 1]
            bits = (bits << 1) | (1 if left > right else 0)
    return bits


def _hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def _cluster_frame_ids(
    frame_paths: dict[str, Path],
    *,
    threshold: int = DUPLICATE_HASH_THRESHOLD,
) -> list[list[str]]:
    frame_ids = list(frame_paths)
    if len(frame_ids) <= 1:
        return [frame_ids] if frame_ids else []

    hashes: dict[str, int | None] = {
        frame_id: _frame_dhash(path) for frame_id, path in frame_paths.items()
    }
    parent = {frame_id: frame_id for frame_id in frame_ids}

    def find(frame_id: str) -> str:
        root = frame_id
        while parent[root] != root:
            parent[root] = parent[parent[root]]
            root = parent[root]
        return root

    def union(left_id: str, right_id: str) -> None:
        parent[find(left_id)] = find(right_id)

    for i, left_id in enumerate(frame_ids):
        left_hash = hashes[left_id]
        if left_hash is None:
            continue
        for right_id in frame_ids[i + 1 :]:
            right_hash = hashes[right_id]
            if right_hash is None:
                continue
            if _hamming_distance(left_hash, right_hash) <= threshold:
                union(left_id, right_id)

    clusters: dict[str, list[str]] = {}
    for frame_id in frame_ids:
        clusters.setdefault(find(frame_id), []).append(frame_id)
    return list(clusters.values())


def apply_duplicate_penalties(
    scores: list[FrameScore],
    frame_paths: dict[str, Path],
    *,
    penalty: float = DUPLICATE_RELEVANCE_PENALTY,
    threshold: int = DUPLICATE_HASH_THRESHOLD,
) -> list[FrameScore]:
    """Penalize near-duplicate frames, keeping the highest-scored keeper per cluster."""
    if len(scores) <= 1:
        return scores

    known_paths = {
        score.frame_id: frame_paths[score.frame_id]
        for score in scores
        if score.frame_id in frame_paths
    }
    if len(known_paths) <= 1:
        return scores

    by_id = {score.frame_id: score for score in scores}
    adjusted = dict(by_id)
    for cluster in _cluster_frame_ids(known_paths, threshold=threshold):
        if len(cluster) <= 1:
            continue
        members = [by_id[frame_id] for frame_id in cluster if frame_id in by_id]
        if len(members) <= 1:
            continue
        best = max(
            members,
            key=lambda score: (score.relevance, score.has_readable_text, -score.timestamp_sec),
        )
        for member in members:
            if member.frame_id == best.frame_id:
                continue
            adjusted[member.frame_id] = replace(
                member,
                relevance=max(0.0, member.relevance - penalty),
            )

    return [adjusted[score.frame_id] for score in scores]


def _build_scout_mosaic_prompt(
    *,
    cols: int,
    rows: int,
    window_label: str,
    time_range: str,
    transcript_excerpt: str,
    frame_list: str,
) -> str:
    transcript = transcript_excerpt or "(no speech)"
    return (
        "<role>\n"
        "You are a senior photo editor reviewing a contact sheet of video screenshots. "
        "Pick frames that will illustrate a written document with the same taste a magazine "
        "editor applies to a storyboard.\n"
        "</role>\n\n"
        "<contact_sheet>\n"
        f"MOSAIC grid: {cols} columns x {rows} rows, read left-to-right, top-to-bottom.\n"
        "Each cell label shows frame_id and timestamp in seconds.\n"
        f"Section: {window_label} ({time_range}, 1 fps sampling).\n"
        "</contact_sheet>\n\n"
        f"<transcript>\n{transcript}\n</transcript>\n\n"
        f"<frames>\n{frame_list}\n</frames>\n\n"
        "<editorial_criteria>\n"
        "Score each frame like a photo editor, not a keyword matcher.\n\n"
        "PRIMARY (document value):\n"
        "- Readable on-screen text (slides, code, UI labels, diagrams)\n"
        "- Unique information not visible in adjacent frames\n"
        "- Decisive moment: demo step or slide fully visible, not mid-transition\n"
        "- Visual that anchors a transcript claim\n\n"
        "SECONDARY (photographic quality):\n"
        "- Sharpness: no motion blur, no loading spinners, no half-rendered UI\n"
        "- Composition: subject separated from background; not half-cropped\n"
        "- Clarity at small size: would this work as a report thumbnail?\n"
        "- Clean frame: no distracting overlays or cursor blocking text\n\n"
        "REJECT or score <= 0.2:\n"
        "- Talking head alone with no supporting visual\n"
        "- Duplicate of a neighboring frame (same slide or pose)\n"
        "- Blank, black, or between-scenes frames\n"
        "</editorial_criteria>\n\n"
        "<scoring_guide>\n"
        "0.0-0.2  Discard — redundant, blank, blurry, or no document value\n"
        "0.3-0.5  Context only — usable if nothing better exists nearby\n"
        "0.6-0.8  Strong keeper — clear visual evidence for the narrative\n"
        "0.9-1.0  Hero frame — title slide, key diagram, completed demo, dense text\n"
        "</scoring_guide>\n\n"
        "<burst_rule>\n"
        "When 2+ consecutive frames show the same slide or scene, score ONLY the "
        "sharpest, most complete, best-composed one high. Penalize near-duplicates "
        "by at least 0.3 versus the best in the group.\n"
        "</burst_rule>\n\n"
        "<instructions>\n"
        "For EVERY frame_id listed in <frames>:\n"
        "1. relevance: 0.0-1.0 using <scoring_guide> and <editorial_criteria>\n"
        "2. has_readable_text: true when on-screen text is legible\n"
        "3. brief: max 12 words — editor note (e.g. hero slide, demo complete)\n"
        "Return exactly one entry per listed frame_id.\n"
        "</instructions>"
    )


def _mosaic_frame_list(frames: list[Frame]) -> str:
    cols, _rows = mosaic_grid(len(frames))
    return "\n".join(
        f"- Cell {i + 1} (row {i // cols + 1}, col {i % cols + 1}): "
        f"{f.frame_id} at {f.timestamp_sec:.1f}s"
        for i, f in enumerate(frames)
    )


def _parse_frame_scores(data: dict, known: dict[str, Frame]) -> list[FrameScore]:
    scores: list[FrameScore] = []
    for item in data.get("frames", []):
        frame_id = item.get("frame_id")
        if not frame_id:
            continue
        frame = known.get(frame_id)
        if frame is None:
            continue
        scores.append(
            FrameScore(
                frame_id=frame_id,
                timestamp_sec=float(item.get("timestamp_sec", frame.timestamp_sec)),
                relevance=float(item.get("relevance", 0.2)),
                brief=str(item.get("brief", "")),
                has_readable_text=bool(item.get("has_readable_text", False)),
            )
        )
    scored_ids = {s.frame_id for s in scores}
    for frame in known.values():
        if frame.frame_id in scored_ids:
            continue
        scores.append(
            FrameScore(
                frame_id=frame.frame_id,
                timestamp_sec=frame.timestamp_sec,
                relevance=0.2,
                brief="",
                has_readable_text=False,
            )
        )
    return scores


def _complete_scout_json(
    msg: dict,
    *,
    response_format: dict,
    max_completion_tokens: int,
) -> tuple[dict, dict]:
    """Call Gemma and parse structured scout JSON with one retry on truncation."""
    last_exc: json.JSONDecodeError | None = None
    tokens = max_completion_tokens
    result = None
    for _ in range(2):
        result = complete(
            [msg],
            response_format=response_format,
            temperature=0.3,
            max_completion_tokens=tokens,
        )
        try:
            return _parse_json(result.content), {
                "usage": result.usage,
                "time_info": result.time_info,
            }
        except json.JSONDecodeError as exc:
            last_exc = exc
            tokens = min(8192, tokens * 2)
    assert last_exc is not None
    raise last_exc


def _build_scout_global_prompt(
    *,
    transcript_excerpt: str,
    region_list: str,
    single_region: ScoutRegion | None = None,
    cols: int | None = None,
    rows: int | None = None,
) -> str:
    transcript = transcript_excerpt or "(no speech)"
    if single_region is not None:
        layout = (
            "<layout>\n"
            "ONE representative frame from a distinct video section.\n"
            f"Section: {single_region.region_id} — {single_region.label}\n"
            "</layout>"
        )
        targets = (
            f"<sections>\n"
            f"- {single_region.region_id}: {single_region.label} "
            f"({single_region.start_sec:.1f}s–{single_region.end_sec:.1f}s)\n"
            f"</sections>"
        )
    else:
        layout = (
            "<layout>\n"
            f"MOSAIC grid: {cols} columns x {rows} rows, read left-to-right, top-to-bottom.\n"
            "Each cell is one representative frame from a distinct section; "
            "cell label shows region_id and title.\n"
            "</layout>"
        )
        targets = f"<sections>\n{region_list}\n</sections>"

    return (
        "<role>\n"
        "You are a photo editor planning coverage for a video-to-document project. "
        "Each section has one preview frame — decide which chapters deserve a full "
        "contact-sheet review versus a light mention in the final write-up.\n"
        "</role>\n\n"
        f"{layout}\n\n"
        f"<transcript>\n{transcript}\n</transcript>\n\n"
        f"{targets}\n\n"
        "<editorial_criteria>\n"
        "HIGH relevance: title moments, demos, architecture diagrams, dense slides, "
        "results, or any section where visuals carry information the transcript alone misses.\n"
        "LOW relevance: intro/outro filler, talking-head-only segments, transitions, "
        "repeated recap of earlier content.\n"
        "Set needs_detail=true when the section likely needs closer frame inspection "
        "(slides, UI walkthroughs, code, diagrams, readable on-screen text).\n"
        "</editorial_criteria>\n\n"
        "<scoring_guide>\n"
        "0.0-0.2  Skip — little document value\n"
        "0.3-0.5  Mention briefly if space allows\n"
        "0.6-0.8  Solid chapter — include in the document\n"
        "0.9-1.0  Anchor chapter — hero content, key teaching or demo moment\n"
        "</scoring_guide>\n\n"
        "<instructions>\n"
        "For every section listed in <sections>:\n"
        "1. relevance: 0.0-1.0 using <scoring_guide>\n"
        "2. needs_detail: true when a contact-sheet pass is warranted\n"
        "3. brief: max 12 words — editor note on why this section matters\n"
        "Return exactly one entry per region_id.\n"
        "</instructions>"
    )


def scout_global_batch(
    batch: list[tuple[ScoutRegion, Frame]],
    transcript_excerpt: str,
) -> tuple[list[RegionScore], dict]:
    """Score coarse video regions from one representative frame each."""
    batch = [(region, frame) for region, frame in batch if frame.path.is_file()]
    if not batch:
        return [], {"usage": {}, "time_info": {}}

    region_list = "\n".join(
        f"- {region.region_id}: {region.label} ({region.start_sec:.1f}s–{region.end_sec:.1f}s)"
        for region, _frame in batch
    )
    if len(batch) == 1:
        region, frame = batch[0]
        prompt = _build_scout_global_prompt(
            transcript_excerpt=transcript_excerpt,
            region_list=region_list,
            single_region=region,
        )
        msg = build_multimodal_message(prompt, [frame.path], detail=False)
    else:
        cols, rows = mosaic_grid(len(batch))
        labels = [
            f"{region.region_id}\n{region.label[:18]}"
            for region, _frame in batch
        ]
        mosaic = build_frame_mosaic(
            [frame.path for _region, frame in batch],
            labels,
        )
        prompt = _build_scout_global_prompt(
            transcript_excerpt=transcript_excerpt,
            region_list=region_list,
            cols=cols,
            rows=rows,
        )
        msg = build_multimodal_message_bytes(prompt, [mosaic])
    data, metrics = _complete_scout_json(
        msg,
        response_format=GLOBAL_SCOUT_SCHEMA,
        max_completion_tokens=2048,
    )
    by_id = {region.region_id: region for region, _ in batch}
    scores: list[RegionScore] = []
    for item in data.get("regions", []):
        region_id = item["region_id"]
        region = by_id.get(region_id)
        if region is None:
            continue
        scores.append(
            RegionScore(
                region_id=region_id,
                region=region,
                relevance=float(item["relevance"]),
                needs_detail=bool(item["needs_detail"]),
                brief=item["brief"],
            )
        )
    # Ensure every batched region has a score (model may omit one).
    scored_ids = {s.region_id for s in scores}
    for region, _frame in batch:
        if region.region_id not in scored_ids:
            scores.append(
                RegionScore(
                    region_id=region.region_id,
                    region=region,
                    relevance=0.25,
                    needs_detail=False,
                    brief=region.label,
                )
            )
    return scores, metrics


def scout_mosaic(
    frames: list[Frame],
    transcript_excerpt: str,
    *,
    window_label: str,
    time_range: str,
) -> tuple[list[FrameScore], dict]:
    """Score frames from a single labeled mosaic (one Cerebras image)."""
    frames = [frame for frame in frames if frame.path.is_file()]
    if not frames:
        return [], {"usage": {}, "time_info": {}}

    cols, rows = mosaic_grid(len(frames))
    labels = [f"{f.frame_id}\n{f.timestamp_sec:.0f}s" for f in frames]
    mosaic = build_frame_mosaic([f.path for f in frames], labels)
    frame_list = _mosaic_frame_list(frames)
    prompt = _build_scout_mosaic_prompt(
        cols=cols,
        rows=rows,
        window_label=window_label,
        time_range=time_range,
        transcript_excerpt=transcript_excerpt,
        frame_list=frame_list,
    )
    msg = build_multimodal_message_bytes(prompt, [mosaic])
    data, metrics = _complete_scout_json(
        msg,
        response_format=SCOUT_SCHEMA,
        max_completion_tokens=_scout_mosaic_max_tokens(len(frames)),
    )
    known = {f.frame_id: f for f in frames}
    scores = _parse_frame_scores(data, known)
    return scores, metrics


def scout_chunk(
    chunk: FrameChunk,
    transcript_excerpt: str,
) -> tuple[list[FrameScore], dict]:
    return scout_mosaic(
        chunk.frames,
        transcript_excerpt,
        window_label="consecutive frames",
        time_range=f"{chunk.start_sec:.1f}s–{chunk.end_sec:.1f}s",
    )


def scout_region_mosaic(
    region: ScoutRegion,
    frames: list[Frame],
    transcript_excerpt: str,
) -> tuple[list[FrameScore], dict]:
    """Score sampled frames from one chapter/segment via mosaic."""
    return scout_mosaic(
        frames,
        transcript_excerpt,
        window_label=region.label,
        time_range=f"{region.start_sec:.1f}s–{region.end_sec:.1f}s",
    )


def select_top_frames(
    all_scores: list[FrameScore],
    frame_paths: dict[str, Path],
    *,
    max_frames: int,
) -> list[tuple[FrameScore, Path]]:
    ranked = sorted(all_scores, key=lambda s: s.relevance, reverse=True)
    selected: list[tuple[FrameScore, Path]] = []
    seen_ts: set[int] = set()
    for score in ranked:
        bucket = int(score.timestamp_sec)
        if bucket in seen_ts:
            continue
        path = frame_paths.get(score.frame_id)
        if path is None:
            continue
        selected.append((score, path))
        seen_ts.add(bucket)
        if len(selected) >= max_frames:
            break
    selected.sort(key=lambda x: x[0].timestamp_sec)
    return selected