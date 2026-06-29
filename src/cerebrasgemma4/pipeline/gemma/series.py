"""Collect time-series observations from frames and transcript."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from cerebrasgemma4.images import MOSAIC_MAX_CELLS, build_frame_mosaic, mosaic_grid
from cerebrasgemma4.llm import build_multimodal_message_bytes, complete
from cerebrasgemma4.pipeline.demo import is_demo_mode
from cerebrasgemma4.pipeline.frames import Frame

FRAME_SAMPLE_MAX = 50
DEMO_FRAME_SAMPLE_MAX = 25

_CHART_HINTS = (
    "graph",
    "chart",
    "plot",
    "graphique",
    "courbe",
    "diagram",
    "visualis",
    "évolution",
    "evolution",
    "trend",
    "time series",
    "series temporelle",
    "metric",
    "métrique",
    "metrique",
    "données",
    "donnees",
    "data",
    "statistics",
    "statistique",
    "compare",
    "comparison",
    "tableau",
)

_METRIC_ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "key": {"type": "string"},
        "label": {"type": "string"},
        "value": {"anyOf": [{"type": "number"}, {"type": "null"}]},
        "unit": {"type": "string"},
    },
    "required": ["key", "label", "value", "unit"],
    "additionalProperties": False,
}

_FRAME_BATCH_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "frame_observations",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "observations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "frame_id": {"type": "string"},
                            "timestamp_sec": {"type": "number"},
                            "metrics": {
                                "type": "array",
                                "items": _METRIC_ITEM_SCHEMA,
                            },
                            "notes": {"type": "string"},
                        },
                        "required": ["frame_id", "timestamp_sec", "metrics", "notes"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["observations"],
            "additionalProperties": False,
        },
    },
}

_TRANSCRIPT_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "transcript_observations",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "observations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "timestamp_sec": {"type": "number"},
                            "metrics": {
                                "type": "array",
                                "items": _METRIC_ITEM_SCHEMA,
                            },
                            "quote": {"type": "string"},
                        },
                        "required": ["timestamp_sec", "metrics", "quote"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["observations"],
            "additionalProperties": False,
        },
    },
}


@dataclass
class MetricValue:
    key: str
    label: str
    value: float | None
    unit: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "value": self.value,
            "unit": self.unit,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MetricValue:
        return cls(
            key=str(data.get("key") or "value"),
            label=str(data.get("label") or data.get("key") or "Value"),
            value=_optional_float(data.get("value")),
            unit=str(data.get("unit") or ""),
        )


@dataclass
class DataObservation:
    timestamp_sec: float
    source: str
    metrics: list[MetricValue] = field(default_factory=list)
    frame_id: str | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp_sec": self.timestamp_sec,
            "source": self.source,
            "frame_id": self.frame_id,
            "metrics": [m.to_dict() for m in self.metrics],
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DataObservation:
        if "metrics" in data:
            metrics = [MetricValue.from_dict(m) for m in data.get("metrics") or []]
        else:
            metrics = _metrics_from_legacy_record(data)
        return cls(
            timestamp_sec=float(data.get("timestamp_sec", 0)),
            source=str(data.get("source") or "image"),
            metrics=metrics,
            frame_id=data.get("frame_id"),
            notes=str(data.get("notes") or data.get("quote") or ""),
        )


def _metrics_from_legacy_record(data: dict[str, Any]) -> list[MetricValue]:
    """Upgrade older flat observation records to generic metrics."""
    skip = {"timestamp_sec", "frame_id", "notes", "quote", "source", "metrics"}
    out: list[MetricValue] = []
    for key, raw in data.items():
        if key in skip:
            continue
        value = _optional_float(raw)
        if value is None:
            continue
        label = key.replace("_", " ").strip().title() or key
        out.append(MetricValue(key=_slug_key(key), label=label, value=value, unit=""))
    return out


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def prompt_requests_charts(text: str | None) -> bool:
    if not text or not text.strip():
        return False
    lowered = text.lower()
    return any(hint in lowered for hint in _CHART_HINTS)


def frame_sample_budget() -> int:
    return DEMO_FRAME_SAMPLE_MAX if is_demo_mode() else FRAME_SAMPLE_MAX


def sample_frames(frames: list[Frame], max_count: int | None = None) -> list[Frame]:
    available = [f for f in frames if f.path.is_file()]
    if not available:
        return []
    cap = max_count or frame_sample_budget()
    if len(available) <= cap:
        return available
    if cap <= 1:
        return [available[0]]
    step = (len(available) - 1) / (cap - 1)
    return [available[round(i * step)] for i in range(cap)]


def _observation_richness(obs: DataObservation) -> int:
    return sum(1 for m in obs.metrics if m.value is not None) + (1 if obs.notes else 0)


def merge_observations(observations: list[DataObservation]) -> list[DataObservation]:
    """Dedupe by source + time (+ frame), keep the richest metric set."""
    buckets: dict[tuple[str, float, str | None], DataObservation] = {}
    for obs in observations:
        key = (obs.source, round(obs.timestamp_sec, 2), obs.frame_id)
        current = buckets.get(key)
        if current is None or _observation_richness(obs) > _observation_richness(current):
            buckets[key] = obs
    return sorted(buckets.values(), key=lambda o: o.timestamp_sec)


def metric_definitions(observations: list[DataObservation]) -> list[MetricValue]:
    """Unique metric definitions across observations (for table headers / charts)."""
    seen: dict[str, MetricValue] = {}
    for obs in observations:
        for metric in obs.metrics:
            if metric.key not in seen:
                seen[metric.key] = metric
    return list(seen.values())


def _parse_json(content: str) -> dict:
    content = content.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise


def _slug_key(raw: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", raw.lower()).strip("_")
    return slug or "value"


def _parse_metrics(items: list[dict[str, Any]]) -> list[MetricValue]:
    metrics: list[MetricValue] = []
    for item in items or []:
        key = _slug_key(str(item.get("key") or item.get("label") or "value"))
        metrics.append(
            MetricValue(
                key=key,
                label=str(item.get("label") or key),
                value=_optional_float(item.get("value")),
                unit=str(item.get("unit") or ""),
            )
        )
    return metrics


def _mosaic_frame_list(frames: list[Frame]) -> str:
    cols, _rows = mosaic_grid(len(frames))
    return "\n".join(
        f"- Cell {i + 1} (row {i // cols + 1}, col {i % cols + 1}): "
        f"{f.frame_id} at {f.timestamp_sec:.1f}s"
        for i, f in enumerate(frames)
    )


def _frame_batch_prompt(
    *,
    frames: list[Frame],
    custom_prompt: str,
    time_range: str,
) -> str:
    cols, rows = mosaic_grid(len(frames))
    user_goal = custom_prompt.strip() or "Extract numeric values visible on screen over time."
    return (
        "<role>\n"
        "You extract structured numeric data from a video contact sheet (mosaic).\n"
        "</role>\n\n"
        f"<user_goal>\n{user_goal}\n</user_goal>\n\n"
        f"<window>{time_range}</window>\n\n"
        f"<mosaic>{cols} columns × {rows} rows — one labeled cell per frame.</mosaic>\n\n"
        f"<frames>\n{_mosaic_frame_list(frames)}\n</frames>\n\n"
        "<instructions>\n"
        "For EVERY frame_id listed above, return one observation.\n"
        "- Read numbers shown on screen: HUD, charts, captions, slides, dashboards, overlays.\n"
        "- Each metric needs: key (short slug), label (human name), value (number or null), unit.\n"
        "- Use consistent keys across cells when the same quantity appears (e.g. price_usd).\n"
        "- Convert units when needed; keep unit string explicit.\n"
        "- notes: max 12 words on what is visible.\n"
        "Return exactly one observation per listed frame_id.\n"
        "</instructions>"
    )


def read_frame_batch(
    frames: list[Frame],
    *,
    custom_prompt: str,
    time_range: str,
) -> tuple[list[DataObservation], dict]:
    frames = [f for f in frames if f.path.is_file()]
    if not frames:
        return [], {"usage": {}, "time_info": {}}

    labels = [f"{f.frame_id}\n{f.timestamp_sec:.0f}s" for f in frames]
    mosaic = build_frame_mosaic([f.path for f in frames], labels)
    prompt = _frame_batch_prompt(
        frames=frames,
        custom_prompt=custom_prompt,
        time_range=time_range,
    )
    msg = build_multimodal_message_bytes(prompt, [mosaic])
    result = complete(
        [msg],
        response_format=_FRAME_BATCH_SCHEMA,
        temperature=0.2,
        max_completion_tokens=min(8192, max(768, len(frames) * 140)),
    )
    data = _parse_json(result.content)
    known = {f.frame_id: f for f in frames}
    observations: list[DataObservation] = []
    for item in data.get("observations", []):
        frame_id = item.get("frame_id")
        frame = known.get(frame_id)
        if frame is None:
            continue
        observations.append(
            DataObservation(
                timestamp_sec=float(item.get("timestamp_sec", frame.timestamp_sec)),
                source="image",
                frame_id=frame_id,
                metrics=_parse_metrics(item.get("metrics") or []),
                notes=str(item.get("notes") or ""),
            )
        )
    return observations, {"usage": result.usage, "time_info": result.time_info}


def _format_transcript_excerpt(segments: list, *, max_chars: int = 10_000) -> str:
    lines: list[str] = []
    total = 0
    for seg in segments:
        start = float(getattr(seg, "start_sec", seg.get("start_sec", 0)))
        end = float(getattr(seg, "end_sec", seg.get("end_sec", start)))
        text = str(getattr(seg, "text", seg.get("text", ""))).strip()
        if not text:
            continue
        line = f"[{start:.1f}s–{end:.1f}s] {text}"
        if total + len(line) > max_chars:
            break
        lines.append(line)
        total += len(line)
    return "\n".join(lines) or "(empty transcript)"


def _transcript_prompt(*, transcript_excerpt: str, custom_prompt: str) -> str:
    user_goal = custom_prompt.strip() or "Extract numeric claims and measurements over time."
    return (
        "<role>\n"
        "You extract structured numeric data from a video transcript.\n"
        "</role>\n\n"
        f"<user_goal>\n{user_goal}\n</user_goal>\n\n"
        f"<transcript>\n{transcript_excerpt}\n</transcript>\n\n"
        "<instructions>\n"
        "- Find quantitative statements: counts, prices, percentages, durations, measurements.\n"
        "- timestamp_sec: best estimate of when the claim occurs (segment start is fine).\n"
        "- Each metric: key (slug), label, value (number), unit (or empty).\n"
        "- quote: short verbatim excerpt supporting the numbers.\n"
        "- Skip vague statements without numeric values.\n"
        "</instructions>"
    )


def read_transcript(
    segments: list,
    *,
    custom_prompt: str,
) -> tuple[list[DataObservation], dict]:
    excerpt = _format_transcript_excerpt(segments)
    if excerpt == "(empty transcript)":
        return [], {"usage": {}, "time_info": {}}

    prompt = _transcript_prompt(
        transcript_excerpt=excerpt,
        custom_prompt=custom_prompt,
    )
    result = complete(
        [{"role": "user", "content": prompt}],
        response_format=_TRANSCRIPT_SCHEMA,
        temperature=0.2,
        max_completion_tokens=4096,
    )
    data = _parse_json(result.content)
    observations: list[DataObservation] = []
    for item in data.get("observations", []):
        metrics = _parse_metrics(item.get("metrics") or [])
        if not any(m.value is not None for m in metrics):
            continue
        observations.append(
            DataObservation(
                timestamp_sec=float(item.get("timestamp_sec", 0)),
                source="transcript",
                metrics=metrics,
                notes=str(item.get("quote") or ""),
            )
        )
    return observations, {"usage": result.usage, "time_info": result.time_info}


def collect_from_frames(
    frames: list[Frame],
    *,
    custom_prompt: str,
    before_call: Callable[[], None] | None = None,
    after_call: Callable[[dict[str, Any]], None] | None = None,
) -> list[DataObservation]:
    selected = sample_frames(frames)
    if not selected:
        return []

    batches = [
        selected[i : i + MOSAIC_MAX_CELLS]
        for i in range(0, len(selected), MOSAIC_MAX_CELLS)
    ]
    observations: list[DataObservation] = []
    for batch in batches:
        if before_call:
            before_call()
        time_range = f"{batch[0].timestamp_sec:.1f}s–{batch[-1].timestamp_sec:.1f}s"
        t0 = time.perf_counter()
        batch_obs, raw = read_frame_batch(
            batch,
            custom_prompt=custom_prompt,
            time_range=time_range,
        )
        wall_sec = time.perf_counter() - t0
        observations.extend(batch_obs)
        if after_call:
            after_call(
                {
                    "stage": "series_collect",
                    "label": f"Frame batch ({len(batch)} cells)",
                    "wall_sec": wall_sec,
                    "usage": raw.get("usage"),
                    "time_info": raw.get("time_info"),
                    "extra": {
                        "source": "image",
                        "frames": len(batch),
                        "observations": len(batch_obs),
                    },
                }
            )
    return observations


def collect_observations(
    *,
    frames: list[Frame],
    transcript_segments: list,
    custom_prompt: str,
    before_call: Callable[[], None] | None = None,
    after_call: Callable[[dict[str, Any]], None] | None = None,
) -> list[DataObservation]:
    """Collect observations from mosaic frames and transcript, then merge."""
    observations: list[DataObservation] = []
    observations.extend(
        collect_from_frames(
            frames,
            custom_prompt=custom_prompt,
            before_call=before_call,
            after_call=after_call,
        )
    )
    if before_call:
        before_call()
    t0 = time.perf_counter()
    text_obs, raw = read_transcript(
        transcript_segments,
        custom_prompt=custom_prompt,
    )
    wall_sec = time.perf_counter() - t0
    observations.extend(text_obs)
    if after_call and text_obs:
        after_call(
            {
                "stage": "series_collect",
                "label": "Transcript extraction",
                "wall_sec": wall_sec,
                "usage": raw.get("usage"),
                "time_info": raw.get("time_info"),
                "extra": {"source": "transcript", "observations": len(text_obs)},
            }
        )
    return merge_observations(observations)