"""Performance tracking for pipeline steps and Cerebras API calls."""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


def _as_dict(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, dict):
        return obj
    return dict(obj)


def _num(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_cerebras_timing(usage: dict, time_info: dict) -> dict[str, Any]:
    usage = _as_dict(usage)
    time_info = _as_dict(time_info)

    prompt_tokens = _num(usage.get("prompt_tokens"))
    completion_tokens = _num(usage.get("completion_tokens"))
    total_tokens = _num(usage.get("total_tokens"))

    image_tokens = None
    details = usage.get("prompt_tokens_details")
    if details:
        details = _as_dict(details)
        image_tokens = _num(details.get("image_tokens"))

    ttft = _num(time_info.get("time_to_first_token"))
    if ttft is None:
        ttft = _num(time_info.get("ttft"))
    tpot = _num(time_info.get("time_per_output_token"))
    if tpot is None:
        tpot = _num(time_info.get("tpot"))
    total_time = _num(time_info.get("total_time"))
    if total_time is None:
        total_time = _num(time_info.get("latency"))
    queue_time = _num(time_info.get("queue_time"))
    tokens_per_sec = _num(time_info.get("tokens_per_second"))
    if tokens_per_sec is None and completion_tokens and total_time and total_time > 0:
        gen_time = total_time - (ttft or 0)
        if gen_time > 0:
            tokens_per_sec = completion_tokens / gen_time

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "image_tokens": image_tokens,
        "ttft_ms": round(ttft * 1000, 1) if ttft is not None else None,
        "tpot_ms": round(tpot * 1000, 2) if tpot is not None else None,
        "total_time_sec": round(total_time, 3) if total_time is not None else None,
        "queue_time_sec": round(queue_time, 3) if queue_time is not None else None,
        "output_tokens_per_sec": round(tokens_per_sec, 1) if tokens_per_sec else None,
    }


def normalize_api_call(
    *,
    stage: str,
    label: str,
    wall_sec: float,
    usage: dict | None = None,
    time_info: dict | None = None,
    extra: dict | None = None,
) -> dict[str, Any]:
    timing = extract_cerebras_timing(usage or {}, time_info or {})
    return {
        "stage": stage,
        "label": label,
        "wall_sec": round(wall_sec, 3),
        **timing,
        **(extra or {}),
    }


def summarize_cerebras(calls: list[dict]) -> dict[str, Any]:
    cerebras_calls = [c for c in calls if c.get("stage") in {"scout", "analyze", "compose"}]
    if not cerebras_calls:
        return {
            "calls": 0,
            "wall_sec": 0.0,
            "api_reported_sec": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "image_tokens": 0,
            "avg_ttft_ms": None,
            "avg_output_tokens_per_sec": None,
        }

    wall = sum(c.get("wall_sec", 0) for c in cerebras_calls)
    api_time = sum(c.get("total_time_sec") or 0 for c in cerebras_calls)
    prompt = sum(c.get("prompt_tokens") or 0 for c in cerebras_calls)
    completion = sum(c.get("completion_tokens") or 0 for c in cerebras_calls)
    image = sum(c.get("image_tokens") or 0 for c in cerebras_calls)

    ttfts = [c["ttft_ms"] for c in cerebras_calls if c.get("ttft_ms") is not None]
    tps_vals = [
        c["output_tokens_per_sec"]
        for c in cerebras_calls
        if c.get("output_tokens_per_sec") is not None
    ]
    queue_sec = sum(c.get("queue_time_sec") or 0 for c in cerebras_calls)
    overhead_sec = sum(
        max(
            0.0,
            (c.get("wall_sec") or 0)
            - (c.get("total_time_sec") or 0)
            - (c.get("queue_time_sec") or 0),
        )
        for c in cerebras_calls
    )

    return {
        "calls": len(cerebras_calls),
        "wall_sec": round(wall, 3),
        "api_reported_sec": round(api_time, 3) if api_time else None,
        "queue_sec": round(queue_sec, 3),
        "client_overhead_sec": round(overhead_sec, 3),
        "prompt_tokens": int(prompt),
        "completion_tokens": int(completion),
        "image_tokens": int(image) if image else None,
        "avg_ttft_ms": round(sum(ttfts) / len(ttfts), 1) if ttfts else None,
        "avg_output_tokens_per_sec": round(sum(tps_vals) / len(tps_vals), 1) if tps_vals else None,
        "by_stage": _rollup_by_stage(cerebras_calls),
    }


def _rollup_by_stage(calls: list[dict]) -> dict[str, dict]:
    stages: dict[str, list[dict]] = {}
    for call in calls:
        stages.setdefault(call["stage"], []).append(call)
    out = {}
    for stage, items in stages.items():
        out[stage] = {
            "calls": len(items),
            "wall_sec": round(sum(i.get("wall_sec", 0) for i in items), 3),
            "completion_tokens": int(sum(i.get("completion_tokens") or 0 for i in items)),
            "avg_ttft_ms": (
                round(
                    sum(i["ttft_ms"] for i in items if i.get("ttft_ms") is not None)
                    / max(1, len([i for i in items if i.get("ttft_ms") is not None])),
                    1,
                )
                if any(i.get("ttft_ms") is not None for i in items)
                else None
            ),
        }
    return out


def apply_wall_elapsed(metrics: dict[str, Any], created_at: str | None) -> dict[str, Any]:
    """Align elapsed_sec with wall clock since job creation (includes queue waits)."""
    if not created_at:
        return metrics
    try:
        created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except ValueError:
        return metrics
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    wall = (datetime.now(timezone.utc) - created).total_seconds()
    metrics["wall_elapsed_sec"] = round(max(0.0, wall), 3)
    metrics["elapsed_sec"] = round(
        max(metrics.get("elapsed_sec") or 0, metrics["wall_elapsed_sec"]),
        3,
    )
    return metrics


def save_metrics_file(path: Path, metrics: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return path


@dataclass
class PerfTracker:
    steps: list[dict[str, Any]] = field(default_factory=list)
    api_calls: list[dict[str, Any]] = field(default_factory=list)
    started_at: float = field(default_factory=time.perf_counter)
    prep_elapsed_sec: float = 0.0

    def merge_prep(self, prep: dict[str, Any]) -> None:
        """Attach ingest/download steps recorded before the pipeline starts."""
        self.prep_elapsed_sec = float(prep.get("elapsed_sec") or 0)
        self.steps.extend(prep.get("steps") or [])

    @contextmanager
    def step(self, step_id: str, label: str, *, kind: str = "local", detail: dict | None = None):
        t0 = time.perf_counter()
        entry: dict[str, Any] = {
            "id": step_id,
            "label": label,
            "kind": kind,
            "status": "running",
            "duration_sec": None,
            "detail": detail or {},
        }
        self.steps.append(entry)
        try:
            yield entry
            entry["status"] = "done"
        except Exception:
            entry["status"] = "failed"
            raise
        finally:
            entry["duration_sec"] = round(time.perf_counter() - t0, 3)

    def record_api_call(self, call: dict[str, Any]) -> dict[str, Any]:
        self.api_calls.append(call)
        return call

    def snapshot(self) -> dict[str, Any]:
        pipeline_elapsed = round(time.perf_counter() - self.started_at, 3)
        elapsed = round(self.prep_elapsed_sec + pipeline_elapsed, 3)
        local_sec = round(
            sum(s.get("duration_sec") or 0 for s in self.steps if s.get("kind") == "local"),
            3,
        )
        ingest_sec = round(
            sum(
                s.get("duration_sec") or 0
                for s in self.steps
                if s.get("kind") == "local"
                and s.get("id")
                in {
                    "youtube_metadata",
                    "thumbnail_download",
                    "youtube_download",
                    "file_upload",
                    "video_probe",
                }
            ),
            3,
        )
        cerebras_summary = summarize_cerebras(self.api_calls)
        return {
            "elapsed_sec": elapsed,
            "prep_elapsed_sec": round(self.prep_elapsed_sec, 3),
            "pipeline_elapsed_sec": pipeline_elapsed,
            "ingest_sec": ingest_sec,
            "local_prep_sec": local_sec,
            "cerebras_llm_sec": cerebras_summary["wall_sec"],
            "steps": list(self.steps),
            "api_calls": list(self.api_calls),
            "cerebras": cerebras_summary,
            "model": "gemma-4-31b",
            "provider": "cerebras",
        }