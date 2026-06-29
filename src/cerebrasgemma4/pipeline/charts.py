"""Render charts from collected observations."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from cerebrasgemma4.pipeline.gemma.series import (
    DataObservation,
    MetricValue,
    metric_definitions,
)


@dataclass
class ChartAsset:
    asset_name: str
    metric_key: str
    label: str
    unit: str


def _format_timestamp(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _metric_points(
    observations: list[DataObservation],
    metric_key: str,
) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for obs in observations:
        for metric in obs.metrics:
            if metric.key == metric_key and metric.value is not None:
                points.append((obs.timestamp_sec, float(metric.value)))
    return sorted(points, key=lambda p: p[0])


def _chart_filename(metric_key: str) -> str:
    safe = re.sub(r"[^a-z0-9_]+", "_", metric_key.lower()).strip("_") or "metric"
    return f"chart_{safe}.png"


def _axis_label(series: MetricValue) -> str:
    if series.unit:
        return f"{series.label} ({series.unit})"
    return series.label


def render_charts(
    observations: list[DataObservation],
    assets_dir: Path,
) -> list[ChartAsset]:
    """Build one PNG per metric that has at least two points."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    assets_dir.mkdir(parents=True, exist_ok=True)
    created: list[ChartAsset] = []
    for series in metric_definitions(observations):
        points = _metric_points(observations, series.key)
        if len(points) < 2:
            continue
        xs = [p[0] / 60.0 for p in points]
        ys = [p[1] for p in points]
        fig, ax = plt.subplots(figsize=(8, 3.2), dpi=120)
        ax.plot(xs, ys, color="#e85d04", linewidth=2, marker="o", markersize=3)
        ax.set_xlabel("Time (minutes)")
        ax.set_ylabel(_axis_label(series))
        ax.set_title(f"{series.label} over time")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        filename = _chart_filename(series.key)
        fig.savefig(assets_dir / filename, bbox_inches="tight")
        plt.close(fig)
        created.append(
            ChartAsset(
                asset_name=filename,
                metric_key=series.key,
                label=series.label,
                unit=series.unit,
            )
        )
    return created


def _format_metric_cell(metric: MetricValue | None) -> str:
    if metric is None or metric.value is None:
        return "—"
    if abs(metric.value) >= 1000:
        return f"{metric.value:,.2f}"
    if abs(metric.value) >= 10:
        return f"{metric.value:,.1f}"
    return f"{metric.value:,.3f}"


def format_table(observations: list[DataObservation]) -> str:
    series = metric_definitions(observations)
    if not series or not observations:
        return "_No structured numeric observations were collected._"

    headers = ["Time", "Source"] + [
        f"{s.label}" + (f" ({s.unit})" if s.unit else "") for s in series
    ]
    align = ["---"] + ["---"] + ["---:" for _ in series]
    rows = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(align) + " |",
    ]
    for obs in observations:
        by_key = {m.key: m for m in obs.metrics}
        cells = [
            _format_timestamp(obs.timestamp_sec),
            obs.source,
            *[_format_metric_cell(by_key.get(s.key)) for s in series],
        ]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def format_section(
    observations: list[DataObservation],
    charts: list[ChartAsset],
) -> str:
    if not observations and not charts:
        return ""

    numeric_rows = sum(
        1 for o in observations if any(m.value is not None for m in o.metrics)
    )
    parts = [
        "Include a ## Data & charts section (or localized equivalent) when this block is present.",
        "Use the table and chart images below; do not invent values.",
        "",
        format_table(observations),
    ]
    if charts:
        parts.append("")
        for chart in charts:
            caption = chart.label + (f" ({chart.unit})" if chart.unit else "")
            parts.append(f"![{caption} over time](assets/{chart.asset_name})")
    parts.append("")
    parts.append(
        f"_{len(observations)} observations ({numeric_rows} with numbers) "
        "from frames and/or transcript._"
    )
    return "\n".join(parts)