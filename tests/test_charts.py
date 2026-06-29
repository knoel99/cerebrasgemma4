from pathlib import Path

from cerebrasgemma4.pipeline.charts import (
    ChartAsset,
    format_section,
    format_table,
    render_charts,
)
from cerebrasgemma4.pipeline.gemma.series import DataObservation, MetricValue


def test_render_charts_creates_pngs(tmp_path: Path):
    observations = [
        DataObservation(
            0.0,
            "image",
            metrics=[MetricValue("count", "Count", 0.0, "")],
            frame_id="f_0000",
        ),
        DataObservation(
            60.0,
            "image",
            metrics=[MetricValue("count", "Count", 120.0, "")],
            frame_id="f_0060",
        ),
        DataObservation(
            120.0,
            "transcript",
            metrics=[MetricValue("count", "Count", 500.0, "")],
        ),
    ]
    charts = render_charts(observations, tmp_path / "assets")
    assert len(charts) == 1
    assert (tmp_path / "assets" / charts[0].asset_name).is_file()


def test_format_section_includes_table_and_images():
    observations = [
        DataObservation(
            0.0,
            "image",
            metrics=[
                MetricValue("price", "Price", 10.0, "USD"),
                MetricValue("qty", "Quantity", 2.0, ""),
            ],
            frame_id="f_0000",
        ),
        DataObservation(
            60.0,
            "transcript",
            metrics=[MetricValue("price", "Price", 20.0, "USD")],
        ),
    ]
    md = format_section(
        observations,
        [ChartAsset("chart_price.png", "price", "Price", "USD")],
    )
    assert "chart_price.png" in md
    assert "Price (USD)" in md
    assert "observations" in md


def test_format_table_handles_missing_values():
    table = format_table(
        [
            DataObservation(
                1.0,
                "image",
                metrics=[MetricValue("a", "A", None, ""), MetricValue("b", "B", 3.0, "")],
                frame_id="f_0001",
            )
        ]
    )
    assert "—" in table
    assert "3.0" in table
    assert "image" in table