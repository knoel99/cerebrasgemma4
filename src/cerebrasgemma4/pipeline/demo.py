"""Hackathon demo mode: sub-60s end-to-end on long YouTube videos."""

from __future__ import annotations

import os

# Target: full pipeline + report visible in under 60 seconds for demo video.
DEMO_MAX_FRAMES = 6
DEMO_MAX_DETAIL_REGIONS = 4
DEMO_REGION_MOSAIC_MAX = 12
DEMO_SAMPLES_PER_REGION = 4
DEMO_MAX_EXTRACT_FRAMES = 48
DEMO_SCOUT_MAX_HEIGHT = 480
DEMO_PROCESSING_MAX_HEIGHT = 720
DEMO_REPORT_BATCH_WORKERS = 6


def is_demo_mode() -> bool:
    raw = os.getenv(
        "FASTYOUTUBEREPORT_DEMO_MODE",
        os.getenv("SIGHTLINE_DEMO_MODE", os.getenv("VID2DOC_DEMO_MODE", "1")),
    )
    return raw.strip().lower() in {"1", "true", "yes", "on"}