"""Re-collect observations from persisted frames and transcript (post-report chat)."""

from __future__ import annotations

from pathlib import Path

from cerebrasgemma4.pipeline.charts import format_section, render_charts
from cerebrasgemma4.pipeline.context import VideoContext, save_context
from cerebrasgemma4.pipeline.frames import load_scout_frames_from_dir
from cerebrasgemma4.pipeline.gemma.chat import ChatEnrichment, ChatTurnResult
from cerebrasgemma4.pipeline.gemma.series import (
    DataObservation,
    collect_observations,
    prompt_requests_charts,
)


def _load_observations(ctx: VideoContext | None) -> list[DataObservation]:
    if ctx is None:
        return []
    return [DataObservation.from_dict(item) for item in ctx.observations]


def try_charts_chat_turn(
    job_dir: Path,
    user_message: str,
    ctx: VideoContext | None,
) -> ChatTurnResult | None:
    if not prompt_requests_charts(user_message):
        return None

    frames = load_scout_frames_from_dir(job_dir / "frames")
    segments = ctx.transcript_segments if ctx else []
    if frames or segments:
        observations = collect_observations(
            frames=frames,
            transcript_segments=segments,
            custom_prompt=user_message,
        )
    else:
        observations = _load_observations(ctx)

    if not observations or not any(
        m.value is not None for o in observations for m in o.metrics
    ):
        return ChatTurnResult(
            reply=(
                "I could not collect enough numeric observations from the stored "
                "frames or transcript. Try regenerating with instructions that mention "
                "charts or metrics."
            ),
            enrichment=None,
            usage={},
            time_info={},
        )

    assets_dir = job_dir / "assets"
    charts = render_charts(observations, assets_dir)
    markdown = format_section(observations, charts)
    if ctx is not None:
        save_context(
            job_dir,
            VideoContext(
                source_name=ctx.source_name,
                duration_sec=ctx.duration_sec,
                transcript_source=ctx.transcript_source,
                transcript_full_text=ctx.transcript_full_text,
                transcript_segments=ctx.transcript_segments,
                analyses=ctx.analyses,
                observations=[o.to_dict() for o in observations],
            ),
        )

    numeric = sum(
        1 for o in observations if any(m.value is not None for m in o.metrics)
    )
    reply = (
        f"Collected {len(observations)} observations ({numeric} with numbers) and "
        f"built {len(charts)} chart(s). Apply the section to add the table and "
        "graphs to your report."
    )
    return ChatTurnResult(
        reply=reply,
        enrichment=ChatEnrichment(
            title="Data & charts",
            markdown=markdown,
        ),
        usage={},
        time_info={},
    )