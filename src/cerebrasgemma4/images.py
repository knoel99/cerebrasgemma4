"""Image encoding and resizing for Cerebras multimodal API."""

from __future__ import annotations

import base64
import io
from pathlib import Path

from PIL import Image

MAX_IMAGES_PER_REQUEST = 5
MAX_PAYLOAD_BYTES = 10 * 1024 * 1024

# Approximate target sizes for visual token budgets (Gemma 4 HF docs).
OVERVIEW_MAX_SIDE = 336
DETAIL_MAX_SIDE = 768


def resize_image(path: Path, max_side: int, quality: int = 85) -> bytes:
    with Image.open(path) as img:
        img = img.convert("RGB")
        w, h = img.size
        scale = min(1.0, max_side / max(w, h))
        if scale < 1.0:
            img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        return buf.getvalue()


def encode_data_uri(image_bytes: bytes, mime: str = "image/jpeg") -> str:
    b64 = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime};base64,{b64}"


def encode_path(path: Path, *, detail: bool = False) -> str:
    max_side = DETAIL_MAX_SIDE if detail else OVERVIEW_MAX_SIDE
    return encode_data_uri(resize_image(path, max_side))


def validate_batch_size(data_uris: list[str]) -> None:
    if len(data_uris) > MAX_IMAGES_PER_REQUEST:
        raise ValueError(f"At most {MAX_IMAGES_PER_REQUEST} images per request")
    total = sum(len(u.encode("utf-8")) for u in data_uris)
    if total > MAX_PAYLOAD_BYTES:
        raise ValueError(
            f"Image payload {total} bytes exceeds {MAX_PAYLOAD_BYTES} byte limit"
        )


def build_image_parts(paths: list[Path], *, detail: bool = False) -> list[dict]:
    uris = [encode_path(p, detail=detail) for p in paths]
    validate_batch_size(uris)
    return [{"type": "image_url", "image_url": {"url": u}} for u in uris]