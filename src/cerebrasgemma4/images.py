"""Image encoding and resizing for Cerebras multimodal API."""

from __future__ import annotations

import base64
import io
import math
from pathlib import Path

from PIL import Image, ImageDraw

MAX_IMAGES_PER_REQUEST = 5
MAX_PAYLOAD_BYTES = 10 * 1024 * 1024

# Approximate target sizes for visual token budgets (Gemma 4 HF docs).
OVERVIEW_MAX_SIDE = 336
DETAIL_MAX_SIDE = 768

MOSAIC_MAX_CELLS = 25
MOSAIC_MIN_CELL_WIDTH = 56
MOSAIC_LABEL_HEIGHT = 28
MOSAIC_CELL_PAD = 2
MOSAIC_BG = (32, 32, 32)
MOSAIC_LABEL_COLOR = (220, 220, 220)


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


def mosaic_grid(cell_count: int) -> tuple[int, int]:
    """Return (cols, rows) for a left-to-right timeline mosaic."""
    if cell_count <= 1:
        return 1, 1
    cols = min(5, max(2, math.ceil(math.sqrt(cell_count * 2))))
    rows = math.ceil(cell_count / cols)
    return cols, rows


def mosaic_layout_dims(
    cell_count: int,
    max_side: int = OVERVIEW_MAX_SIDE,
) -> tuple[int, int, int, int]:
    """Return (cols, rows, cell_w, cell_h) fitting inside max_side."""
    cols, rows = mosaic_grid(cell_count)
    budget = max_side
    for _ in range(8):
        cell_w = (budget - (cols + 1) * MOSAIC_CELL_PAD) // cols
        cell_img_h = int(cell_w * 9 / 16)
        cell_h = cell_img_h + MOSAIC_LABEL_HEIGHT
        mosaic_w = cols * cell_w + (cols + 1) * MOSAIC_CELL_PAD
        mosaic_h = rows * cell_h + (rows + 1) * MOSAIC_CELL_PAD
        if (
            cell_w >= MOSAIC_MIN_CELL_WIDTH
            and mosaic_w <= max_side
            and mosaic_h <= max_side
        ):
            return cols, rows, cell_w, cell_h
        budget -= 8
    cell_w = MOSAIC_MIN_CELL_WIDTH
    cell_img_h = int(cell_w * 9 / 16)
    cell_h = cell_img_h + MOSAIC_LABEL_HEIGHT
    return cols, rows, cell_w, cell_h


def _draw_centered_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    box: tuple[int, int, int, int],
    *,
    fill: tuple[int, int, int],
    valign: str = "center",
) -> None:
    """Draw single- or multi-line text centered horizontally inside a box."""
    left, top, right, bottom = box
    lines = [line.strip() for line in text.replace("\r", "").split("\n") if line.strip()]
    if not lines:
        return

    metrics: list[tuple[int, int]] = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line)
        metrics.append((bbox[2] - bbox[0], bbox[3] - bbox[1]))

    line_gap = 1
    total_h = sum(h for _, h in metrics) + line_gap * max(0, len(lines) - 1)
    if valign == "top":
        y = top + 2
    else:
        y = top + max(0, (bottom - top - total_h) // 2)
    for line, (width, height) in zip(lines, metrics):
        x = left + max(0, (right - left - width) // 2)
        draw.text((x, y), line, fill=fill)
        y += height + line_gap


def build_frame_mosaic(
    paths: list[Path],
    labels: list[str] | None = None,
    *,
    max_side: int = OVERVIEW_MAX_SIDE,
    max_cells: int = MOSAIC_MAX_CELLS,
) -> bytes:
    """Tile frames into one JPEG mosaic sized for the Cerebras overview budget."""
    if not paths:
        raise ValueError("mosaic requires at least one frame")
    if len(paths) > max_cells:
        raise ValueError(f"mosaic supports at most {max_cells} cells, got {len(paths)}")

    labels = labels or [p.stem for p in paths]
    cols, rows, cell_w, cell_h = mosaic_layout_dims(len(paths), max_side)
    cell_img_h = cell_h - MOSAIC_LABEL_HEIGHT
    mosaic_w = cols * cell_w + (cols + 1) * MOSAIC_CELL_PAD
    mosaic_h = rows * cell_h + (rows + 1) * MOSAIC_CELL_PAD

    canvas = Image.new("RGB", (mosaic_w, mosaic_h), MOSAIC_BG)
    draw = ImageDraw.Draw(canvas)

    for idx, path in enumerate(paths):
        col = idx % cols
        row = idx // cols
        x = MOSAIC_CELL_PAD + col * (cell_w + MOSAIC_CELL_PAD)
        y = MOSAIC_CELL_PAD + row * (cell_h + MOSAIC_CELL_PAD)
        img_top = y
        img_bottom = y + cell_img_h
        label_top = img_bottom
        label_bottom = y + cell_h

        with Image.open(path) as img:
            img = img.convert("RGB")
            img.thumbnail((cell_w, cell_img_h), Image.Resampling.LANCZOS)
            off_x = x + (cell_w - img.width) // 2
            off_y = img_top + (cell_img_h - img.height) // 2
            canvas.paste(img, (off_x, off_y))

        _draw_centered_text(
            draw,
            labels[idx][:32],
            (x, label_top, x + cell_w, label_bottom),
            fill=MOSAIC_LABEL_COLOR,
            valign="top",
        )

    buf = io.BytesIO()
    canvas.save(buf, format="JPEG", quality=85, optimize=True)
    return buf.getvalue()


def build_image_parts_bytes(images: list[bytes]) -> list[dict]:
    uris = [encode_data_uri(img) for img in images]
    validate_batch_size(uris)
    return [{"type": "image_url", "image_url": {"url": u}} for u in uris]