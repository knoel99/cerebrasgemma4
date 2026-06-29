import io
from pathlib import Path

from PIL import Image

from cerebrasgemma4.images import (
    OVERVIEW_MAX_SIDE,
    build_frame_mosaic,
    encode_path,
    mosaic_grid,
    mosaic_layout_dims,
    validate_batch_size,
)


def test_encode_and_validate_batch(tmp_path: Path):
    path = tmp_path / "img.png"
    Image.new("RGB", (1920, 1080), color=(255, 0, 0)).save(path)
    uri = encode_path(path, detail=False)
    assert uri.startswith("data:image/jpeg;base64,")
    validate_batch_size([uri] * 5)


def test_mosaic_grid_prefers_wide_layout():
    assert mosaic_grid(5) == (4, 2)
    assert mosaic_grid(25) == (5, 5)


def test_mosaic_fits_overview_budget(tmp_path: Path):
    paths = []
    for i in range(6):
        path = tmp_path / f"f_{i}.jpg"
        Image.new("RGB", (640, 360), color=(i * 20, 40, 80)).save(path)
        paths.append(path)
    cols, rows, cell_w, cell_h = mosaic_layout_dims(len(paths))
    mosaic = build_frame_mosaic(paths, [f"f_{i}" for i in range(6)])
    with Image.open(io.BytesIO(mosaic)) as img:
        assert img.width <= OVERVIEW_MAX_SIDE
        assert img.height <= OVERVIEW_MAX_SIDE
        assert cols * rows >= 6
        assert cell_w >= 56