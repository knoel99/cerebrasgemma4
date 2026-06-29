from pathlib import Path

from PIL import Image

from cerebrasgemma4.images import encode_path, validate_batch_size


def test_encode_and_validate_batch(tmp_path: Path):
    path = tmp_path / "img.png"
    Image.new("RGB", (1920, 1080), color=(255, 0, 0)).save(path)
    uri = encode_path(path, detail=False)
    assert uri.startswith("data:image/jpeg;base64,")
    validate_batch_size([uri] * 5)