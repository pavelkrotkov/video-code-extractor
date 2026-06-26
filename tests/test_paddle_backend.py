import sys
from pathlib import Path

import pytest

from vce.backends.base import ExtractionBackend
from vce.backends.paddle import PaddleOCRBackend, _poly_to_bbox, _to_extraction
from vce.types import BBox, Frame

FRAME = Frame(path=Path("f.jpg"), timestamp_ms=0)


class FakeEngine:
    """Stands in for a real PaddleOCR engine, returning its classic nested format."""

    def __init__(self, result):
        self._result = result

    def ocr(self, img, cls=True):
        return self._result


PADDLE_RESULT = [
    [
        [[[10, 10], [50, 10], [50, 30], [10, 30]], ("import jax", 0.98)],
        [[[10, 40], [80, 40], [80, 60], [10, 60]], ("def f():", 0.90)],
    ]
]


def test_satisfies_backend_protocol():
    assert isinstance(PaddleOCRBackend(), ExtractionBackend)
    assert PaddleOCRBackend().name == "paddleocr"


def test_poly_to_bbox():
    assert _poly_to_bbox([[10, 10], [50, 10], [50, 30], [10, 30]]) == BBox(10, 10, 40, 20)


def test_to_extraction_maps_lines_conf_and_boxes():
    ext = _to_extraction(PADDLE_RESULT, FRAME)
    assert ext.text == "import jax\ndef f():"
    assert ext.backend == "paddleocr"
    assert ext.confidence == pytest.approx((0.98 + 0.90) / 2)
    assert ext.bboxes == (BBox(10, 10, 40, 20), BBox(10, 40, 70, 20))


def test_to_extraction_handles_empty():
    ext = _to_extraction([None], FRAME)
    assert ext.text == ""
    assert ext.confidence == 0.0
    assert ext.bboxes == ()


def test_extract_uses_injected_engine():
    backend = PaddleOCRBackend(engine=FakeEngine(PADDLE_RESULT))
    ext = backend.extract(Path("crop.png"), FRAME)
    assert ext.text == "import jax\ndef f():"


def test_extract_without_paddle_installed_raises(monkeypatch):
    # Force `from paddleocr import PaddleOCR` to fail regardless of the environment.
    monkeypatch.setitem(sys.modules, "paddleocr", None)
    with pytest.raises(ImportError, match="paddle"):
        PaddleOCRBackend().extract(Path("crop.png"), FRAME)


@pytest.mark.paddle
def test_real_paddleocr_reads_rendered_text(tmp_path):
    """Integration check against a real PaddleOCR install (skipped without the extra)."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (240, 60), "white")
    ImageDraw.Draw(img).text((5, 20), "import jax", fill="black")
    path = tmp_path / "code.png"
    img.save(path)

    ext = PaddleOCRBackend().extract(path, FRAME)
    assert "import" in ext.text.lower()
