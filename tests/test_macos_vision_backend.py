import sys
from pathlib import Path

import pytest

from vce.backends.base import ExtractionBackend
from vce.backends.macos_vision import (
    MacOSVisionBackend,
    UnsupportedPlatformError,
    _to_extraction,
    _vision_bbox_to_pixels,
)
from vce.types import BBox, Frame

FRAME = Frame(path=Path("f.jpg"), timestamp_ms=0)

# Two lines of code, top-to-bottom. Vision boxes are normalized with a bottom-left origin, so the
# first ("higher up") line has the larger y. Image is 100x100 for easy mental arithmetic.
VISION_ANNOTATIONS = [
    ("import jax", 0.98, (0.1, 0.8, 0.4, 0.1)),  # y in [0.8, 0.9] from bottom -> top of image
    ("def f():", 0.90, (0.1, 0.6, 0.3, 0.1)),  # y in [0.6, 0.7] from bottom -> below the first
]

# Pixel boxes the annotations above map to in a 100x100 image (top-left origin), in reading order.
EXPECTED_BBOXES = (BBox(10, 10, 40, 10), BBox(10, 30, 30, 10))


def _render_png(tmp_path: Path, size: tuple[int, int] = (100, 100)) -> Path:
    from PIL import Image

    path = tmp_path / "crop.png"
    Image.new("RGB", size, "white").save(path)
    return path


def test_satisfies_backend_protocol():
    assert isinstance(MacOSVisionBackend(), ExtractionBackend)
    assert MacOSVisionBackend().name == "macos-vision"


def test_importing_module_does_not_pull_in_apple_frameworks():
    # The module must stay importable on Linux: no ocrmac/PyObjC at import time.
    assert "ocrmac" not in sys.modules


# --- coordinate conversion (pure helper) --------------------------------------------------


def test_vision_bbox_to_pixels_flips_origin_and_scales():
    # x in [0.1, 0.5], y in [0.8, 0.9] from the bottom of a 100x100 image. Top-left origin:
    #   left=10, right=50; top=(1-0.9)*100=10, bottom=(1-0.8)*100=20.
    assert _vision_bbox_to_pixels((0.1, 0.8, 0.4, 0.1), 100, 100) == BBox(10, 10, 40, 10)


def test_vision_bbox_to_pixels_rounds_subpixel_edges():
    # Sub-pixel edges round to the nearest pixel: left 12.6 -> 13, right 52.6 -> 53.
    box = _vision_bbox_to_pixels((0.126, 0.8, 0.4, 0.1), 100, 100)
    assert box == BBox(13, 10, 40, 10)


def test_vision_bbox_to_pixels_clamps_to_image_bounds():
    # A box that overflows past the right/bottom edges is clamped, not allowed to spill over.
    box = _vision_bbox_to_pixels((0.9, 0.0, 0.5, 0.5), 100, 100)
    assert box == BBox(90, 50, 10, 50)


def test_vision_bbox_to_pixels_full_frame():
    assert _vision_bbox_to_pixels((0.0, 0.0, 1.0, 1.0), 200, 120) == BBox(0, 0, 200, 120)


# --- annotation mapping -------------------------------------------------------------------


def test_to_extraction_maps_lines_conf_and_boxes_in_reading_order():
    ext = _to_extraction(VISION_ANNOTATIONS, FRAME, 100, 100)
    assert ext.text == "import jax\ndef f():"
    assert ext.backend == "macos-vision"
    assert ext.confidence == pytest.approx((0.98 + 0.90) / 2)
    assert ext.bboxes == EXPECTED_BBOXES


def test_to_extraction_reconstructs_order_from_shuffled_boxes():
    # Lines on distinct rows handed over scrambled must come back top-to-bottom.
    shuffled = [
        ("third", 0.9, (0.1, 0.4, 0.2, 0.05)),
        ("first", 0.9, (0.1, 0.8, 0.2, 0.05)),
        ("fourth", 0.9, (0.1, 0.2, 0.2, 0.05)),
        ("second", 0.9, (0.1, 0.6, 0.2, 0.05)),
    ]
    ext = _to_extraction(shuffled, FRAME, 100, 100)
    assert ext.text == "first\nsecond\nthird\nfourth"


def test_to_extraction_groups_same_line_fragments_left_to_right():
    # Two boxes on one visual line (code + trailing comment) join into a single line, ordered
    # left-to-right — even though the comment box sits a pixel *higher* (smaller y) than the code.
    # A strict (y, x) sort would emit "# note" first and split them; line-grouping must not.
    annotations = [
        ("# note", 0.9, (0.5, 0.81, 0.2, 0.05)),  # to the right, ~1px higher
        ("x = 1", 0.9, (0.1, 0.80, 0.2, 0.05)),  # to the left
        ("y = 2", 0.9, (0.1, 0.60, 0.2, 0.05)),  # next line down
    ]
    ext = _to_extraction(annotations, FRAME, 100, 100)
    assert ext.text == "x = 1 # note\ny = 2"


def test_to_extraction_empty_returns_empty_extraction():
    ext = _to_extraction([], FRAME, 100, 100)
    assert ext.text == ""
    assert ext.confidence == 0.0
    assert ext.bboxes == ()


# --- backend wiring -----------------------------------------------------------------------


def test_extract_uses_injected_recognizer(tmp_path):
    png = _render_png(tmp_path)
    backend = MacOSVisionBackend(recognizer=lambda _path: VISION_ANNOTATIONS)
    ext = backend.extract(png, FRAME)
    assert ext.text == "import jax\ndef f():"
    assert ext.bboxes == EXPECTED_BBOXES


def test_extract_empty_recognition(tmp_path):
    png = _render_png(tmp_path)
    ext = MacOSVisionBackend(recognizer=lambda _path: []).extract(png, FRAME)
    assert ext.text == ""
    assert ext.confidence == 0.0


def test_extract_on_non_macos_without_recognizer_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", "linux")
    with pytest.raises(UnsupportedPlatformError, match="requires macOS"):
        MacOSVisionBackend().extract(_render_png(tmp_path), FRAME)


def test_extract_without_ocrmac_installed_raises(monkeypatch, tmp_path):
    # On macOS, a missing ocrmac surfaces a clear install message rather than a bare ImportError.
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setitem(sys.modules, "ocrmac", None)
    with pytest.raises(ImportError, match="ocrmac"):
        MacOSVisionBackend().extract(_render_png(tmp_path), FRAME)


@pytest.mark.skipif(sys.platform != "darwin", reason="requires macOS Apple Vision")
@pytest.mark.macos
def test_real_apple_vision_reads_rendered_text(tmp_path):
    """Integration check against real Apple Vision OCR (skipped off macOS)."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (320, 80), "white")
    ImageDraw.Draw(img).text((10, 30), "import jax", fill="black")
    path = tmp_path / "code.png"
    img.save(path)

    ext = MacOSVisionBackend().extract(path, FRAME)
    assert "import" in ext.text.lower()
