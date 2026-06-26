import pytest
from PIL import Image

from vce.cropping import crop_region, merge_boxes
from vce.types import BBox, Frame


@pytest.fixture
def frame_with_red_square(tmp_path):
    """100x100 white image with a 20x20 red square at (10, 10)."""
    img = Image.new("RGB", (100, 100), "white")
    for y in range(10, 30):
        for x in range(10, 30):
            img.putpixel((x, y), (255, 0, 0))
    path = tmp_path / "frame_000001.jpg"
    img.save(path)
    return Frame(path=path, timestamp_ms=0)


def test_crop_region_writes_expected_size_and_content(frame_with_red_square, tmp_path):
    out = tmp_path / "crops"
    cropped = crop_region(frame_with_red_square, BBox(10, 10, 20, 20), out)
    assert cropped.exists()
    assert cropped.parent == out
    with Image.open(cropped) as img:
        assert img.size == (20, 20)
        # predominantly red — exact (255,0,0) isn't guaranteed through JPEG input
        r, g, b = img.convert("RGB").getpixel((5, 5))
        assert r > 200 and g < 80 and b < 80


def test_crop_region_distinct_names_per_region(frame_with_red_square, tmp_path):
    a = crop_region(frame_with_red_square, BBox(0, 0, 10, 10), tmp_path)
    b = crop_region(frame_with_red_square, BBox(10, 10, 20, 20), tmp_path)
    assert a != b


def test_crop_region_rejects_out_of_bounds(frame_with_red_square, tmp_path):
    with pytest.raises(ValueError):
        crop_region(frame_with_red_square, BBox(90, 90, 50, 50), tmp_path)


def test_crop_region_rejects_zero_area(frame_with_red_square, tmp_path):
    with pytest.raises(ValueError):
        crop_region(frame_with_red_square, BBox(0, 0, 0, 10), tmp_path)


def test_crop_region_rejects_negative_coords(frame_with_red_square, tmp_path):
    with pytest.raises(ValueError):
        crop_region(frame_with_red_square, BBox(-5, 0, 10, 10), tmp_path)


def test_crop_region_leaves_no_dir_on_validation_failure(frame_with_red_square, tmp_path):
    out = tmp_path / "not_created"
    with pytest.raises(ValueError):
        crop_region(frame_with_red_square, BBox(90, 90, 50, 50), out)
    assert not out.exists()


def test_crop_region_distinct_names_across_frames(tmp_path):
    # two frames whose paths share a stem but differ in timestamp must not collide
    img = Image.new("RGB", (50, 50), "white")
    d1, d2 = tmp_path / "a", tmp_path / "b"
    d1.mkdir()
    d2.mkdir()
    img.save(d1 / "frame_000001.jpg")
    img.save(d2 / "frame_000001.jpg")
    # same stem AND same timestamp, different source dir -> must still differ (path digest)
    f1 = Frame(path=d1 / "frame_000001.jpg", timestamp_ms=0)
    f2 = Frame(path=d2 / "frame_000001.jpg", timestamp_ms=0)
    crops = tmp_path / "crops"
    assert crop_region(f1, BBox(0, 0, 10, 10), crops) != crop_region(f2, BBox(0, 0, 10, 10), crops)


def test_merge_boxes_overlapping_returns_union():
    merged = merge_boxes([BBox(0, 0, 10, 10), BBox(5, 5, 10, 10)])
    assert merged == BBox(0, 0, 15, 15)


def test_merge_boxes_disjoint_returns_bounding_box():
    merged = merge_boxes([BBox(0, 0, 2, 2), BBox(10, 20, 5, 5)])
    assert merged == BBox(0, 0, 15, 25)


def test_merge_boxes_single():
    assert merge_boxes([BBox(3, 4, 5, 6)]) == BBox(3, 4, 5, 6)


def test_merge_boxes_accepts_iterator():
    # a one-shot generator must work (single-pass implementation)
    gen = (b for b in [BBox(0, 0, 10, 10), BBox(5, 5, 10, 10)])
    assert merge_boxes(gen) == BBox(0, 0, 15, 15)


def test_merge_boxes_empty_raises():
    with pytest.raises(ValueError):
        merge_boxes([])
