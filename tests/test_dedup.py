import pytest

from vce.dedup import _phash, dedup_frames
from vce.types import Frame


@pytest.fixture
def frames(tmp_path):
    """Three frames where A and B are near-identical and C is clearly different.

    Returns ``(A, B, C)`` Frames pointing at PNGs on disk. The images carry real structure
    (rendered text) because a perceptual hash is meaningless on a flat fill: A and B differ only
    by a tiny stroke (Hamming distance <= the default threshold) while C shows different content.
    """
    from PIL import Image, ImageDraw

    def render(name, lines, *, extra_stroke=False):
        img = Image.new("RGB", (128, 128), "white")
        draw = ImageDraw.Draw(img)
        for i, line in enumerate(lines):
            draw.text((4, 4 + i * 12), line, fill="black")
        if extra_stroke:
            draw.line((4, 60, 9, 60), fill="black")  # tiny visible change vs the base image
        path = tmp_path / name
        img.save(path)
        return path

    base_lines = ["def foo():", "    return 1", "x = 2", "y = 3"]
    other_lines = ["class Bar:", "  pass", "import os", "print(1)", "# different", "a=[1,2,3]"]
    a = render("a.png", base_lines)
    b = render("b.png", base_lines, extra_stroke=True)  # ~2 bits from A: a near-duplicate
    c = render("c.png", other_lines)  # clearly different content
    return (
        Frame(path=a, timestamp_ms=0),
        Frame(path=b, timestamp_ms=1000),
        Frame(path=c, timestamp_ms=2000),
    )


def test_drops_near_identical_neighbour(frames):
    a, b, c = frames
    kept = dedup_frames([a, b, c])
    assert kept == [a, c]


def test_preserves_original_timestamps(frames):
    a, _b, c = frames
    kept = dedup_frames(frames)
    assert [f.timestamp_ms for f in kept] == [a.timestamp_ms, c.timestamp_ms]


def test_identical_list_collapses_to_one(frames):
    a, _b, _c = frames
    kept = dedup_frames([a, a, a])
    assert kept == [a]


def test_empty_returns_empty():
    assert dedup_frames([]) == []


def test_keeps_first_of_each_run(frames):
    a, b, c = frames
    # A, B near-dupes -> keep A; C distinct -> keep C; B again near C? no, near A -> dropped run
    kept = dedup_frames([a, b, b, c, a])
    # runs: [a, b, b] -> a ; then c is distinct from last kept (a) -> c ; then a distinct from c -> a
    assert kept == [a, c, a]


def test_hash_func_is_injectable(frames):
    a, b, c = frames
    calls = []

    def fake_hash(frame):
        calls.append(frame)
        # constant hash so every frame collapses to the first, without disk I/O
        return 0

    kept = dedup_frames([a, b, c], hash_func=fake_hash)
    assert kept == [a]
    assert calls == [a, b, c]


def test_tight_max_distance_keeps_near_duplicate(frames):
    a, b, c = frames
    # A and B sit ~2 bits apart; a sub-2 threshold no longer treats B as a duplicate of A.
    kept = dedup_frames([a, b, c], max_distance=1)
    assert kept == [a, b, c]


def test_phash_raises_clear_error_for_missing_file(tmp_path):
    missing = Frame(path=tmp_path / "does-not-exist.png", timestamp_ms=0)
    with pytest.raises(ValueError, match="cannot open frame image"):
        _phash(missing)


def test_negative_max_distance_raises(frames):
    with pytest.raises(ValueError, match="non-negative"):
        dedup_frames(frames, max_distance=-1)
