"""Stage 4 — crop the likely code region out of a frame.

Cropping to the code region first means OCR/vision runs only on code, not on facecam, slide
titles, the file tree, or the terminal. Two modes are supported: a fixed config crop (when the
course layout is stable) and a bbox-merge helper that unions text-dense rectangles into a single
region.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from PIL import Image

from vce.types import BBox, Frame


def merge_boxes(boxes: Iterable[BBox]) -> BBox:
    """Return the bounding box that encloses every box in ``boxes``.

    Single pass, so it accepts any iterable (including one-shot iterators/generators).
    Raises ``ValueError`` on empty input — there is no meaningful union of zero boxes.
    """
    iterator = iter(boxes)
    try:
        first = next(iterator)
    except StopIteration:
        raise ValueError("cannot merge an empty sequence of boxes") from None
    left, top = first.x, first.y
    right, bottom = first.x + first.width, first.y + first.height
    for b in iterator:
        left = min(left, b.x)
        top = min(top, b.y)
        right = max(right, b.x + b.width)
        bottom = max(bottom, b.y + b.height)
    return BBox(x=left, y=top, width=right - left, height=bottom - top)


def crop_region(frame: Frame, region: BBox, out_dir: Path) -> Path:
    """Crop ``region`` from ``frame``'s image and write it under ``out_dir``; return the path.

    The output filename encodes the frame's timestamp and the region, so crops of different
    frames (even ones whose paths share a stem) or different regions never collide.
    Raises ``ValueError`` if the region is empty, has negative coordinates, or extends outside
    the image bounds. Validation runs before ``out_dir`` is created, so a failed call never
    leaves an empty directory behind.
    """
    if region.width <= 0 or region.height <= 0:
        raise ValueError(f"region must have positive area, got {region}")
    if region.x < 0 or region.y < 0:
        raise ValueError(f"region coordinates must be non-negative, got {region}")
    with Image.open(frame.path) as img:
        img_w, img_h = img.size
        right, bottom = region.x + region.width, region.y + region.height
        if right > img_w or bottom > img_h:
            raise ValueError(f"region {region} is outside image bounds {img_w}x{img_h}")
        cropped = img.crop((region.x, region.y, right, bottom))
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / (
            f"{frame.path.stem}_{frame.timestamp_ms}"
            f"_{region.x}_{region.y}_{region.width}_{region.height}.png"
        )
        cropped.save(out_path)
    return out_path
