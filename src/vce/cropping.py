"""Stage 4 — crop the likely code region out of a frame.

Cropping to the code region first means OCR/vision runs only on code, not on facecam, slide
titles, the file tree, or the terminal. Two modes are supported: a fixed config crop (when the
course layout is stable) and a bbox-merge helper that unions text-dense rectangles into a single
region.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from PIL import Image

from vce.types import BBox, Frame


def merge_boxes(boxes: Sequence[BBox]) -> BBox:
    """Return the bounding box that encloses every box in ``boxes``.

    Raises ``ValueError`` on empty input — there is no meaningful union of zero boxes.
    """
    if not boxes:
        raise ValueError("cannot merge an empty sequence of boxes")
    left = min(b.x for b in boxes)
    top = min(b.y for b in boxes)
    right = max(b.x + b.width for b in boxes)
    bottom = max(b.y + b.height for b in boxes)
    return BBox(x=left, y=top, width=right - left, height=bottom - top)


def crop_region(frame: Frame, region: BBox, out_dir: Path) -> Path:
    """Crop ``region`` from ``frame``'s image and write it under ``out_dir``; return the path.

    The output filename encodes the region so distinct crops of the same frame never collide.
    Raises ``ValueError`` if the region is empty or extends outside the image bounds.
    """
    if region.width <= 0 or region.height <= 0:
        raise ValueError(f"region must have positive area, got {region}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(frame.path) as img:
        img_w, img_h = img.size
        right, bottom = region.x + region.width, region.y + region.height
        if region.x < 0 or region.y < 0 or right > img_w or bottom > img_h:
            raise ValueError(f"region {region} is outside image bounds {img_w}x{img_h}")
        cropped = img.crop((region.x, region.y, right, bottom))
        out_path = out_dir / (
            f"{frame.path.stem}_{region.x}_{region.y}_{region.width}_{region.height}.png"
        )
        cropped.save(out_path)
    return out_path
