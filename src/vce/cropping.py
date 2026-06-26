"""Stage 4 — crop the likely code region out of a frame.

Stub: implemented in the "Region cropping" issue.
"""

from __future__ import annotations

from pathlib import Path

from vce.types import BBox, Frame


def crop_region(frame: Frame, region: BBox, out_dir: Path) -> Path:
    """Crop ``region`` from ``frame`` and write it under ``out_dir``; return the path."""
    raise NotImplementedError("see issue: Region cropping")
