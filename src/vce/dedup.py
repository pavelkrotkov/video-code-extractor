"""Stage 2 — drop near-duplicate frames (perceptual hash / SSIM).

Stub: implemented in the "Frame dedup" issue.
"""

from __future__ import annotations

from collections.abc import Sequence

from vce.types import Frame


def dedup_frames(frames: Sequence[Frame], *, max_distance: int = 4) -> list[Frame]:
    """Return ``frames`` with perceptually near-identical neighbours removed."""
    raise NotImplementedError("see issue: Frame dedup")
