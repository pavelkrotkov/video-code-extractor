"""Stage 1 — candidate frame extraction (fps sampling + scene-change frames).

Stub: implemented in the "Frame extraction" issue.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from vce.types import Frame


def extract_frames(video: Path, out_dir: Path, *, fps: float = 1.0) -> list[Frame]:
    """Sample ``video`` into timestamped frames under ``out_dir``."""
    raise NotImplementedError("see issue: Frame extraction")


def scene_change_frames(video: Path, out_dir: Path, *, threshold: float = 0.3) -> Iterable[Frame]:
    """Yield frames at detected scene cuts."""
    raise NotImplementedError("see issue: Frame extraction")
