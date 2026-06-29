"""Shared data types passed between pipeline stages.

These are intentionally small, immutable value objects so each stage can be
developed and tested in isolation. The pipeline is:

    Frame  -> Candidate -> Extraction -> MergedSnippet

See ``docs/architecture.md`` for the full design.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Frame:
    """A single sampled video frame on disk, tagged with its source timestamp."""

    path: Path
    timestamp_ms: int

    @property
    def timecode(self) -> str:
        """``HH:MM:SS.mmm`` rendering of :attr:`timestamp_ms`."""
        ms = self.timestamp_ms % 1000
        s_total = self.timestamp_ms // 1000
        s = s_total % 60
        m = (s_total // 60) % 60
        h = s_total // 3600
        return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


@dataclass(frozen=True)
class BBox:
    """Axis-aligned pixel bounding box (top-left origin)."""

    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class Candidate:
    """A frame scored for how likely it is to contain code (``0.0``..``1.0``)."""

    frame: Frame
    score: float


@dataclass(frozen=True)
class Extraction:
    """Text recovered from one frame by an :class:`~vce.backends.base.ExtractionBackend`."""

    frame: Frame
    text: str
    confidence: float
    bboxes: tuple[BBox, ...] = ()
    backend: str = ""


@dataclass(frozen=True)
class MergedSnippet:
    """A de-duplicated code snippet merged across one or more source frames."""

    code: str
    sources: tuple[Frame, ...] = field(default_factory=tuple)
    notes: str = ""


@dataclass(frozen=True)
class PipelineStats:
    """Counters and wall-clock timings collected during a Pipeline.run() call."""

    frames_raw: int
    frames_after_dedup: int
    frames_passed_scoring: int
    escalated_count: int
    snippets_merged: int
    output_lines: int
    output_chars: int
    stage_times: tuple[tuple[str, float], ...]
    total_time: float
