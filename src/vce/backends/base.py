"""The interface every code-extraction backend implements.

Backends are interchangeable so they can be benchmarked head-to-head (see the
backend-benchmark issue). A backend takes a path to an image (a cropped code
region) and returns an :class:`~vce.types.Extraction`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from vce.types import Extraction, Frame


@runtime_checkable
class ExtractionBackend(Protocol):
    """Recover text from a single code-region image."""

    #: Stable identifier used in provenance and benchmark output, e.g. ``"paddleocr"``.
    name: str

    def extract(self, image_path: Path, frame: Frame) -> Extraction:
        """Return the text visible in ``image_path`` for the given ``frame``.

        Implementations MUST act as faithful OCR: transcribe only what is
        visible and never infer or autocomplete missing code.
        """
        ...
