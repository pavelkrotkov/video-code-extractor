"""PaddleOCR extraction backend — the cheap, local option.

PaddleOCR (and its ``paddlepaddle`` runtime) is heavy, so it lives behind the optional ``paddle``
extra and is imported lazily: importing this module never pulls in paddleocr. The engine can also
be injected, which keeps the result→:class:`~vce.types.Extraction` mapping unit-testable without
installing the extra.
"""

from __future__ import annotations

import math
from pathlib import Path
from statistics import fmean
from typing import Any, Protocol

from vce.types import BBox, Extraction, Frame


class _Engine(Protocol):
    def ocr(self, img: str, cls: bool = ...) -> Any: ...


def _poly_to_bbox(points: list[list[float]]) -> BBox:
    """Convert PaddleOCR's 4-point polygon to an axis-aligned :class:`BBox`.

    Uses floor for the top-left and ceil for the bottom-right so the box fully encloses the
    detected text rather than clipping fractional edges. Returns an empty box for empty input.
    """
    if not points:
        return BBox(x=0, y=0, width=0, height=0)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    left, top = math.floor(min(xs)), math.floor(min(ys))
    right, bottom = math.ceil(max(xs)), math.ceil(max(ys))
    return BBox(x=left, y=top, width=right - left, height=bottom - top)


def _to_extraction(raw: Any, frame: Frame) -> Extraction:
    """Map PaddleOCR's nested result for a single image to an :class:`Extraction`.

    Empty results map to an empty extraction. ``raw[0]`` is guarded by the falsy check on
    ``raw`` (an empty list short-circuits, so the index is never evaluated). Individual entries
    with an unexpected shape are skipped rather than crashing the whole frame, since PaddleOCR's
    result structure varies across versions.
    """
    page = raw[0] if raw else None
    texts: list[str] = []
    confs: list[float] = []
    bboxes: list[BBox] = []
    for entry in page or []:
        try:
            polygon, (text, conf) = entry[0], entry[1]
            bbox, conf_value = _poly_to_bbox(polygon), float(conf)
        except (ValueError, IndexError, TypeError):
            continue  # skip a malformed entry, keep the rest of the frame
        # append together so a partial failure never misaligns the parallel lists
        texts.append(text)
        confs.append(conf_value)
        bboxes.append(bbox)
    return Extraction(
        frame=frame,
        text="\n".join(texts),
        confidence=fmean(confs) if confs else 0.0,
        bboxes=tuple(bboxes),
        backend="paddleocr",
    )


class PaddleOCRBackend:
    """:class:`~vce.backends.base.ExtractionBackend` backed by PaddleOCR."""

    name = "paddleocr"

    def __init__(self, *, lang: str = "en", engine: _Engine | None = None) -> None:
        self._lang = lang
        self._engine = engine

    def _get_engine(self) -> _Engine:
        if self._engine is None:
            try:
                from paddleocr import PaddleOCR  # ty: ignore[unresolved-import]
            except ImportError as exc:  # pragma: no cover - exercised via monkeypatched import
                raise ImportError(
                    "PaddleOCR is not installed. Install the optional extra: "
                    "uv sync --extra paddle  (or pip install 'video-code-extractor[paddle]')"
                ) from exc
            self._engine = PaddleOCR(use_angle_cls=True, lang=self._lang, show_log=False)
        return self._engine

    def extract(self, image_path: Path, frame: Frame) -> Extraction:
        raw = self._get_engine().ocr(str(image_path), cls=True)
        return _to_extraction(raw, frame)
