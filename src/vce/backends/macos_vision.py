"""Apple Vision extraction backend — the cheap, local option on macOS.

Apple's Vision framework (``VNRecognizeTextRequest``) ships with macOS, runs fully on-device, and
exposes recognized text, per-line confidence, and bounding boxes. We reach it through the
[`ocrmac`](https://pypi.org/project/ocrmac/) wrapper. ``ocrmac`` (and the PyObjC bridge under it)
is macOS-only, so it is imported lazily: importing this module never pulls in Apple frameworks and
the package stays importable and testable on Linux CI. The recognizer can also be injected, which
keeps the annotation→:class:`~vce.types.Extraction` mapping unit-testable without a real OCR call.

Vision reports each box in *normalized*, *bottom-left-origin* coordinates; downstream stages expect
pixel-space, top-left-origin :class:`~vce.types.BBox` values, so the conversion (and the vertical
flip it implies) lives in :func:`_vision_bbox_to_pixels`.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from statistics import fmean

from vce.types import BBox, Extraction, Frame

#: One Vision annotation: ``(text, confidence, (x, y, width, height))`` with the bounding box in
#: normalized [0, 1] coordinates and a bottom-left origin (Apple's convention).
Annotation = tuple[str, float, Sequence[float]]

#: A recognizer turns an image path into Vision annotations. The real one wraps ``ocrmac``; tests
#: inject a fake so the mapping is exercised without macOS.
Recognizer = Callable[[Path], Sequence[Annotation]]


class UnsupportedPlatformError(RuntimeError):
    """Raised when the macOS-only Vision backend is used on a non-macOS host."""


def _vision_bbox_to_pixels(norm_bbox: Sequence[float], width: int, height: int) -> BBox:
    """Convert one normalized, bottom-left-origin Vision box to a pixel, top-left-origin ``BBox``.

    Vision gives ``(x, y, w, h)`` as fractions of the image with the origin at the bottom-left, so
    the box's top edge in top-left pixel space is ``(1 - (y + h)) * height``. We clamp each edge to
    the unit square first (so a box flush with an edge can't spill past it) and round to the nearest
    pixel — OCR boxes are already approximate, and rounding keeps clean fractional inputs landing on
    clean pixels rather than drifting a pixel under floating-point error.
    """
    x, y, w, h = norm_bbox
    left = min(max(x, 0.0), 1.0) * width
    right = min(max(x + w, 0.0), 1.0) * width
    # Flip the vertical axis: Vision's bottom-left origin -> our top-left origin.
    top = min(max(1.0 - (y + h), 0.0), 1.0) * height
    bottom = min(max(1.0 - y, 0.0), 1.0) * height
    ileft, iright = round(left), round(right)
    itop, ibottom = round(top), round(bottom)
    return BBox(x=ileft, y=itop, width=max(0, iright - ileft), height=max(0, ibottom - itop))


def _to_extraction(
    annotations: Sequence[Annotation], frame: Frame, width: int, height: int
) -> Extraction:
    """Map Vision annotations for one image to an :class:`Extraction` in reading order.

    Each annotation is line-level, so we convert every box to pixel space and order them
    deterministically top-to-bottom then left-to-right. Empty input yields an empty extraction with
    ``confidence == 0.0``. Confidence is the mean of the per-line Vision confidences.
    """
    converted: list[tuple[BBox, str, float]] = []
    for text, conf, norm_bbox in annotations:
        bbox = _vision_bbox_to_pixels(norm_bbox, width, height)
        converted.append((bbox, str(text), float(conf)))
    # Reading order from the boxes themselves so a shuffled annotation list is reconstructed
    # deterministically: top edge first, then left edge to break ties on the same line.
    converted.sort(key=lambda c: (c[0].y, c[0].x))
    texts = [c[1] for c in converted]
    confs = [c[2] for c in converted]
    bboxes = tuple(c[0] for c in converted)
    return Extraction(
        frame=frame,
        text="\n".join(texts),
        confidence=fmean(confs) if confs else 0.0,
        bboxes=bboxes,
        backend="macos-vision",
    )


def _image_size(image_path: Path) -> tuple[int, int]:
    """Return ``(width, height)`` of ``image_path`` in pixels (Pillow is a base dependency)."""
    from PIL import Image

    with Image.open(image_path) as img:
        return img.width, img.height


class MacOSVisionBackend:
    """:class:`~vce.backends.base.ExtractionBackend` backed by Apple Vision OCR via ``ocrmac``."""

    name = "macos-vision"

    def __init__(
        self,
        *,
        language_preference: Sequence[str] = ("en-US",),
        recognition_level: str = "accurate",
        recognizer: Recognizer | None = None,
    ) -> None:
        self._language_preference = list(language_preference)
        self._recognition_level = recognition_level
        self._recognizer = recognizer

    def _recognize(self, image_path: Path) -> Sequence[Annotation]:
        if self._recognizer is not None:
            return self._recognizer(image_path)
        if sys.platform != "darwin":
            raise UnsupportedPlatformError(
                "the macos-vision backend requires macOS; on this platform run with "
                "--backend vision-gpt4v (needs OPENAI_API_KEY)"
            )
        try:
            from ocrmac import ocrmac  # ty: ignore[unresolved-import]
        except ImportError as exc:  # pragma: no cover - exercised via monkeypatched import
            raise ImportError(
                "ocrmac is required for the macos-vision backend on macOS: pip install ocrmac"
            ) from exc
        return ocrmac.OCR(
            str(image_path),
            recognition_level=self._recognition_level,
            language_preference=self._language_preference,
        ).recognize()

    def extract(self, image_path: Path, frame: Frame) -> Extraction:
        annotations = self._recognize(image_path)
        width, height = _image_size(image_path)
        return _to_extraction(annotations, frame, width, height)
