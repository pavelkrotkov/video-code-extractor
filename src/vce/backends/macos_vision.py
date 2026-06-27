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

import math
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from statistics import fmean, median

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


def _group_lines(items: list[tuple[BBox, str, float]]) -> list[list[tuple[BBox, str, float]]]:
    """Group boxes into visual lines, tolerant of small vertical jitter, in reading order.

    Sorting purely by top edge is fragile: two fragments on one visual line (e.g. a statement and a
    trailing inline comment) routinely differ by a pixel of OCR estimation noise, and a strict
    ``(y, x)`` sort would then split them onto separate lines or even swap their order (emitting the
    comment before the code). Instead we walk boxes top-down and attach each to an open line whose
    anchor (its first, topmost box) is vertically close, starting a new line only when none is.

    "Close" is a center-to-center distance within half of the *taller* of the two boxes. Using the
    larger height — rather than the anchor's — is what keeps a short leading glyph (a hyphen, quote,
    or dot that sorts first and would otherwise be a tiny anchor) from splitting the rest of its
    line off. Lines come back top-to-bottom, each ordered left-to-right — the box-derived reading
    order issue #22 asks for, just jitter-tolerant.
    """
    lines: list[list[tuple[BBox, str, float]]] = []
    for item in sorted(items, key=lambda it: (it[0].y, it[0].x)):
        bbox = item[0]
        center = bbox.y + bbox.height / 2
        for line in lines:
            anchor = line[0][0]
            anchor_center = anchor.y + anchor.height / 2
            if abs(center - anchor_center) <= max(anchor.height, bbox.height) * 0.5:
                line.append(item)
                break
        else:  # no open line was vertically close enough — start a new one
            lines.append([item])
    lines.sort(key=lambda line: min(it[0].y for it in line))
    for line in lines:
        line.sort(key=lambda it: it[0].x)
    return lines


def _parse_annotation(entry: object, width: int, height: int) -> tuple[BBox, str, float] | None:
    """Convert one raw Vision annotation to ``(pixel BBox, text, confidence)``; ``None`` if malformed.

    Vision (via ``ocrmac``) yields well-formed :data:`Annotation` tuples, but engine/version drift
    can produce the odd wrong-arity entry, non-numeric confidence, or short bounding box. Rather than
    trust the happy-path type, this validates structurally and returns ``None`` for anything it can't
    honestly read, so one bad annotation is skipped instead of crashing the whole frame. (This is why
    :func:`_to_extraction` accepts ``Sequence[object]``: tolerating malformed input is part of the
    contract, so the parameter is typed as honestly as the body actually treats it.)
    """
    if not isinstance(entry, Sequence) or isinstance(entry, str | bytes) or len(entry) != 3:
        return None
    text_obj, conf_obj, bbox_obj = entry[0], entry[1], entry[2]
    if not isinstance(conf_obj, int | float | str):
        return None
    if (
        not isinstance(bbox_obj, Sequence)
        or isinstance(bbox_obj, str | bytes)
        or len(bbox_obj) != 4
    ):
        return None
    coords: list[float] = []
    for value in bbox_obj:
        # Reject non-finite coords (NaN/inf): they pass the float check but would raise in
        # round() during conversion, aborting the whole frame instead of skipping one bad box.
        if not isinstance(value, int | float) or not math.isfinite(value):
            return None
        coords.append(float(value))
    try:
        confidence = float(conf_obj)
    except ValueError:
        return None
    return (_vision_bbox_to_pixels(coords, width, height), str(text_obj), confidence)


def _estimate_char_width(items: Sequence[tuple[BBox, str, float]]) -> float:
    """Estimate the pixel width of one monospace character from the recognized boxes.

    The median of ``box.width / len(text)`` over non-empty fragments (robust to the odd very short or
    very wide box), falling back to a fraction of the median box height, then to ``1.0``. This sets
    the tolerance for snapping line-start x-positions into indentation columns.
    """
    widths = [bbox.width / len(text) for bbox, text, _ in items if text]
    if widths:
        return median(widths)
    heights = [bbox.height for bbox, _, _ in items if bbox.height > 0]
    if heights:
        return max(median(heights) * 0.6, 1.0)
    return 1.0


def _indent_prefixes(
    lines: Sequence[Sequence[tuple[BBox, str, float]]], char_width: float
) -> list[str]:
    """Reconstruct each visual line's leading indentation from box geometry.

    Each line's leftmost x-edge is snapped to an indentation *column*: distinct left edges more than
    ~one character width apart seed new columns, and each column is rendered as a consistent
    four-space level ranked left-to-right. This turns Vision's pixel offsets back into nested Python
    indentation (deeper code → larger x → deeper level) and normalises whatever step size or
    tab/space mix the source used to clean, parseable four-space levels — while lines sharing a
    column (a block body and the later line that dedents back to it) collapse to the same level.

    Lines all sharing one column (the common single-column case the existing tests exercise) get an
    empty prefix, so flat snippets are unchanged.
    """
    if not lines:
        return []
    lefts = [min(it[0].x for it in line) for line in lines]
    tol = max(char_width, 1.0)
    columns: list[int] = []
    for x in sorted(set(lefts)):
        if not columns or x - columns[-1] > tol:
            columns.append(x)
    # Snap each line to its nearest column (ties -> leftmost) so a left edge near a column boundary
    # can't drift to the wrong level. ``default=0`` makes the invariant explicit: ``columns`` is
    # non-empty whenever ``lefts`` is, so ``min`` never actually sees an empty range here.
    return [
        "    " * min(range(len(columns)), key=lambda i: abs(x - columns[i]), default=0)
        for x in lefts
    ]


def _to_extraction(
    annotations: Sequence[object], frame: Frame, width: int, height: int
) -> Extraction:
    """Map Vision annotations for one image to an :class:`Extraction` in reading order.

    Each box is converted to pixel space, grouped into visual lines (top-to-bottom, fragments within
    a line left-to-right; see :func:`_group_lines`), and given a leading indent reconstructed from
    its left-edge geometry (:func:`_indent_prefixes`) so nested blocks keep their structure instead
    of being flattened to column zero. The result therefore reads in deterministic source order
    regardless of the order Vision returned the annotations. Empty input yields an empty extraction
    with ``confidence == 0.0``; confidence is the mean of the Vision confidences.

    Entries are accepted as ``Sequence[object]`` and validated per-annotation by
    :func:`_parse_annotation`: a single malformed annotation (wrong arity, a non-4 bounding box, a
    non-numeric confidence) is skipped rather than crashing the whole frame.
    """
    converted: list[tuple[BBox, str, float]] = []
    for entry in annotations:
        parsed = _parse_annotation(entry, width, height)
        if parsed is not None:
            converted.append(parsed)
    lines = _group_lines(converted)
    prefixes = _indent_prefixes(lines, _estimate_char_width(converted))
    texts = [
        prefix + " ".join(it[1] for it in line)
        for prefix, line in zip(prefixes, lines, strict=True)
    ]
    confs = [it[2] for it in converted]
    bboxes = tuple(it[0] for line in lines for it in line)
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
            from ocrmac import ocrmac
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
