"""Smoke tests: the package imports and shared types behave. Keeps CI green pre-feature."""

from pathlib import Path

import vce
from vce.backends.base import ExtractionBackend
from vce.types import BBox, Candidate, Extraction, Frame, MergedSnippet


def test_version_exposed():
    assert isinstance(vce.__version__, str)


def test_frame_timecode():
    frame = Frame(path=Path("frame.jpg"), timestamp_ms=3_725_500)
    assert frame.timecode == "01:02:05.500"


def test_types_construct():
    frame = Frame(path=Path("f.jpg"), timestamp_ms=0)
    cand = Candidate(frame=frame, score=0.9)
    ext = Extraction(frame=frame, text="import jax", confidence=0.8, bboxes=(BBox(0, 0, 1, 1),))
    snip = MergedSnippet(code="import jax", sources=(frame,))
    assert cand.score == 0.9
    assert ext.text == "import jax"
    assert snip.sources == (frame,)


def test_backend_protocol_is_runtime_checkable():
    class Dummy:
        name = "dummy"

        def extract(self, image_path, frame):  # pragma: no cover - structural check only
            return Extraction(frame=frame, text="", confidence=0.0)

    assert isinstance(Dummy(), ExtractionBackend)
