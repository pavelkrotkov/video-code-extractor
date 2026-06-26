"""Stage 3 — score frames for code-likeness (the real "does this frame contain code?" gate).

Stub: implemented in the "Code-likeness scoring" issue.
"""

from __future__ import annotations

from vce.types import Candidate, Frame


def score_code_likeness(frame: Frame, text: str) -> Candidate:
    """Return a ``0.0``..``1.0`` code-likeness score for ``frame`` given its OCR ``text``."""
    raise NotImplementedError("see issue: Code-likeness scoring")
