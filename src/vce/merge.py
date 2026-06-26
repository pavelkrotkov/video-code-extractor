"""Stage 6 — merge overlapping snippets across frames + emit provenance.

Stub: implemented in the "Cross-frame merge" issue.
"""

from __future__ import annotations

from collections.abc import Sequence

from vce.types import Extraction, MergedSnippet


def merge_snippets(extractions: Sequence[Extraction]) -> list[MergedSnippet]:
    """Merge de-duplicated, provenance-tagged snippets from per-frame extractions."""
    raise NotImplementedError("see issue: Cross-frame merge")
