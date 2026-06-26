"""Stage 6 — merge overlapping snippets across frames + emit provenance.

This is where crude "OCR every frame" pipelines turn to mush: the same block of code is
captured in a dozen near-identical frames, each transcription slightly different (an OCR typo
here, a stray cursor glyph there), and naively concatenating them produces a garbled mess.
This stage instead *groups* near-identical extractions into one snippet and keeps the clearest
representative — **without inventing anything** — then records, for every output snippet, which
source frames it came from so the result is auditable.

Scope (per the issue) is the **static-snippet** path: cluster near-identical extractions and keep
the highest-confidence one. Continuous live-coding diff reconstruction (where the code grows line
by line across frames) is deliberately deferred.

Design
------
The work splits into three small, independently testable pieces, mirrored by three functions:

* **clustering** (:func:`_cluster`) — greedy single-pass grouping by *normalized* edit-distance
  similarity. Text is normalized (trailing whitespace and blank lines stripped) before comparison
  so trivial indentation/cursor noise doesn't split a group. Similarity is
  ``1 - levenshtein(a, b) / max(len(a), len(b))`` in ``[0, 1]``, reusing
  :func:`vce.bench.levenshtein_distance` so there is a single edit-distance implementation.
* **representative selection** (:func:`_choose_representative`) — within a cluster, pick the
  extraction with the highest confidence (deterministic tie-break: earliest timestamp, then path).
  An optional ``merge_fn`` lets a caller reconcile variants with an LLM; it is injected and mocked
  in tests so the default path stays pure and deterministic.
* **provenance serialization** (:func:`build_provenance`, :func:`write_provenance`) — emit the
  sidecar mapping every source extraction to the cleaned code it contributed to.

Conflicts (multiple genuinely different transcriptions that happened to cluster together, with no
clear confidence winner) and low-confidence representatives are **flagged in**
:attr:`~vce.types.MergedSnippet.notes` rather than silently resolved, so a human can audit them.

Everything here is pure and deterministic given its inputs.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path

from vce.bench import levenshtein_distance
from vce.types import Extraction, Frame, MergedSnippet

# Default thresholds. Tuned for the static-snippet path: cluster transcriptions that are
# typo-distance apart, flag genuinely ambiguous merges, and flag shaky OCR.
DEFAULT_SIMILARITY = 0.85
DEFAULT_LOW_CONFIDENCE = 0.5
DEFAULT_CONFLICT_MARGIN = 0.1

# A reconciler that fuses a cluster's variants into one cleaned snippet (e.g. an LLM call).
# Injectable so the default path stays pure; mocked in tests. Should be deterministic for callers
# that want reproducible output.
MergeFn = Callable[[Sequence[Extraction]], str]


def _normalize(text: str) -> str:
    """Strip trailing whitespace from each line and drop leading/trailing blank lines.

    Normalizing before comparison keeps cursor glyphs, trailing spaces, and blank-line jitter
    (all common OCR noise) from splitting two captures of the same code into separate clusters.
    Interior blank lines are preserved — they can be meaningful in source.
    """
    lines = [line.rstrip() for line in text.splitlines()]
    start, end = 0, len(lines)
    while start < end and not lines[start]:
        start += 1
    while end > start and not lines[end - 1]:
        end -= 1
    return "\n".join(lines[start:end])


def _similarity(a: str, b: str) -> float:
    """Normalized edit-distance similarity in ``[0, 1]``; two empty strings are ``1.0``."""
    longest = max(len(a), len(b))
    if longest == 0:
        return 1.0
    return 1.0 - levenshtein_distance(a, b) / longest


def _frame_sort_key(frame: Frame) -> tuple[int, str]:
    """Deterministic ordering for frames: earliest timestamp first, then path as a tie-break."""
    return (frame.timestamp_ms, str(frame.path))


def _cluster(
    extractions: Sequence[Extraction], similarity_threshold: float
) -> list[list[Extraction]]:
    """Greedily group extractions whose normalized texts are within the similarity threshold.

    Single pass in input order: each extraction joins the first existing cluster whose *seed*
    (first member) it is similar enough to, otherwise it seeds a new cluster. Comparing against a
    stable seed (rather than a moving representative) keeps the grouping order-deterministic. This
    is intentionally simple — adequate for the static-snippet path where captures of one block are
    mutually similar — and documented as such rather than a full transitive-closure clustering.
    """
    clusters: list[list[Extraction]] = []
    seeds: list[str] = []
    for extraction in extractions:
        norm = _normalize(extraction.text)
        for i, seed in enumerate(seeds):
            if _similarity(norm, seed) >= similarity_threshold:
                clusters[i].append(extraction)
                break
        else:
            clusters.append([extraction])
            seeds.append(norm)
    return clusters


def _choose_representative(cluster: Sequence[Extraction]) -> Extraction:
    """Pick the clearest extraction: highest confidence, tie-broken by earliest frame then path.

    Uses ``min`` over ``(-confidence, timestamp_ms, path)`` so every component sorts ascending:
    negating confidence makes the highest-confidence extraction the smallest, and the natural
    ordering of ``timestamp_ms`` then path string then breaks ties toward the earliest, then
    lexicographically smallest, path. (Negating code points under ``max`` would mishandle the case
    where one path is a prefix of another, since tuple length dominates the comparison.)
    """
    return min(
        cluster,
        key=lambda e: (-e.confidence, e.frame.timestamp_ms, str(e.frame.path)),
    )


def _build_notes(
    cluster: Sequence[Extraction],
    representative: Extraction,
    *,
    low_confidence_threshold: float,
    conflict_margin: float,
) -> str:
    """Flag low-confidence and conflicting merges (see module docstring); empty string if clean.

    Two independent checks, each appended as a human-readable note:

    * **low confidence** — the representative's confidence is below ``low_confidence_threshold``,
      so the whole snippet is shaky and worth a human look.
    * **conflict** — the cluster holds more than one genuinely distinct transcription and the
      runner-up distinct variant's confidence is within ``conflict_margin`` of the
      representative's, i.e. there is no clear winner and we may have kept the wrong one.
    """
    notes: list[str] = []

    if representative.confidence < low_confidence_threshold:
        notes.append(
            f"low confidence: representative confidence {representative.confidence:.2f} "
            f"< {low_confidence_threshold:.2f}"
        )

    rep_norm = _normalize(representative.text)
    distinct = {_normalize(e.text) for e in cluster}
    if len(distinct) > 1:
        # Highest confidence among members whose text differs from the representative's.
        runner_up = max(
            (e.confidence for e in cluster if _normalize(e.text) != rep_norm),
            default=None,
        )
        if runner_up is not None and representative.confidence - runner_up <= conflict_margin:
            notes.append(
                f"conflict: {len(distinct)} differing transcriptions with near-equal confidence "
                f"(representative {representative.confidence:.2f} vs runner-up {runner_up:.2f}); "
                f"kept the highest-confidence one"
            )

    return "; ".join(notes)


def merge_snippets(
    extractions: Sequence[Extraction],
    *,
    similarity_threshold: float = DEFAULT_SIMILARITY,
    low_confidence_threshold: float = DEFAULT_LOW_CONFIDENCE,
    conflict_margin: float = DEFAULT_CONFLICT_MARGIN,
    merge_fn: MergeFn | None = None,
) -> list[MergedSnippet]:
    """Merge de-duplicated, provenance-tagged snippets from per-frame extractions.

    Groups ``extractions`` by normalized edit-distance similarity, keeps the highest-confidence
    representative of each group as the cleaned ``code``, and records every contributing frame in
    :attr:`~vce.types.MergedSnippet.sources`. Conflicting and low-confidence merges are flagged in
    :attr:`~vce.types.MergedSnippet.notes` rather than silently resolved.

    Pure and deterministic: identical inputs yield identical output, ordered by each snippet's
    earliest source frame (timestamp, then path).

    Args:
        extractions: Per-frame extractions to merge. May be empty (returns ``[]``).
        similarity_threshold: Minimum normalized similarity in ``[0, 1]`` for two extractions to
            land in the same cluster. Higher is stricter (fewer, tighter groups).
        low_confidence_threshold: Representatives below this confidence are flagged in ``notes``.
        conflict_margin: When a cluster holds differing transcriptions, it is flagged as a conflict
            if the runner-up distinct variant's confidence is within this margin of the
            representative's.
        merge_fn: Optional reconciler (e.g. an LLM call) turning a cluster's variants into the
            cleaned code. Injected so it can be mocked; when ``None`` the representative's text is
            used verbatim, keeping the merge pure. Never invents code beyond what the caller's
            function chooses to.

    Raises:
        ValueError: if any threshold is outside ``[0, 1]``.
    """
    for name, value in (
        ("similarity_threshold", similarity_threshold),
        ("low_confidence_threshold", low_confidence_threshold),
        ("conflict_margin", conflict_margin),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be within [0, 1], got {value}")

    merged: list[MergedSnippet] = []
    for cluster in _cluster(extractions, similarity_threshold):
        representative = _choose_representative(cluster)
        code = merge_fn(cluster) if merge_fn is not None else representative.text
        sources = tuple(sorted((e.frame for e in cluster), key=_frame_sort_key))
        notes = _build_notes(
            cluster,
            representative,
            low_confidence_threshold=low_confidence_threshold,
            conflict_margin=conflict_margin,
        )
        merged.append(MergedSnippet(code=code, sources=sources, notes=notes))

    merged.sort(key=lambda m: _frame_sort_key(m.sources[0]) if m.sources else (0, ""))
    return merged


def build_provenance(
    extractions: Sequence[Extraction], merged: Sequence[MergedSnippet]
) -> list[dict[str, object]]:
    """Build the provenance sidecar: one entry per source extraction, linked to its cleaned code.

    Each entry is ``{timestamp, screenshot, raw_ocr, cleaned_code}`` where ``timestamp`` is the
    frame's ``timestamp_ms``, ``screenshot`` is the frame image path, ``raw_ocr`` is the original
    per-frame transcription, and ``cleaned_code`` is the merged snippet that extraction fed into.
    This is the audit trail: every cleaned snippet can be traced back to the exact frames (and
    their raw OCR) it was derived from. Entries are ordered by timestamp, then path, for
    determinism.

    Attribution is per *extraction*, not per frame: a single frame can yield more than one
    extraction (e.g. a hard frame re-run through two backends, or several cropped regions at one
    timestamp), and those can land in different clusters. Keying only by frame would let the last
    snippet written for that frame clobber the others and mislabel an extraction's cleaned code, so
    each extraction is matched to the snippet whose sources include its frame, disambiguated by
    text similarity when a frame feeds several snippets (see :func:`_snippet_for`).

    Args:
        extractions: The same extractions passed to :func:`merge_snippets`.
        merged: The snippets returned by :func:`merge_snippets` over those extractions.
    """
    entries: list[dict[str, object]] = []
    for extraction in extractions:
        snippet = _snippet_for(extraction, merged)
        entries.append(
            {
                "timestamp": extraction.frame.timestamp_ms,
                "screenshot": str(extraction.frame.path),
                "raw_ocr": extraction.text,
                "cleaned_code": snippet.code if snippet is not None else "",
            }
        )

    entries.sort(key=lambda e: (e["timestamp"], e["screenshot"]))
    return entries


def _snippet_for(extraction: Extraction, merged: Sequence[MergedSnippet]) -> MergedSnippet | None:
    """Find the merged snippet ``extraction`` contributed to, or ``None`` if it contributed to none.

    Candidates are the snippets whose ``sources`` include the extraction's frame. With one
    candidate the answer is unambiguous. When the same frame fed several snippets, the extraction
    is attributed to the candidate whose cleaned ``code`` is most similar to the extraction's own
    text — i.e. the cluster it actually belongs to — with ties broken by the snippets' deterministic
    order in ``merged``.
    """
    candidates = [m for m in merged if extraction.frame in m.sources]
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    norm = _normalize(extraction.text)
    return max(candidates, key=lambda m: _similarity(norm, _normalize(m.code)))


def write_provenance(path: Path | str, entries: Sequence[dict[str, object]]) -> None:
    """Serialize provenance ``entries`` to ``path`` as pretty-printed, UTF-8 JSON.

    Kept separate from :func:`build_provenance` so the in-memory provenance can be built and
    asserted on without touching the disk.
    """
    Path(path).write_text(
        json.dumps(list(entries), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
