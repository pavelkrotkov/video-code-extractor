"""Stage 4½ — language-aware code-quality signals: validity, suspicion, cleaning, reconciliation.

Apple Vision reports *recognition* confidence — how sure it is it read the glyphs correctly — which
is a poor proxy for whether the recognized text is valid *code*. A frame of rendered notebook
output, a prose heading, or a line whose brackets got mangled can all come back at 0.9+ confidence.
This module supplies the orthogonal signal the pipeline was missing: does the transcription actually
*look like*, and *parse as*, code? The pipeline ORs the two so high-confidence-but-broken OCR stays
eligible for the remote accuracy tier, and so a local-only (``--no-escalate``) run can flag snippets
it could not validate instead of silently presenting them as clean (issue #24).

Three concerns, kept separate and pure:

* **validity** — does a detected-Python transcription compile? (:func:`parses_as_python`)
* **suspicion** — is a code-like transcription structurally broken, or polluted with notebook
  chrome / rendered output, and therefore worth escalating or flagging? (:func:`is_suspect`)
* **cleaning + reconciliation** — strip notebook chrome / rendered output, and pick the most
  complete valid variant when several captures of one cell cluster together.
  (:func:`clean_transcription`, :func:`reconcile_cluster`)

Everything here is deterministic and import-light (stdlib ``ast`` only), so it is safe to call once
per frame inside the pipeline and is trivially unit-testable without OCR, network, or disk.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Sequence

from vce.types import Extraction

# Jupyter/IPython cell prompts and execution-count chrome: "In [12]:", "Out[3]:", "In [ ]:".
# These are interface chrome the notebook renders around a cell, never part of the source itself.
_NOTEBOOK_PROMPT = re.compile(r"^\s*(?:In|Out)\s*\[\s*[\d ]*\]\s*:?\s*$")

# A bare line made up entirely of numeric / array punctuation — the fingerprint of *rendered*
# output (a printed ndarray slice or numeric table), not source.
_NUMERIC_LINE = re.compile(r"^[\s\d.,eE+\-]*[\[(][\s\d.,eE+\-\[\]()]*$")
# A line that is a printed repr of an array/tensor/frame, e.g. ``array([0., 0., 0.])``.
_ARRAY_REPR = re.compile(r"^(?:array|tensor|matrix|DataFrame|Series)\s*\(")

# Strong, line-anchored Python markers used to decide whether to run the Python validator at all.
# Deliberately conservative: prose and other languages must not be labelled Python and then flagged.
_PYTHON_MARKERS = re.compile(
    r"(?m)^\s*(?:from\s+[\w.]+\s+import\b|import\s+[\w.]+|def\s+\w+\s*\(|async\s+def\s+\w+\s*\(|"
    r"class\s+\w+\b|@\w[\w.]*\s*(?:\(|$))"
)
_PYTHON_BLOCK = re.compile(
    r"(?m)^\s*(?:if|elif|else|for|while|with|try|except|finally)\b.*:\s*(?:#.*)?$"
)


def detect_language(text: str) -> str | None:
    """Best-effort source-language label for ``text`` (currently ``"python"`` or ``None``).

    Intentionally conservative: returns ``"python"`` only on strong Python markers (``import`` /
    ``from ... import`` statements, ``def`` / ``async def`` / ``class`` / decorator headers, or a
    control-flow block header ending in ``:``). Prose and other languages fall through to ``None``
    so they are never run through the Python validator and wrongly flagged as broken code.
    """
    if _PYTHON_MARKERS.search(text) or _PYTHON_BLOCK.search(text):
        return "python"
    return None


def parses_as_python(text: str) -> bool:
    """Does ``text`` compile as a Python module? Trailing/leading blank tolerant; never raises.

    A clean structural quality signal: OCR that mangles brackets, f-string braces, or identifiers
    (``__init__`` → ``init``, ``max_len`` → ``max len``) almost always fails to parse, while a
    faithful transcription of a self-contained cell parses. Returns ``False`` for empty text.
    """
    stripped = text.strip("\n")
    if not stripped.strip():
        return False
    try:
        ast.parse(stripped)
    except (SyntaxError, ValueError):  # ValueError covers e.g. source with embedded null bytes
        return False
    return True


def _is_chrome_or_output(line: str) -> bool:
    """Is ``line`` notebook chrome (a cell prompt) or rendered array/numeric output, not source?

    Rendered output is recognised conservatively: a reasonably long, digit-bearing line with no
    assignment (``=``) that is either pure numeric/bracket content or a printed array/tensor/frame
    repr. The no-assignment and length guards keep real code — ``shape = (1, 28, 28)``,
    ``arr = np.zeros(3)`` — from being mistaken for output.
    """
    if _NOTEBOOK_PROMPT.match(line):
        return True
    stripped = line.strip()
    if len(stripped) < 12 or "=" in stripped or not any(ch.isdigit() for ch in stripped):
        return False
    if _NUMERIC_LINE.match(stripped):
        return True
    if _ARRAY_REPR.match(stripped):
        body = stripped[stripped.index("(") :]
        return all(ch.isdigit() or ch in " .,+-eE[]()" for ch in body)
    return False


def contains_notebook_chrome(text: str) -> bool:
    """Does any line of ``text`` look like notebook chrome or rendered output (see :func:`is_suspect`)?"""
    return any(_is_chrome_or_output(line) for line in text.splitlines())


def clean_transcription(text: str, *, language: str | None = None) -> str:
    """Return ``text`` with notebook chrome and rendered output lines removed.

    Drops Jupyter cell prompts (``In [n]:`` / ``Out[n]:``) and lines that are clearly rendered
    output (a printed array / tensor / numeric table) rather than source, then trims the leading and
    trailing blank lines that removal can leave behind. Interior structure is otherwise untouched —
    valid code passes through unchanged. The *raw* transcription is never mutated: this returns a
    cleaned copy, and callers keep the original for provenance.

    ``language`` is accepted for forward-compatibility (so per-language cleaning can be added without
    a signature change) but is not required by the current, language-agnostic filters.
    """
    kept = [line for line in text.splitlines() if not _is_chrome_or_output(line)]
    while kept and not kept[0].strip():
        kept.pop(0)
    while kept and not kept[-1].strip():
        kept.pop()
    return "\n".join(kept)


def is_suspect(text: str) -> bool:
    """Is this transcription structurally broken or polluted enough to distrust as code?

    Returns ``True`` when the text carries notebook chrome / rendered output, or when it *looks* like
    Python but does not compile (the usual fingerprint of OCR-mangled brackets, f-string braces, or
    identifiers). Empty / non-code text is never suspect here — dropping that is the upstream
    code-likeness gate's job, not this signal's. This is the validity half of the escalation
    decision: it keeps high-confidence-but-broken OCR eligible for the accuracy tier.
    """
    if not text.strip():
        return False
    if contains_notebook_chrome(text):
        return True
    return detect_language(text) == "python" and not parses_as_python(text)


def _variant_rank(extraction: Extraction) -> tuple[int, int, int, float, int]:
    """Sort key for reconciliation: most complete *valid* variant first (see :func:`reconcile_cluster`).

    Ascending-``max`` tuple: prefer a transcription that parses as its detected language, then the
    most complete (most non-blank lines, then longest), then the highest confidence, tie-broken by
    the earliest frame (``-timestamp`` so earliest wins under ``max``). Every component is a number,
    so the ordering is total and deterministic.
    """
    text = extraction.text
    nonblank = sum(1 for line in text.splitlines() if line.strip())
    valid = int(detect_language(text) == "python" and parses_as_python(text))
    return (valid, nonblank, len(text), extraction.confidence, -extraction.frame.timestamp_ms)


def reconcile_cluster(extractions: Sequence[Extraction]) -> str:
    """Deterministically reconcile a cluster of near-duplicate captures into one clean snippet.

    Picks the *most complete valid* variant (see :func:`_variant_rank`) and returns it with notebook
    chrome and rendered output stripped. This is the deterministic reconciler the pipeline injects as
    ``merge_fn`` (:func:`vce.merge.merge_results`) so overlapping captures of one cell collapse to a
    single best block instead of being concatenated as separate snippets (issue #24). It never
    invents text — it only *selects among*, and *filters*, what the OCR backends actually read.

    ``extractions`` is always non-empty (clusters are seeded by at least one extraction).
    """
    best = max(extractions, key=_variant_rank)
    return clean_transcription(best.text, language=detect_language(best.text))
