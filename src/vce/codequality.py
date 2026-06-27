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

import re
from collections.abc import Sequence

from vce.types import Extraction

# Jupyter/IPython cell prompts and execution-count chrome: "In [12]:", "Out[3]:", "In [ ]:".
# Matches the prompt as a line *prefix* (with named kind) so it is stripped whether it sits on its
# own line or shares a line with the cell payload, e.g. "In [1]: import numpy as np".
_PROMPT = re.compile(r"^[ \t]*(?P<kind>In|Out)\s*\[\s*[\d ]*\]\s*:?[ \t]*")

# A bare line made up entirely of numeric / array punctuation — the fingerprint of *rendered*
# output (a printed ndarray slice or numeric table), not source.
_NUMERIC_LINE = re.compile(r"^[\s\d.,eE+\-]*[\[(][\s\d.,eE+\-\[\]()]*$")
# A line that is a printed repr of an array/tensor, e.g. ``array([0., 0., 0.])``. Deliberately
# excludes pandas ``DataFrame(...)`` / ``Series(...)``: pandas prints those as text tables, never in
# constructor form, so ``DataFrame([1, 2, 3])`` is only ever *source* and must not be stripped.
_ARRAY_REPR = re.compile(r"^(?:array|tensor|matrix)\s*\(")

# Strong, line-anchored Python markers used to decide whether to run the Python validator at all.
# Deliberately conservative: prose and other languages must not be labelled Python and then flagged.
_PYTHON_MARKERS = re.compile(
    r"(?m)^\s*(?:from\s+[\w.]+\s+import\b|import\s+[\w.]+|def\s+\w+\s*\(|async\s+def\s+\w+\s*\(|"
    r"class\s+\w+\b|@\w[\w.]*\s*(?:\(|$))"
)
_PYTHON_BLOCK = re.compile(
    r"(?m)^\s*(?:if|elif|else|for|while|with|try|except|finally)\b.*:\s*(?:#.*)?$"
)
# Keyword-less Python statements: an assignment / augmented-assignment (``y = ...``, ``a.b += 1``,
# ``x[i] = ...``) or a call at line start (``model.fit(...)``). These carry no def/import/class
# keyword, so without this a broken ``y = jnp.ones((3, 3)`` would never be detected as Python and
# would slip past validity flagging. The call form requires ``name(`` with no space so prose like
# "options (see above)" does not match.
_PYTHON_STATEMENT = re.compile(
    r"^\s*[\w.]+(?:\[[^\]\n]*\])?\s*(?:[-+*/%@&|^]|//|\*\*)?=(?!=)"  # assignment / aug-assign
    r"|^\s*[\w.]+\(",  # call: name( with no intervening space
    re.MULTILINE,
)


def detect_language(text: str) -> str | None:
    """Best-effort source-language label for ``text`` (currently ``"python"`` or ``None``).

    Returns ``"python"`` on strong Python markers (``import`` / ``from ... import`` statements,
    ``def`` / ``async def`` / ``class`` / decorator headers, control-flow block headers ending in
    ``:``) *or* keyword-less Python statements (an assignment or a call). The latter matters because
    much real code — ``y = jnp.ones((3, 3))``, ``model.fit(x)`` — carries no keyword, and without it
    a broken such line would never be validated or flagged. Clear prose (no assignment, call, or
    keyword) still falls through to ``None`` so it is never run through the Python validator.
    """
    if _PYTHON_MARKERS.search(text) or _PYTHON_BLOCK.search(text) or _PYTHON_STATEMENT.search(text):
        return "python"
    return None


def parses_as_python(text: str) -> bool:
    """Does ``text`` compile as a Python module? Trailing/leading blank tolerant; never raises.

    A clean structural quality signal: OCR that mangles brackets, f-string braces, or identifiers
    (``__init__`` → ``init``, ``max_len`` → ``max len``) almost always fails to compile, while a
    faithful transcription of a self-contained cell does. Returns ``False`` for empty text.

    Uses :func:`compile` (not :func:`ast.parse`) so that statements that parse but are invalid at
    module level — a top-level ``return`` / ``break`` / ``await`` left behind when OCR drops the
    enclosing block or its indentation — are correctly reported as invalid, honoring the
    "compile as a Python module" contract.
    """
    stripped = text.strip("\n")
    if not stripped.strip():
        return False
    try:
        compile(stripped, "<ocr>", "exec")
    except (SyntaxError, ValueError, RecursionError):
        # SyntaxError: malformed source (incl. misplaced return/break/await); ValueError: e.g.
        # embedded null bytes; RecursionError: pathologically deep nesting. Contract: never raises.
        return False
    return True


def _is_rendered_output(line: str) -> bool:
    """Is ``line`` a rendered array / numeric repr (printed output), not source?

    Recognised conservatively: a reasonably long, digit-bearing line with no assignment (``=``) that
    is either pure numeric/bracket content or a printed ``array``/``tensor``/``matrix`` repr. The
    no-assignment and length guards keep real code — ``shape = (1, 28, 28)``, ``arr = np.zeros(3)`` —
    from being mistaken for output. Note this is only ever applied to transcriptions that *do not*
    already parse as valid Python (see :func:`clean_transcription`), so a genuine multi-line literal
    row like ``[1, 2, 3, 4],`` inside otherwise-valid source is never reached, let alone stripped.
    """
    stripped = line.strip()
    if len(stripped) < 12 or "=" in stripped or not any(ch.isdigit() for ch in stripped):
        return False
    if _NUMERIC_LINE.match(stripped):
        # Distinguish a real Python list/tuple row from a rendered array even in the not-yet-parsing
        # fallback: a comma-separated row (``[1, 2, 3, 4],``) is valid Python and is preserved, while
        # a printed numpy/torch repr uses *space* separators (``[1. 2. 3.]``), fails to parse, and is
        # stripped. This stops a single syntax error elsewhere from silently deleting data rows.
        return not parses_as_python(stripped)
    if _ARRAY_REPR.match(stripped):
        body = stripped[stripped.index("(") :]
        return all(ch.isdigit() or ch in " .,+-eE[]()" for ch in body)
    return False


def _has_prompt(text: str) -> bool:
    """Does any line carry a Jupyter cell prompt (``In [n]:`` / ``Out[n]:``)? Definitive chrome."""
    return any(_PROMPT.match(line) for line in text.splitlines())


def _has_rendered_output(text: str) -> bool:
    """Does any line look like a rendered array/numeric repr (see :func:`_is_rendered_output`)?"""
    return any(_is_rendered_output(line) for line in text.splitlines())


def contains_notebook_chrome(text: str) -> bool:
    """Does any line of ``text`` carry a notebook cell prompt or rendered output (see :func:`is_suspect`)?"""
    return any(_PROMPT.match(line) or _is_rendered_output(line) for line in text.splitlines())


def clean_transcription(text: str, *, language: str | None = None) -> str:
    """Return ``text`` with notebook chrome and rendered output removed.

    Two safeguards, in order:

    1. **Never corrupt valid source.** If ``text`` has no cell prompts, carries no rendered-output
       line, and already parses as Python, it is real code — returned untouched (only edge blank
       lines trimmed). This is what keeps a genuine multi-line literal — rows like ``[1, 2, 3, 4],``
       — from being mistaken for a rendered array and silently deleted. (The prompt check matters
       because a stray ``Out[1]: array(...)`` line can itself happen to parse as a Python annotation;
       the rendered-output check catches a bare ``array([...])`` repr captured without its prompt,
       which also parses but is output, not source.)
    2. **Otherwise strip notebook chrome.** Jupyter cell prompts are removed whether standalone or
       inline (``In [1]: import x`` keeps ``import x``); an ``Out[n]:`` prompt opens an *output
       region* whose following lines (the rendered repr, numeric or not) are dropped until the next
       prompt or blank line; and standalone rendered-array/numeric lines are dropped too. Leading and
       trailing blank lines left behind are trimmed.

    The *raw* transcription is never mutated: this returns a cleaned copy, and callers keep the
    original for provenance. ``language`` is accepted for forward-compatibility (per-language cleaning
    can be added without a signature change) but the current filters are language-agnostic.
    """
    if not _has_prompt(text) and not _has_rendered_output(text) and parses_as_python(text):
        return text.strip("\n")

    kept: list[str] = []
    in_output = False
    for line in text.splitlines():
        prompt = _PROMPT.match(line)
        if prompt:
            rest = line[prompt.end() :]
            if prompt.group("kind") == "Out":
                in_output = True  # rendered output follows; drop this line's payload and the rest
                continue
            in_output = False  # an input prompt: real code resumes on/after this line
            if rest.strip():
                kept.append(rest)  # keep inline code after "In [n]:"
            continue
        if not line.strip():
            in_output = False  # a blank line closes an output block
            kept.append(line)
            continue
        if in_output or _is_rendered_output(line):
            continue
        kept.append(line)

    while kept and not kept[0].strip():
        kept.pop(0)
    while kept and not kept[-1].strip():
        kept.pop()
    return "\n".join(kept)


def is_suspect(text: str) -> bool:
    """Is this transcription structurally broken or polluted enough to distrust as code?

    Returns ``True`` when the text carries notebook cell prompts, or when it *looks* like Python but
    does not compile (the usual fingerprint of OCR-mangled brackets, f-string braces, or
    identifiers), or — for non-Python transcriptions — when it carries rendered output lines. Empty /
    non-code text is never suspect here; dropping that is the upstream code-likeness gate's job. This
    is the validity half of the escalation decision: it keeps high-confidence-but-broken OCR eligible
    for the accuracy tier.

    A valid Python snippet that merely contains comma-separated numeric literal rows
    (``[1, 2, 3, 4],``) is not flagged: those rows parse, so :func:`_is_rendered_output` does not
    treat them as output. But a snippet carrying a genuine rendered repr — a bare ``array([...])`` or
    a space-separated ``[1. 2. 3.]`` captured without its ``Out[n]:`` prompt — is suspect even though
    it may itself parse, so the rendered-output check runs regardless of validity.
    """
    if not text.strip():
        return False
    if _has_prompt(text) or _has_rendered_output(text):
        return True
    if detect_language(text) == "python":
        return not parses_as_python(text)
    return False


def _variant_rank(extraction: Extraction) -> tuple[float, ...]:
    """Sort key for reconciliation: best variant first (see :func:`reconcile_cluster`).

    ``valid`` (parses as Python) is always the top priority, so a valid variant beats a broken one
    independent of keyword-based detection — keyword-less code (``y = jnp.ones((3, 3))``) still wins
    over a broken sibling. The remaining keys are *validity-dependent*, because the two cases want
    opposite tie-breaks:

    * **Valid variants** rank **completeness before confidence** (most non-blank lines, then longest,
      then confidence). Among captures that all compile, the goal is the *most complete* one, so a
      fuller cell is not dropped for a shorter higher-confidence capture that lost an optional tail.
    * **Invalid / non-Python variants** rank **confidence before completeness**, so a higher-
      confidence clean read is not lost to a lower-confidence variant that merely has an extra
      cursor/noise line (which would also inflate its line count).

    Earliest frame (``-timestamp``) is the final deterministic tie-break. Cross-class comparison only
    ever reaches the leading ``valid`` element, so the two key layouts never compare against each
    other beyond it.
    """
    text = extraction.text
    nonblank = sum(1 for line in text.splitlines() if line.strip())
    completeness = (nonblank, len(text))
    confidence = extraction.confidence
    earliest = -extraction.frame.timestamp_ms
    if parses_as_python(text):
        return (1, *completeness, confidence, earliest)
    return (0, confidence, *completeness, earliest)


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
