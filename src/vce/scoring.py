"""Stage 3 — score frames for code-likeness (the real "does this frame contain code?" gate).

Scene-change detection only answers "did pixels change?"; this answers "does this frame contain
code?", so non-code frames (narrator, slide titles, browser chrome) are dropped before the
expensive crop/extract/merge stages.

The score is a weighted sum of *structural* signals (def/class headers, imports, operators, block
colons, indentation, calls) plus a deliberately tiny weight for bare keywords. Structure is what
separates code from prose: English contains words like "if", "for", "return", and "class", but not
``def name(``, ``==``, ``x = y``, or indented ``...:`` blocks. Each signal contributes once
(presence, not count) and the total saturates at ``1.0``, keeping the score bounded and stable.
"""

from __future__ import annotations

import re

from vce.types import Candidate, Frame

# (compiled pattern, weight). Order is irrelevant; each contributes at most once.
_SIGNALS: list[tuple[re.Pattern[str], float]] = [
    # function / class definitions — require a name and an opening ``(``, ``:`` or ``{`` so prose
    # like "the class later" does not match, while brace-style (Java/JS/C#) headers do.
    (re.compile(r"\b(?:def|function|fn)\s+\w+\s*\(|\bclass\s+\w+\s*[({:]"), 0.5),
    # import statements (anchored to line start so "From the beginning" is not a hit).
    (re.compile(r"(?m)^\s*from\s+[\w.]+\s+import\b|^\s*import\s+[\w.]+|#include\b"), 0.4),
    # SQL: SELECT ... FROM, but reject English stop-words between them so prose like
    # "select one of the options from the menu" does not match an actual SELECT list.
    (
        re.compile(
            r"(?is)\bselect\b"
            r"(?:(?!\b(?:the|of|a|an|to|please|your|our|you|this|that|these|those)\b)[\s\S]){0,120}?"
            r"\bfrom\b"
        ),
        0.4,
    ),
    # block headers ending in ``:`` or ``{`` (brace languages), with an optional trailing comment.
    (
        re.compile(
            r"(?m)^\s*(?:if|elif|else|for|while|def|class|try|except|finally|with|match|case|switch)\b.*[:{]\s*(?:#.*|//.*)?$"
        ),
        0.35,
    ),
    # shell / package-manager lines.
    (re.compile(r"(?m)^\s*\$ |\bpip install\b|\bnpm install\b|\bapt-get\b|\bcargo\s+\w+"), 0.35),
    # HTML / XML tags.
    (re.compile(r"</?[a-zA-Z][\w:-]*(?:\s[^<>]*)?/?>"), 0.3),
    # multi-character operators that are rare in prose.
    (re.compile(r"==|!=|<=|>=|=>|->|&&|\|\||\+=|-=|::"), 0.3),
    # single ``=`` assignment at the start of a line (lookahead so a trailing ``x =`` matches).
    (re.compile(r"(?m)^\s*[\w.\[\]\"']+\s*=\s*(?!=)"), 0.25),
    # typed / keyword variable declarations: ``const x =``, ``int x =``, ``let y =``.
    (
        re.compile(
            r"(?m)^\s*(?:const|let|var|final|static|public|private|int|float|double|long|char|bool|boolean|string|auto)\s+\w+\s*="
        ),
        0.25,
    ),
    # function call: identifier followed by parentheses, excluding prose plurals like "word(s)".
    (re.compile(r"\b\w+\((?![sS]\)|[eE][sS]\))[^)]*\)"), 0.25),
    # statement-terminating semicolons (optionally followed by a trailing comment).
    (re.compile(r"(?m);\s*(?:#.*|//.*)?$"), 0.25),
    # indentation: at least one indented, non-blank line.
    (re.compile(r"(?m)^[ \t]+\S"), 0.2),
    # brackets and braces.
    (re.compile(r"[{}\[\]]"), 0.15),
    # snake_case / UPPER_CASE / camelCase / dotted.names identifiers.
    (re.compile(r"\b\w+_\w+\b|\b[a-z]+[A-Z]\w*\b|\b\w+\.\w+\b"), 0.12),
    # code comments.
    (re.compile(r"(?m)(?:^|\s)(?:#|//|/\*)"), 0.1),
    # bare keywords — intentionally tiny, since these also appear in English.
    (
        re.compile(
            r"\b(?:def|class|import|from|return|elif|lambda|yield|async|await|const|let|var|void|struct)\b"
        ),
        0.05,
    ),
]


def _score_text(text: str) -> float:
    if not text.strip():
        return 0.0
    total = sum(weight for pattern, weight in _SIGNALS if pattern.search(text))
    return min(1.0, total)


def score_code_likeness(frame: Frame, text: str) -> Candidate:
    """Return a ``0.0``..``1.0`` code-likeness score for ``frame`` given its OCR ``text``."""
    return Candidate(frame=frame, score=_score_text(text))
