"""Deterministic OCR quality checks and cleanup used by the extraction pipeline."""

from __future__ import annotations

import re
from collections.abc import Sequence

from vce.types import Extraction

_PROMPT = re.compile(r"^[ \t]*(In|Out)\s*\[\s*[\d ]*\]\s*:?[ \t]*")
_NUMERIC = re.compile(r"[\s\d.,eE+\-\[\]()]+")
_ARRAY = re.compile(r"(?:array|tensor|matrix)\s*\([\s\d.,eE+\-\[\]()]+\)")
_PYTHON = re.compile(
    r"(?m)^\s*(?:from\s+[\w.]+\s+import\b|import\s+[\w.]+|(?:async\s+)?def\s+\w+\s*\(|"
    r"class\s+\w+\b|@\w|(?:if|elif|else|for|while|with|try|except|finally)\b.*:|"
    r"[\w.]+(?:\[[^\]\n]*\])?\s*(?:[-+*/%@&|^]|//|\*\*)?=(?!=)|[\w.]+\()"
)


def parses_as_python(text: str) -> bool:
    """Return whether non-empty text compiles as a Python module."""
    if not text.strip():
        return False
    try:
        compile(text.strip("\n"), "<ocr>", "exec")
    except (SyntaxError, ValueError, RecursionError):
        return False
    return True


def _looks_python(text: str) -> bool:
    return bool(_PYTHON.search(text))


def _prompt(line: str) -> tuple[str, str] | None:
    match = _PROMPT.match(line)
    if match is None:
        return None
    return match.group(1), line[match.end() :]


def _rendered_output(line: str) -> bool:
    text = line.strip()
    if len(text) < 12:
        return False
    if "=" in text or not any(char.isdigit() for char in text):
        return False
    if _ARRAY.fullmatch(text):
        return not parses_as_python(text)
    return bool(_NUMERIC.fullmatch(text)) and not parses_as_python(text)


def _clean_line(line: str, in_output: bool) -> tuple[str | None, bool]:
    prompt = _prompt(line)
    if prompt is not None:
        kind, line = prompt
        if kind == "Out":
            return None, True
        return (line if line.strip() else None), False
    if in_output:
        return None, bool(line.strip())
    return (None if _rendered_output(line) else line), False


def clean_transcription(text: str) -> str:
    """Strip notebook prompts/output while leaving ordinary source unchanged."""
    kept: list[str] = []
    in_output = False
    for line in text.splitlines():
        line, in_output = _clean_line(line, in_output)
        if line is not None:
            kept.append(line)
    return "\n".join(kept).strip("\n")


def is_suspect(text: str) -> bool:
    """Flag notebook pollution or Python-looking text that does not compile."""
    if not text.strip():
        return False
    if any(_prompt(line) is not None or _rendered_output(line) for line in text.splitlines()):
        return True
    return _looks_python(text) and not parses_as_python(text)


def _variant_rank(extraction: Extraction) -> tuple[float, ...]:
    text = clean_transcription(extraction.text)
    nonblank = sum(1 for line in text.splitlines() if line.strip())
    complete = (nonblank, len(text))
    earliest = -extraction.frame.timestamp_ms
    if parses_as_python(text):
        return (1, *complete, extraction.confidence, earliest)
    return (0, extraction.confidence, *complete, earliest)


def best_extraction(extractions: Sequence[Extraction]) -> Extraction:
    """Choose the most complete valid extraction, then confidence and time."""
    return max(extractions, key=_variant_rank)


def reconcile_cluster(extractions: Sequence[Extraction]) -> str:
    """Choose the best visible variant in a cluster and return its cleaned text."""
    return clean_transcription(best_extraction(extractions).text)
