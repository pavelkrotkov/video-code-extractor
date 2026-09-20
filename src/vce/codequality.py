"""Deterministic OCR quality checks and cleanup used by the extraction pipeline."""

from __future__ import annotations

import re
from collections.abc import Sequence

from vce.types import Extraction

_PROMPT = re.compile(r"^[ \t]*(In|Out)\s*\[\s*[\d ]*\]\s*:[ \t]*")
_NUMERIC = re.compile(r"[\s\d.,eE+\-\[\]()]+")
_ARRAY = re.compile(r"(?:array|tensor|matrix)\s*\([\s\d.,eE+\-\[\]()]+\)")
# Python detection gates suspicion only; it must never decide what source text gets deleted.
_PYTHON = re.compile(
    r"(?m)^\s*(?:from\s+[\w.]+\s+import\b|import\s+[\w.]+|(?:async\s+)?def\s+\w+\s*\(|"
    r"class\s+\w+\b|@\w|(?:if|elif|else|for|while|with|try|except|finally)\b.*:|"
    r"(?:return|raise|yield|break|continue|await)\b|"
    r"[\w.]+(?:\[[^\]\n]*\])?\s*(?:[-+*/%@&|^]|//|\*\*)?=(?!=)|[\w.]+\()"
)


# Structural validity is independent of OCR confidence, and compile() catches misplaced returns too.
def parses_as_python(text: str) -> bool:
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
    if match is not None:
        return match.group(1), line[match.end() :]
    match = _CONTINUATION.match(line)
    return ("In", line[match.end() :]) if match is not None else None


def _rendered_output(line: str) -> bool:
    text = line.strip()
    if len(text) < 12:
        return False
    if "=" in text or not any(char.isdigit() for char in text):
        return False
    candidate = _ARRAY.fullmatch(text) or _NUMERIC.fullmatch(text)
    return bool(candidate) and not parses_as_python(text)


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


# Cleaning is derived-only: the upstream Extraction keeps raw OCR untouched for provenance.
def clean_transcription(text: str) -> str:
    kept: list[str] = []
    in_output = False
    for line in text.splitlines():
        line, in_output = _clean_line(line, in_output)
        if line is not None:
            kept.append(line)
    return "\n".join(kept).strip("\n")


# Prompts/output are definitive chrome; prose is left to the upstream code-likeness gate.
def is_suspect(text: str) -> bool:
    if not text.strip():
        return False
    if any(_prompt(line) is not None or _rendered_output(line) for line in text.splitlines()):
        return True
    return _looks_python(text) and not parses_as_python(text)


# Validity outranks confidence; among valid captures, completeness outranks confidence.
def _variant_rank(extraction: Extraction, prefer_validity: bool) -> tuple[float, ...]:
    text = clean_transcription(extraction.text)
    nonblank = sum(1 for line in text.splitlines() if line.strip())
    complete = (nonblank, len(text))
    earliest = -extraction.frame.timestamp_ms
    if prefer_validity and parses_as_python(text):
        return (1, *complete, extraction.confidence, earliest)
    return (0, extraction.confidence, *complete, earliest)


def best_extraction(extractions: Sequence[Extraction]) -> Extraction:
    prefer_validity = all(_looks_python(clean_transcription(e.text)) for e in extractions)
    return max(extractions, key=lambda e: _variant_rank(e, prefer_validity))


def reconcile_cluster(extractions: Sequence[Extraction]) -> str:
    return clean_transcription(best_extraction(extractions).text)
