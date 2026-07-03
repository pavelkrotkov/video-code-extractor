"""Shared OCR-model selection and vision-LLM output normalization.

Both remote OCR paths — the legacy synchronous backend (:mod:`vce.backends.vision`) and the
OpenAI Batch flow (:mod:`vce.batch_ocr`) — need the same three things: which model to call,
how to turn raw model text into code, and how to encode a screenshot for the API. Keeping
them here means the two paths cannot drift apart on the default model or the
``OPENAI_OCR_MODEL`` override.
"""

from __future__ import annotations

import base64
import mimetypes
import os
import re
from pathlib import Path

#: Default vision-OCR model; override with :data:`OCR_MODEL_ENV` or an explicit argument.
DEFAULT_OCR_MODEL = "gpt-5.4-mini"

#: Environment variable that overrides :data:`DEFAULT_OCR_MODEL`.
OCR_MODEL_ENV = "OPENAI_OCR_MODEL"

# OpenAI rejects images larger than 20 MB; fail fast with a clear message before the API call.
MAX_IMAGE_BYTES = 20 * 1024 * 1024

# Closing fence must be at the start of a line (MULTILINE ``^```) so triple-backticks *inside*
# the code (e.g. a Markdown string in a tutorial) aren't mistaken for the close. The optional
# ``\Z`` means a truncated response missing its closing fence still yields the code.
_FENCE_RE = re.compile(r"```[^\n]*\n(.*?)(?:^```|\Z)", re.DOTALL | re.MULTILINE)


def resolve_ocr_model(explicit: str | None = None) -> str:
    """The OCR model to use: ``explicit`` if given, else ``$OPENAI_OCR_MODEL``, else the default.

    A set-but-empty environment variable counts as unset, so ``OPENAI_OCR_MODEL=`` in a shell
    profile can't silently select an empty model name.
    """
    if explicit:
        return explicit
    return os.environ.get(OCR_MODEL_ENV, "").strip() or DEFAULT_OCR_MODEL


def strip_fence(content: str) -> str:
    """Return the contents of the first fenced code block, or the trimmed text if unfenced.

    Strips leading/trailing newlines (not spaces) so an extra blank line after the opening fence
    is removed while the first code line keeps its indentation.
    """
    match = _FENCE_RE.search(content)
    if match:
        return match.group(1).strip("\n")
    # Unfenced fallback: strip only surrounding newlines so the first line keeps its indentation.
    return content.strip("\n")


def text_confidence(text: str) -> float:
    """Heuristic confidence: start high, penalize each ambiguous ``[?]`` marker the model emits.

    Empty or whitespace-only output means a failed/blank transcription, so confidence is low.
    """
    if not text.strip():
        return 0.1
    return max(0.1, 0.9 - 0.1 * text.count("[?]"))


def image_data_uri(image_path: Path) -> str:
    """Base64 ``data:`` URI for ``image_path``, sized-checked against OpenAI's image limit.

    Raises :class:`ValueError` when the file exceeds :data:`MAX_IMAGE_BYTES` so the failure is a
    clear local message rather than a rejected (and possibly paid-for) API request.
    """
    size = image_path.stat().st_size
    if size > MAX_IMAGE_BYTES:
        raise ValueError(f"image {image_path} is {size / 1e6:.1f} MB, exceeds OpenAI's 20 MB limit")
    mime_type = mimetypes.guess_type(image_path)[0] or "image/png"
    b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{b64}"
