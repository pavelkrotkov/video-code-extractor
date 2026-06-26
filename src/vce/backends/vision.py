"""Vision-LLM extraction backend (OpenAI GPT-4V) — the accuracy option.

The #1 correctness risk with vision LLMs is hallucination: they "autocomplete" plausible but
invisible code. The mitigation lives in :data:`OCR_SYSTEM_PROMPT`, which forbids inference and
asks the model to act strictly as OCR. The OpenAI client is imported lazily and is injectable, so
the response→:class:`~vce.types.Extraction` mapping is unit-testable without any network call.
"""

from __future__ import annotations

import base64
import re
from pathlib import Path
from typing import Any, Protocol

from vce.types import Extraction, Frame

OCR_SYSTEM_PROMPT = (
    "You are an OCR engine, not a programmer. Transcribe ONLY the code that is visibly present "
    "in the image. Do NOT infer, complete, or correct missing or off-screen lines. Preserve "
    "indentation, punctuation, capitalization, and line breaks exactly. If a character is "
    "ambiguous, mark it as [?] rather than guessing. Return ONLY a single fenced code block with "
    "no commentary."
)

_FENCE_RE = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)


class _ChatClient(Protocol):
    chat: Any


def _strip_fence(content: str) -> str:
    """Return the contents of the first fenced code block, or the trimmed text if unfenced."""
    match = _FENCE_RE.search(content)
    if match:
        return match.group(1).rstrip("\n")
    return content.strip()


def _confidence(text: str) -> float:
    """Heuristic confidence: start high, penalize each ambiguous ``[?]`` marker the model emits."""
    return max(0.1, 0.9 - 0.1 * text.count("[?]"))


class VisionLLMBackend:
    """:class:`~vce.backends.base.ExtractionBackend` backed by an OpenAI vision model."""

    name = "vision-gpt4v"

    def __init__(
        self,
        *,
        model: str = "gpt-4o",
        api_key: str | None = None,
        client: _ChatClient | None = None,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._client = client

    def _get_client(self) -> _ChatClient:
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - exercised via monkeypatched import
                raise ImportError(
                    "openai is required for the vision backend: pip install openai"
                ) from exc
            self._client = OpenAI(api_key=self._api_key)
        return self._client

    def _build_messages(self, image_path: Path) -> list[dict[str, Any]]:
        b64 = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
        data_uri = f"data:image/png;base64,{b64}"
        return [
            {"role": "system", "content": OCR_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Transcribe the code in this screenshot."},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            },
        ]

    def extract(self, image_path: Path, frame: Frame) -> Extraction:
        client = self._get_client()
        response = client.chat.completions.create(
            model=self._model,
            messages=self._build_messages(image_path),
            temperature=0,
        )
        content = response.choices[0].message.content or ""
        text = _strip_fence(content)
        return Extraction(
            frame=frame,
            text=text,
            confidence=_confidence(text),
            backend=self.name,
        )
