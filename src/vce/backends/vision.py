"""Legacy synchronous vision-LLM extraction backend (kept for the interactive escalation tier).

.. deprecated::
    Bulk OCR now goes through the OpenAI Batch API — see :mod:`vce.batch_ocr` and the
    ``vce ocr-submit`` / ``vce ocr-fetch`` commands (issue #32). This synchronous per-frame
    path remains only for the two-tier ``vce extract`` pipeline, where the escalation tier
    must answer within the run and cannot wait on a batch's completion window.

The #1 correctness risk with vision LLMs is hallucination: they "autocomplete" plausible but
invisible code. The mitigation lives in :data:`OCR_SYSTEM_PROMPT`, which forbids inference and
asks the model to act strictly as OCR. The OpenAI client is imported lazily and is injectable, so
the response→:class:`~vce.types.Extraction` mapping is unit-testable without any network call.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from vce.ocr import image_data_uri, resolve_ocr_model, strip_fence, text_confidence
from vce.types import Extraction, Frame

OCR_SYSTEM_PROMPT = (
    "You are an OCR engine, not a programmer. Transcribe ONLY the code that is visibly present "
    "in the image. Do NOT infer, complete, or correct missing or off-screen lines. Preserve "
    "indentation, punctuation, capitalization, and line breaks exactly. If a character is "
    "ambiguous, mark it as [?] rather than guessing. Return ONLY a single fenced code block with "
    "no commentary."
)


class _ChatClient(Protocol):
    chat: Any


class VisionLLMBackend:
    """:class:`~vce.backends.base.ExtractionBackend` backed by an OpenAI vision model.

    The model defaults to :func:`vce.ocr.resolve_ocr_model` — ``gpt-5.4-mini`` unless
    ``$OPENAI_OCR_MODEL`` or the ``model`` argument overrides it — so the escalation tier and
    the batch flow use the same model by default.
    """

    name = "vision-gpt4v"

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        client: _ChatClient | None = None,
    ) -> None:
        self._model = resolve_ocr_model(model)
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
        data_uri = image_data_uri(image_path)
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
        if not response.choices:
            raise RuntimeError("OpenAI returned no choices for the vision request")
        choice = response.choices[0]
        content = choice.message.content or ""
        text = strip_fence(content)
        confidence = text_confidence(text)
        # A completion truncated at the output-token limit is partial code; cap its confidence so
        # the merge stage doesn't treat a silently-cut snippet as a reliable extraction.
        if getattr(choice, "finish_reason", None) == "length":
            confidence = min(confidence, 0.3)
        return Extraction(
            frame=frame,
            text=text,
            confidence=confidence,
            backend=self.name,
        )
