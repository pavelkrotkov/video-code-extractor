"""Bulk code OCR through the OpenAI Batch API (replaces the legacy GPT-4V call path, issue #32).

The OCR workload is offline, frame-based, and naturally batchable: an upstream stage has already
sampled, deduplicated, and (optionally) cropped candidate frames, so each screenshot becomes one
JSONL request line submitted to the Batch API against ``/v1/responses``. Because a batch can take
hours, the flow is split into two user-visible steps (``vce ocr-submit`` / ``vce ocr-fetch``)
joined by a *manifest* file: submit records each request's ``custom_id`` together with the frame
provenance (video, timestamp, image path), and fetch matches results back **by custom_id only** —
batch output order is never assumed to match input order.

Fetch keeps the raw batch output on disk for debugging and normalizes every result into one
timestamped OCR record with a ``status`` of ``ok`` / ``no_code_visible`` / ``uncertain`` /
``error``. The module follows the project's testability pattern: request building and output
parsing are pure and client-free, and the two API wrappers take an injected client.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from vce.ocr import image_data_uri, resolve_ocr_model, strip_fence, text_confidence
from vce.types import Extraction, Frame, format_timecode

BATCH_ENDPOINT = "/v1/responses"
COMPLETION_WINDOW = "24h"
MAX_OUTPUT_TOKENS = 1200

#: Backend identifier recorded on extractions produced from batch OCR records.
BACKEND_NAME = "openai-batch"

#: Exact sentinel the prompt asks for when a screenshot contains no code.
NO_CODE_SENTINEL = "NO_CODE_VISIBLE"

OCR_PROMPT = (
    "Extract only source code visibly present in this screenshot. Do not infer missing lines. "
    "Do not complete partial code. Preserve indentation, punctuation, capitalization, and line "
    f"breaks. If no source code is visible, return exactly: {NO_CODE_SENTINEL}."
)

# Normalized record statuses.
STATUS_OK = "ok"
STATUS_NO_CODE = "no_code_visible"
STATUS_UNCERTAIN = "uncertain"
STATUS_ERROR = "error"

# The Batch API requires custom_id to be a unique string of at most 64 characters.
_CUSTOM_ID_MAX = 64
_UNSAFE_CHARS_RE = re.compile(r"[^A-Za-z0-9_-]+")


@dataclass(frozen=True)
class OCRRequest:
    """One screenshot to OCR: the stable ``custom_id`` plus the provenance needed at fetch time.

    ``image_path`` is the image actually sent to the API (the crop when cropping is enabled);
    ``frame_path`` is the full source frame, kept so merge provenance can point at the original
    screenshot.
    """

    custom_id: str
    video: str
    timestamp_ms: int
    frame_path: str
    image_path: str


def make_custom_id(video_stem: str, timestamp_ms: int, seq: int) -> str:
    """Stable, per-frame batch ``custom_id``: ``<stem>_<HHMMSS>_<seq>``, e.g. ``lesson01_000314_000``.

    The id is derived only from the video name, the frame timestamp, and the frame's position in
    the submitted sequence, so resubmitting the same frames yields the same ids. ``seq``
    disambiguates frames sharing a second (fps sampling above 1 Hz, scene cuts). The stem is
    sanitized to the Batch API's allowed shape and truncated so the whole id stays within its
    64-character limit.
    """
    if timestamp_ms < 0:
        raise ValueError(f"timestamp_ms must be non-negative, got {timestamp_ms}")
    stem = _UNSAFE_CHARS_RE.sub("-", video_stem).strip("-_") or "video"
    s_total = timestamp_ms // 1000
    compact = f"{s_total // 3600:02d}{(s_total // 60) % 60:02d}{s_total % 60:02d}"
    suffix = f"_{compact}_{seq:03d}"
    return stem[: _CUSTOM_ID_MAX - len(suffix)] + suffix


def build_requests(video: Path, images: list[tuple[Frame, Path]]) -> list[OCRRequest]:
    """One :class:`OCRRequest` per ``(frame, image)`` pair, with sequential stable ids."""
    stem = video.stem or "video"
    return [
        OCRRequest(
            custom_id=make_custom_id(stem, frame.timestamp_ms, seq),
            video=video.name,
            timestamp_ms=frame.timestamp_ms,
            frame_path=str(frame.path),
            image_path=str(image),
        )
        for seq, (frame, image) in enumerate(images)
    ]


def request_line(request: OCRRequest, model: str) -> dict[str, Any]:
    """The dict for one batch-input JSONL line: a ``/v1/responses`` OCR call for one screenshot."""
    return {
        "custom_id": request.custom_id,
        "method": "POST",
        "url": BATCH_ENDPOINT,
        "body": {
            "model": model,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": OCR_PROMPT},
                        {
                            "type": "input_image",
                            "image_url": image_data_uri(Path(request.image_path)),
                        },
                    ],
                }
            ],
            "max_output_tokens": MAX_OUTPUT_TOKENS,
        },
    }


def write_batch_input(path: Path, requests: list[OCRRequest], model: str) -> None:
    """Write the batch-input JSONL (one request line per screenshot) to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for request in requests:
            fh.write(json.dumps(request_line(request, model), ensure_ascii=False) + "\n")


def make_client(api_key: str | None = None) -> Any:
    """A real OpenAI client; kept behind a function so the CLI and tests can swap it out."""
    from openai import OpenAI

    return OpenAI(api_key=api_key)


def submit_batch(client: Any, jsonl_path: Path, *, description: str = "code screenshot OCR") -> str:
    """Upload ``jsonl_path`` and create the batch; return the batch id to poll later."""
    with jsonl_path.open("rb") as fh:
        uploaded = client.files.create(file=fh, purpose="batch")
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint=BATCH_ENDPOINT,
        completion_window=COMPLETION_WINDOW,
        metadata={"description": description},
    )
    return batch.id


def write_manifest(
    path: Path, *, batch_id: str, model: str, video: str, requests: list[OCRRequest]
) -> None:
    """Persist what fetch needs: the batch id and the custom_id → frame provenance mapping."""
    payload = {
        "batch_id": batch_id,
        "model": model,
        "video": video,
        "requests": [asdict(r) for r in requests],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_manifest(path: Path) -> tuple[str, str, str, list[OCRRequest]]:
    """Load a manifest written by :func:`write_manifest`; returns (batch_id, model, video, requests)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    try:
        requests = [OCRRequest(**entry) for entry in data["requests"]]
        return data["batch_id"], data["model"], data["video"], requests
    except (KeyError, TypeError) as exc:
        raise ValueError(f"{path} is not a valid batch manifest: {exc}") from exc


def _response_text(body: dict[str, Any]) -> str:
    """Concatenated ``output_text`` parts of a ``/v1/responses`` body (raw JSON shape)."""
    parts: list[str] = []
    for item in body.get("output") or []:
        if item.get("type") != "message":
            continue
        for part in item.get("content") or []:
            if part.get("type") == "output_text":
                parts.append(part.get("text") or "")
    return "".join(parts)


def _classify_line(obj: dict[str, Any]) -> tuple[str, str, str]:
    """Normalize one raw batch output/error line into ``(status, text, detail)``."""
    error = obj.get("error")
    if error:
        return STATUS_ERROR, "", f"batch error: {error}"
    response = obj.get("response") or {}
    status_code = response.get("status_code")
    if status_code != 200:
        return STATUS_ERROR, "", f"HTTP {status_code}"
    body = response.get("body") or {}
    text = strip_fence(_response_text(body))
    if text.strip() == NO_CODE_SENTINEL:
        return STATUS_NO_CODE, "", ""
    if body.get("status") == "incomplete":
        reason = (body.get("incomplete_details") or {}).get("reason", "unknown")
        return STATUS_UNCERTAIN, text, f"incomplete response: {reason}"
    if not text.strip():
        return STATUS_UNCERTAIN, "", "empty model output"
    return STATUS_OK, text, ""


def parse_batch_output(
    raw: str, requests: list[OCRRequest], *, error_raw: str = ""
) -> list[dict[str, Any]]:
    """Match raw batch output back to ``requests`` by ``custom_id`` and normalize each result.

    Returns one record per *request*, in request (timeline) order regardless of the order lines
    appear in the batch output. Requests with no matching line (expired, or the line was
    unparseable) become ``status="error"`` records, so a partial batch can never silently drop
    frames. Lines with unknown custom_ids are ignored. ``error_raw`` is the batch's error file
    (failed request lines), parsed with the same rules.
    """
    by_id: dict[str, tuple[str, str, str]] = {}
    for line in (raw + "\n" + error_raw).splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue  # unmatchable; the affected request is reported as missing below
        custom_id = obj.get("custom_id")
        if not custom_id or custom_id in by_id:
            continue
        by_id[custom_id] = _classify_line(obj)

    records: list[dict[str, Any]] = []
    for request in requests:
        status, text, detail = by_id.get(
            request.custom_id, (STATUS_ERROR, "", "missing from batch output")
        )
        record: dict[str, Any] = {
            "id": request.custom_id,
            "video": request.video,
            "timestamp": format_timecode(request.timestamp_ms),
            "timestamp_ms": request.timestamp_ms,
            "frame_path": request.frame_path,
            "image_path": request.image_path,
            "status": status,
            "text": text,
        }
        if detail:
            record["detail"] = detail
        records.append(record)
    return records


def write_records(path: Path, records: list[dict[str, Any]]) -> None:
    """Write normalized OCR ``records`` as JSONL (one record per line) to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def records_to_extractions(records: list[dict[str, Any]]) -> list[Extraction]:
    """Usable records as :class:`~vce.types.Extraction` objects for the existing merge stage.

    ``no_code_visible`` and ``error`` records carry no code and are skipped. ``uncertain``
    records keep their text but are capped at low confidence, mirroring how the synchronous
    backend treats truncated completions, so the merge stage prefers cleaner reads.
    """
    extractions: list[Extraction] = []
    for record in records:
        status = record["status"]
        if status not in (STATUS_OK, STATUS_UNCERTAIN) or not record["text"].strip():
            continue
        confidence = text_confidence(record["text"])
        if status == STATUS_UNCERTAIN:
            confidence = min(confidence, 0.3)
        extractions.append(
            Extraction(
                frame=Frame(path=Path(record["frame_path"]), timestamp_ms=record["timestamp_ms"]),
                text=record["text"],
                confidence=confidence,
                backend=BACKEND_NAME,
            )
        )
    return extractions


__all__ = [
    "BACKEND_NAME",
    "BATCH_ENDPOINT",
    "COMPLETION_WINDOW",
    "MAX_OUTPUT_TOKENS",
    "NO_CODE_SENTINEL",
    "OCR_PROMPT",
    "STATUS_ERROR",
    "STATUS_NO_CODE",
    "STATUS_OK",
    "STATUS_UNCERTAIN",
    "OCRRequest",
    "build_requests",
    "make_client",
    "make_custom_id",
    "parse_batch_output",
    "read_manifest",
    "records_to_extractions",
    "request_line",
    "resolve_ocr_model",
    "submit_batch",
    "write_batch_input",
    "write_manifest",
    "write_records",
]
