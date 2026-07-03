"""Tests for the OpenAI Batch OCR flow: request building, submission, and output parsing.

Everything runs offline: the pure helpers are exercised directly, and the two API wrappers get a
fake client that records the calls, mirroring the injection pattern of ``test_vision_backend``.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from vce import batch_ocr
from vce.batch_ocr import (
    BATCH_ENDPOINT,
    COMPLETION_WINDOW,
    MAX_OUTPUT_TOKENS,
    NO_CODE_SENTINEL,
    OCR_PROMPT,
    OCRRequest,
    build_requests,
    make_custom_id,
    parse_batch_output,
    read_manifest,
    records_to_extractions,
    request_line,
    submit_batch,
    write_batch_input,
    write_manifest,
)
from vce.ocr import DEFAULT_OCR_MODEL, resolve_ocr_model
from vce.types import Frame


@pytest.fixture
def png(tmp_path):
    from PIL import Image

    path = tmp_path / "frame_000001.png"
    Image.new("RGB", (10, 10), "white").save(path)
    return path


def _request(custom_id="lesson01_000314_000", ts=194_000, image="crop.png"):
    return OCRRequest(
        custom_id=custom_id,
        video="lesson01.mp4",
        timestamp_ms=ts,
        frame_path="frames/frame_000194.jpg",
        image_path=image,
    )


def _output_line(custom_id, text, *, status_code=200, body_status="completed", error=None):
    """One raw batch output line in the documented ``/v1/responses`` batch shape."""
    body = {
        "status": body_status,
        "output": [
            {"type": "reasoning", "summary": []},
            {"type": "message", "content": [{"type": "output_text", "text": text}]},
        ],
    }
    if body_status == "incomplete":
        body["incomplete_details"] = {"reason": "max_output_tokens"}
    return json.dumps(
        {
            "custom_id": custom_id,
            "response": {"status_code": status_code, "body": body},
            "error": error,
        }
    )


# --- model resolution ----------------------------------------------------------------------


def test_default_model(monkeypatch):
    monkeypatch.delenv("OPENAI_OCR_MODEL", raising=False)
    assert resolve_ocr_model() == DEFAULT_OCR_MODEL == "gpt-5.4-mini"


def test_model_env_override(monkeypatch):
    monkeypatch.setenv("OPENAI_OCR_MODEL", "gpt-6-max")
    assert resolve_ocr_model() == "gpt-6-max"


def test_explicit_model_beats_env(monkeypatch):
    monkeypatch.setenv("OPENAI_OCR_MODEL", "gpt-6-max")
    assert resolve_ocr_model("flag-model") == "flag-model"


def test_empty_env_model_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("OPENAI_OCR_MODEL", "  ")
    assert resolve_ocr_model() == DEFAULT_OCR_MODEL


# --- custom ids -----------------------------------------------------------------------------


def test_custom_id_matches_issue_shape():
    # 00:03:14 into lesson01, first frame of the batch
    assert make_custom_id("lesson01", 194_000, 0) == "lesson01_000314_000"


def test_custom_id_is_stable():
    assert make_custom_id("lesson01", 194_000, 7) == make_custom_id("lesson01", 194_000, 7)


def test_custom_id_sequence_disambiguates_same_second():
    ids = {make_custom_id("lesson01", 500, seq) for seq in range(3)}
    assert len(ids) == 3


def test_custom_id_sanitizes_stem():
    assert make_custom_id("my lesson (final).v2", 0, 1) == "my-lesson-final-v2_000000_001"


def test_custom_id_respects_64_char_limit():
    long_stem = "x" * 200
    cid = make_custom_id(long_stem, 3_599_999, 999)
    assert len(cid) <= 64
    assert cid.endswith("_005959_999")


def test_custom_id_empty_stem_falls_back():
    assert make_custom_id("///", 0, 0) == "video_000000_000"


def test_custom_id_rejects_negative_timestamp():
    with pytest.raises(ValueError, match="non-negative"):
        make_custom_id("lesson", -1, 0)


def test_build_requests_assigns_sequential_ids(tmp_path):
    frames = [
        (Frame(path=tmp_path / "a.jpg", timestamp_ms=0), tmp_path / "a_crop.jpg"),
        (Frame(path=tmp_path / "b.jpg", timestamp_ms=1000), tmp_path / "b_crop.jpg"),
    ]
    requests = build_requests(Path("lesson01.mp4"), frames)
    assert [r.custom_id for r in requests] == ["lesson01_000000_000", "lesson01_000001_001"]
    assert requests[0].video == "lesson01.mp4"
    assert requests[0].frame_path.endswith("a.jpg")
    assert requests[0].image_path.endswith("a_crop.jpg")


# --- request lines --------------------------------------------------------------------------


def test_request_line_shape(png):
    line = request_line(_request(image=str(png)), "gpt-5.4-mini")
    assert line["custom_id"] == "lesson01_000314_000"
    assert line["method"] == "POST"
    assert line["url"] == BATCH_ENDPOINT == "/v1/responses"
    body = line["body"]
    assert body["model"] == "gpt-5.4-mini"
    assert body["max_output_tokens"] == MAX_OUTPUT_TOKENS
    (message,) = body["input"]
    assert message["role"] == "user"
    text_part, image_part = message["content"]
    assert text_part == {"type": "input_text", "text": OCR_PROMPT}
    assert image_part["type"] == "input_image"
    assert image_part["image_url"].startswith("data:image/png;base64,")


def test_prompt_matches_issue_contract():
    assert "Do not infer missing lines" in OCR_PROMPT
    assert "Do not complete partial code" in OCR_PROMPT
    assert f"return exactly: {NO_CODE_SENTINEL}" in OCR_PROMPT


def test_write_batch_input_is_one_line_per_request(tmp_path, png):
    requests = [_request("a_000000_000", image=str(png)), _request("b_000001_001", image=str(png))]
    path = tmp_path / "batch_input.jsonl"
    write_batch_input(path, requests, "gpt-5.4-mini")
    lines = path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["custom_id"] for line in lines] == ["a_000000_000", "b_000001_001"]


def test_request_line_rejects_oversized_image(tmp_path):
    big = tmp_path / "big.png"
    big.write_bytes(b"\0" * (20 * 1024 * 1024 + 1))
    with pytest.raises(ValueError, match="20 MB"):
        request_line(_request(image=str(big)), "gpt-5.4-mini")


# --- submission -----------------------------------------------------------------------------


class FakeBatchClient:
    """Records files.create / batches.create / batches.retrieve / files.content calls."""

    def __init__(self, *, batch=None, file_texts=None):
        self.captured = {}
        self._batch = batch or SimpleNamespace(id="batch_abc123", status="completed")
        self._file_texts = file_texts or {}

        def files_create(*, file, purpose):
            self.captured["uploaded_bytes"] = file.read()
            self.captured["purpose"] = purpose
            return SimpleNamespace(id="file_in_1")

        def batches_create(**kwargs):
            self.captured["batches_create"] = kwargs
            return self._batch

        def batches_retrieve(batch_id):
            self.captured["retrieved"] = batch_id
            return self._batch

        def files_content(file_id):
            return SimpleNamespace(text=self._file_texts[file_id])

        self.files = SimpleNamespace(create=files_create, content=files_content)
        self.batches = SimpleNamespace(create=batches_create, retrieve=batches_retrieve)


def test_submit_batch_uploads_then_creates(tmp_path, png):
    path = tmp_path / "batch_input.jsonl"
    write_batch_input(path, [_request(image=str(png))], "gpt-5.4-mini")
    client = FakeBatchClient()

    batch_id = submit_batch(client, path)

    assert batch_id == "batch_abc123"
    assert client.captured["purpose"] == "batch"
    assert client.captured["uploaded_bytes"] == path.read_bytes()
    created = client.captured["batches_create"]
    assert created["input_file_id"] == "file_in_1"
    assert created["endpoint"] == "/v1/responses"
    assert created["completion_window"] == COMPLETION_WINDOW
    assert created["metadata"] == {"description": "code screenshot OCR"}


# --- manifest -------------------------------------------------------------------------------


def test_manifest_round_trip(tmp_path):
    requests = [_request()]
    path = tmp_path / "lesson01.batch.json"
    write_manifest(
        path, batch_id="batch_1", model="gpt-5.4-mini", video="lesson01.mp4", requests=requests
    )
    assert read_manifest(path) == ("batch_1", "gpt-5.4-mini", "lesson01.mp4", requests)


def test_read_manifest_rejects_garbage(tmp_path):
    path = tmp_path / "not_a_manifest.json"
    path.write_text('{"foo": 1}', encoding="utf-8")
    with pytest.raises(ValueError, match="not a valid batch manifest"):
        read_manifest(path)


# --- output parsing -------------------------------------------------------------------------


def test_parse_matches_by_custom_id_not_order():
    requests = [
        _request("lesson01_000000_000", ts=0),
        _request("lesson01_000001_001", ts=1000),
    ]
    # output arrives in reverse order
    raw = "\n".join(
        [
            _output_line("lesson01_000001_001", "y = 2"),
            _output_line("lesson01_000000_000", "x = 1"),
        ]
    )
    records = parse_batch_output(raw, requests)
    assert [(r["id"], r["text"]) for r in records] == [
        ("lesson01_000000_000", "x = 1"),
        ("lesson01_000001_001", "y = 2"),
    ]
    assert all(r["status"] == "ok" for r in records)


def test_parse_record_shape_matches_issue():
    record = parse_batch_output(
        _output_line("lesson01_000314_000", "import jax\nimport jax.numpy as jnp"),
        [_request()],
    )[0]
    assert record["id"] == "lesson01_000314_000"
    assert record["video"] == "lesson01.mp4"
    assert record["timestamp"] == "00:03:14.000"
    assert record["image_path"] == "crop.png"
    assert record["status"] == "ok"
    assert record["text"] == "import jax\nimport jax.numpy as jnp"


def test_parse_normalizes_no_code_sentinel():
    raw = _output_line("lesson01_000314_000", f"  {NO_CODE_SENTINEL}\n")
    (record,) = parse_batch_output(raw, [_request()])
    assert record["status"] == "no_code_visible"
    assert record["text"] == ""


def test_parse_strips_code_fences():
    raw = _output_line("lesson01_000314_000", "```python\nimport jax\n```")
    (record,) = parse_batch_output(raw, [_request()])
    assert record["text"] == "import jax"


def test_parse_missing_request_becomes_error():
    (record,) = parse_batch_output("", [_request()])
    assert record["status"] == "error"
    assert record["detail"] == "missing from batch output"
    assert record["text"] == ""


def test_parse_line_error_becomes_error():
    raw = json.dumps(
        {"custom_id": "lesson01_000314_000", "response": None, "error": {"message": "expired"}}
    )
    (record,) = parse_batch_output(raw, [_request()])
    assert record["status"] == "error"
    assert "expired" in record["detail"]


def test_parse_non_200_becomes_error():
    raw = _output_line("lesson01_000314_000", "irrelevant", status_code=500)
    (record,) = parse_batch_output(raw, [_request()])
    assert record["status"] == "error"
    assert record["detail"] == "HTTP 500"


def test_parse_incomplete_body_is_uncertain():
    raw = _output_line("lesson01_000314_000", "import ja", body_status="incomplete")
    (record,) = parse_batch_output(raw, [_request()])
    assert record["status"] == "uncertain"
    assert record["text"] == "import ja"
    assert "max_output_tokens" in record["detail"]


def test_parse_empty_text_is_uncertain():
    raw = _output_line("lesson01_000314_000", "   ")
    (record,) = parse_batch_output(raw, [_request()])
    assert record["status"] == "uncertain"
    assert record["detail"] == "empty model output"


def test_parse_reads_error_file_lines():
    error_raw = json.dumps(
        {"custom_id": "lesson01_000314_000", "response": None, "error": {"code": "rate_limited"}}
    )
    (record,) = parse_batch_output("", [_request()], error_raw=error_raw)
    assert record["status"] == "error"
    assert "rate_limited" in record["detail"]


def test_parse_ignores_unknown_and_malformed_lines():
    raw = "\n".join(
        [
            "{not json",
            _output_line("some_other_batch_000", "junk"),
            _output_line("lesson01_000314_000", "x = 1"),
        ]
    )
    (record,) = parse_batch_output(raw, [_request()])
    assert record["status"] == "ok"
    assert record["text"] == "x = 1"


def test_parse_first_line_wins_on_duplicate_custom_id():
    raw = "\n".join(
        [
            _output_line("lesson01_000314_000", "first"),
            _output_line("lesson01_000314_000", "second"),
        ]
    )
    (record,) = parse_batch_output(raw, [_request()])
    assert record["text"] == "first"


# --- records → extractions -------------------------------------------------------------------


def test_records_to_extractions_maps_usable_records():
    records = parse_batch_output(
        "\n".join(
            [
                _output_line("lesson01_000000_000", "x = 1"),
                _output_line("lesson01_000001_001", NO_CODE_SENTINEL),
                _output_line("lesson01_000002_002", "y = [?]", body_status="incomplete"),
            ]
        ),
        [
            _request("lesson01_000000_000", ts=0),
            _request("lesson01_000001_001", ts=1000),
            _request("lesson01_000002_002", ts=2000),
            _request("lesson01_000003_003", ts=3000),  # missing → error → skipped
        ],
    )
    extractions = records_to_extractions(records)
    assert [e.text for e in extractions] == ["x = 1", "y = [?]"]
    assert extractions[0].backend == "openai-batch"
    assert extractions[0].confidence == pytest.approx(0.9)
    assert extractions[0].frame.timestamp_ms == 0
    assert str(extractions[0].frame.path) == "frames/frame_000194.jpg"
    # uncertain records are capped at low confidence so merge prefers cleaner reads
    assert extractions[1].confidence <= 0.3


def test_write_records_is_jsonl(tmp_path):
    records = parse_batch_output(_output_line("lesson01_000314_000", "x = 1"), [_request()])
    path = tmp_path / "lesson01.ocr.jsonl"
    batch_ocr.write_records(path, records)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["text"] == "x = 1"
