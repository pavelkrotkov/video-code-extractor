"""Tests for the thin CLI shim: arg parsing, backend wiring, and error translation.

The pipeline itself is covered in ``test_pipeline.py``; here we monkeypatch the ``Pipeline`` the
CLI constructs so we can assert on what the CLI does with arguments and exceptions without running
any real stage.
"""

import sys

import pytest

import vce
from vce import cli
from vce.frames import FFmpegNotFoundError
from vce.types import BBox, PipelineStats


@pytest.fixture
def on_macos(monkeypatch):
    """Pretend we're on macOS so the default macos-vision backend resolves without a real Mac."""
    monkeypatch.setattr(sys, "platform", "darwin")


_FAKE_STATS = PipelineStats(
    frames_raw=8,
    frames_after_dedup=8,
    frames_passed_scoring=5,
    escalated_count=0,
    snippets_merged=2,
    output_lines=10,
    output_chars=200,
    stage_times=(),
    total_time=1.5,
)


class _FakeResult:
    script_path = "out/lesson.py"
    provenance_path = "out/lesson.provenance.json"
    num_snippets = 2
    frames_kept = 5
    frames_total = 8
    stats = _FAKE_STATS


def _install_fake_pipeline(monkeypatch, *, run=None):
    """Replace ``cli.Pipeline`` with a recorder; return the dict capturing construction args."""
    captured = {}

    class FakePipeline:
        def __init__(self, primary, config, *, escalation=None):
            captured["primary"] = primary
            captured["config"] = config
            captured["escalation"] = escalation

        def run(self, video):
            captured["video"] = video
            if run is not None:
                return run()
            return _FakeResult()

    monkeypatch.setattr(cli, "Pipeline", FakePipeline)
    return captured


# --- top level ----------------------------------------------------------------------------


def test_version(capsys):
    assert cli.main(["--version"]) == 0
    assert capsys.readouterr().out.strip() == vce.__version__


def test_no_command_is_usage_error(capsys):
    assert cli.main([]) == 2
    err = capsys.readouterr().err
    assert "usage: vce" in err
    assert "extract" in err
    assert "ocr-submit" in err


# --- arg parsing --------------------------------------------------------------------------


def test_extract_defaults():
    args = cli.build_parser().parse_args(["extract", "video.mp4"])
    assert args.command == "extract"
    assert str(args.video) == "video.mp4"
    assert args.fps == 1.0
    assert args.backend == cli.MACOS_VISION
    assert str(args.out) == "out"
    assert args.score_threshold == 0.4
    assert args.crop is None


def test_extract_backend_choice_and_crop():
    args = cli.build_parser().parse_args(
        ["extract", "v.mp4", "--backend", "vision-gpt4v", "--crop", "10,20,300,400"]
    )
    assert args.backend == cli.VISION
    assert args.crop == BBox(10, 20, 300, 400)


@pytest.mark.parametrize("bad", ["1,2,3", "a,b,c,d", "0,0,0,10", "-1,0,10,10"])
def test_crop_rejects_bad_values(bad):
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["extract", "v.mp4", "--crop", bad])


# --- backend wiring -----------------------------------------------------------------------


def test_macos_vision_primary_with_key_enables_escalation(monkeypatch, capsys, on_macos):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    captured = _install_fake_pipeline(monkeypatch)

    assert cli.main(["extract", "v.mp4"]) == 0
    assert captured["primary"].name == cli.MACOS_VISION
    assert captured["escalation"] is not None
    assert captured["escalation"].name == cli.VISION


def test_macos_vision_primary_without_key_disables_escalation(monkeypatch, capsys, on_macos):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    captured = _install_fake_pipeline(monkeypatch)

    assert cli.main(["extract", "v.mp4"]) == 0
    assert captured["escalation"] is None
    assert "escalation disabled" in capsys.readouterr().err


def test_non_macos_default_backend_is_clean_error(monkeypatch, capsys):
    # The local backend is macOS-only; off-Mac the CLI must point at vision-gpt4v, not crash.
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    _install_fake_pipeline(monkeypatch)

    assert cli.main(["extract", "v.mp4"]) == 1
    err = capsys.readouterr().err
    assert "vce: error:" in err
    assert "requires macOS" in err
    assert "vision-gpt4v" in err


def test_no_escalate_flag(monkeypatch, capsys, on_macos):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    captured = _install_fake_pipeline(monkeypatch)

    assert cli.main(["extract", "v.mp4", "--no-escalate"]) == 0
    assert captured["escalation"] is None
    assert "--no-escalate" in capsys.readouterr().err


def test_vision_primary_without_key_is_error(monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    _install_fake_pipeline(monkeypatch)

    assert cli.main(["extract", "v.mp4", "--backend", "vision-gpt4v"]) == 1
    assert "needs an OpenAI API key" in capsys.readouterr().err


def test_config_threaded_from_args(monkeypatch, on_macos):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    captured = _install_fake_pipeline(monkeypatch)

    cli.main(["extract", "v.mp4", "--fps", "2", "--score-threshold", "0.7", "--out", "build"])
    config = captured["config"]
    assert config.fps == 2.0
    assert config.score_threshold == 0.7
    assert str(config.out_dir) == "build"


# --- error translation --------------------------------------------------------------------


def test_missing_ffmpeg_is_clean_error(monkeypatch, capsys, on_macos):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    def boom():
        raise FFmpegNotFoundError("ffmpeg not found on PATH; install it to extract frames")

    _install_fake_pipeline(monkeypatch, run=boom)

    assert cli.main(["extract", "v.mp4"]) == 1
    err = capsys.readouterr().err
    assert "vce: error:" in err
    assert "ffmpeg not found" in err


def test_missing_video_is_clean_error(monkeypatch, capsys, on_macos):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    def boom():
        raise FileNotFoundError("video not found: v.mp4")

    _install_fake_pipeline(monkeypatch, run=boom)

    assert cli.main(["extract", "v.mp4"]) == 1
    assert "video not found" in capsys.readouterr().err


def test_bad_threshold_is_clean_error(monkeypatch, capsys, on_macos):
    # Config validation fails before any stage runs, so this needs no fake pipeline / ffmpeg.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert cli.main(["extract", "v.mp4", "--score-threshold", "5"]) == 1
    err = capsys.readouterr().err
    assert "vce: error:" in err
    assert "score_threshold must be within" in err


def test_io_error_is_clean_error(monkeypatch, capsys, on_macos):
    # A non-FileNotFound OSError (e.g. PermissionError writing outputs) gets the I/O error path.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    def boom():
        raise PermissionError("output directory is read-only")

    _install_fake_pipeline(monkeypatch, run=boom)

    assert cli.main(["extract", "v.mp4"]) == 1
    err = capsys.readouterr().err
    assert "vce: error:" in err
    assert "I/O error" in err


def test_missing_ocrmac_dependency_is_clean_error(monkeypatch, capsys, on_macos):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    def boom():
        raise ImportError(
            "ocrmac is required for the macos-vision backend on macOS: pip install ocrmac"
        )

    _install_fake_pipeline(monkeypatch, run=boom)

    assert cli.main(["extract", "v.mp4"]) == 1
    assert "ocrmac is required" in capsys.readouterr().err


# --- batch OCR commands ---------------------------------------------------------------------


@pytest.fixture
def batch_env(monkeypatch, tmp_path):
    """API key set, model env unset, and frame/dedup stages faked with two tiny real images."""
    from PIL import Image

    from vce.types import Frame

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("OPENAI_OCR_MODEL", raising=False)

    frames = []
    for i in range(2):
        path = tmp_path / f"frame_{i:06d}.png"
        Image.new("RGB", (8, 8), "white").save(path)
        frames.append(Frame(path=path, timestamp_ms=i * 1000))

    monkeypatch.setattr(cli, "candidate_frames", lambda video, config: frames)
    monkeypatch.setattr(cli, "dedup_frames", lambda fs, max_distance: list(fs))
    return tmp_path


class _FakeOCRClient:
    """Just enough OpenAI client surface for the ocr-* commands."""

    def __init__(self, *, status="completed", output_text="", error_text=None):
        import json as _json
        from types import SimpleNamespace

        self.created = {}
        batch = SimpleNamespace(
            id="batch_abc123",
            status=status,
            output_file_id="file_out" if output_text is not None else None,
            error_file_id="file_err" if error_text is not None else None,
            request_counts=SimpleNamespace(total=2, completed=2, failed=0),
        )
        texts = {"file_out": output_text or "", "file_err": error_text or ""}

        def files_create(*, file, purpose):
            self.created["purpose"] = purpose
            self.created["uploaded"] = file.read()
            return SimpleNamespace(id="file_in")

        def batches_create(**kwargs):
            self.created["batch"] = kwargs
            return batch

        self.files = SimpleNamespace(
            create=files_create, content=lambda fid: SimpleNamespace(text=texts[fid])
        )
        self.batches = SimpleNamespace(create=batches_create, retrieve=lambda bid: batch)
        self._json = _json


def _install_fake_client(monkeypatch, client):
    from vce import batch_ocr

    monkeypatch.setattr(batch_ocr, "make_client", lambda api_key=None: client)
    return client


def _ocr_output_line(custom_id, text):
    import json as _json

    return _json.dumps(
        {
            "custom_id": custom_id,
            "response": {
                "status_code": 200,
                "body": {
                    "status": "completed",
                    "output": [
                        {"type": "message", "content": [{"type": "output_text", "text": text}]}
                    ],
                },
            },
            "error": None,
        }
    )


def test_ocr_submit_requires_api_key(monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert cli.main(["ocr-submit", "lesson.mp4"]) == 1
    assert "OPENAI_API_KEY" in capsys.readouterr().err


def test_ocr_fetch_requires_api_key(monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert cli.main(["ocr-fetch", "out/lesson.batch.json"]) == 1
    assert "OPENAI_API_KEY" in capsys.readouterr().err


def test_ocr_submit_writes_input_and_manifest(monkeypatch, capsys, batch_env):
    import json

    client = _install_fake_client(monkeypatch, _FakeOCRClient())
    out = batch_env / "out"

    assert cli.main(["ocr-submit", "lesson01.mp4", "--out", str(out)]) == 0

    input_lines = (out / "lesson01.batch_input.jsonl").read_text().splitlines()
    assert len(input_lines) == 2
    first = json.loads(input_lines[0])
    assert first["url"] == "/v1/responses"
    assert first["body"]["model"] == "gpt-5.4-mini"
    assert client.created["purpose"] == "batch"
    assert client.created["batch"]["endpoint"] == "/v1/responses"

    manifest = json.loads((out / "lesson01.batch.json").read_text())
    assert manifest["batch_id"] == "batch_abc123"
    assert [r["custom_id"] for r in manifest["requests"]] == [
        "lesson01_000000_000",
        "lesson01_000001_001",
    ]
    assert "batch_abc123" in capsys.readouterr().out


def test_ocr_submit_model_env_and_flag(monkeypatch, capsys, batch_env):
    import json

    _install_fake_client(monkeypatch, _FakeOCRClient())
    monkeypatch.setenv("OPENAI_OCR_MODEL", "env-model")
    out = batch_env / "out"

    assert cli.main(["ocr-submit", "lesson01.mp4", "--out", str(out)]) == 0
    manifest = json.loads((out / "lesson01.batch.json").read_text())
    assert manifest["model"] == "env-model"

    assert cli.main(["ocr-submit", "lesson01.mp4", "--out", str(out), "--model", "flag-model"]) == 0
    manifest = json.loads((out / "lesson01.batch.json").read_text())
    assert manifest["model"] == "flag-model"


def test_ocr_status_accepts_manifest_or_id(monkeypatch, capsys, batch_env):
    from vce import batch_ocr

    _install_fake_client(monkeypatch, _FakeOCRClient(status="in_progress"))
    manifest = batch_env / "lesson01.batch.json"
    batch_ocr.write_manifest(
        manifest, batch_id="batch_abc123", model="m", video="lesson01.mp4", requests=[]
    )

    assert cli.main(["ocr-status", str(manifest)]) == 0
    assert "batch_abc123: in_progress" in capsys.readouterr().out

    assert cli.main(["ocr-status", "batch_abc123"]) == 0
    assert "batch_abc123: in_progress" in capsys.readouterr().out


def test_ocr_fetch_incomplete_batch_is_clean_error(monkeypatch, capsys, batch_env):
    from vce import batch_ocr

    _install_fake_client(monkeypatch, _FakeOCRClient(status="in_progress"))
    manifest = batch_env / "lesson01.batch.json"
    batch_ocr.write_manifest(
        manifest, batch_id="batch_abc123", model="m", video="lesson01.mp4", requests=[]
    )

    assert cli.main(["ocr-fetch", str(manifest)]) == 1
    err = capsys.readouterr().err
    assert "not complete" in err
    assert "in_progress" in err


def test_ocr_fetch_writes_raw_and_records(monkeypatch, capsys, batch_env):
    import json

    from vce import batch_ocr
    from vce.batch_ocr import OCRRequest

    raw = "\n".join(
        [
            # deliberately out of input order; fetch must reorder by manifest
            _ocr_output_line("lesson01_000001_001", "y = 2"),
            _ocr_output_line("lesson01_000000_000", "NO_CODE_VISIBLE"),
        ]
    )
    _install_fake_client(monkeypatch, _FakeOCRClient(output_text=raw))
    manifest = batch_env / "lesson01.batch.json"
    requests = [
        OCRRequest("lesson01_000000_000", "lesson01.mp4", 0, "f0.png", "f0.png"),
        OCRRequest("lesson01_000001_001", "lesson01.mp4", 1000, "f1.png", "f1.png"),
    ]
    batch_ocr.write_manifest(
        manifest, batch_id="batch_abc123", model="m", video="lesson01.mp4", requests=requests
    )

    assert cli.main(["ocr-fetch", str(manifest)]) == 0

    assert (batch_env / "lesson01.batch_output.jsonl").read_text() == raw
    records = [
        json.loads(line) for line in (batch_env / "lesson01.ocr.jsonl").read_text().splitlines()
    ]
    assert [(r["id"], r["status"]) for r in records] == [
        ("lesson01_000000_000", "no_code_visible"),
        ("lesson01_000001_001", "ok"),
    ]
    out = capsys.readouterr().out
    assert "1 ok" in out
    assert "1 no_code_visible" in out


def test_ocr_fetch_merge_writes_script_and_provenance(monkeypatch, capsys, batch_env):
    import json

    from vce import batch_ocr
    from vce.batch_ocr import OCRRequest

    raw = _ocr_output_line("lesson01_000000_000", "import os\nprint(os.getcwd())")
    _install_fake_client(monkeypatch, _FakeOCRClient(output_text=raw))
    manifest = batch_env / "lesson01.batch.json"
    batch_ocr.write_manifest(
        manifest,
        batch_id="batch_abc123",
        model="m",
        video="lesson01.mp4",
        requests=[OCRRequest("lesson01_000000_000", "lesson01.mp4", 0, "f0.png", "f0.png")],
    )

    assert cli.main(["ocr-fetch", str(manifest), "--merge"]) == 0

    assert (batch_env / "lesson01.py").read_text() == "import os\nprint(os.getcwd())\n"
    provenance = json.loads((batch_env / "lesson01.provenance.json").read_text())
    assert provenance[0]["raw_ocr"] == "import os\nprint(os.getcwd())"


def _manifest_with_one_request(batch_env):
    from vce import batch_ocr
    from vce.batch_ocr import OCRRequest

    manifest = batch_env / "lesson01.batch.json"
    batch_ocr.write_manifest(
        manifest,
        batch_id="batch_abc123",
        model="m",
        video="lesson01.mp4",
        requests=[OCRRequest("lesson01_000000_000", "lesson01.mp4", 0, "f0.png", "f0.png")],
    )
    return manifest


def test_ocr_fetch_failed_batch_is_terminal_error(monkeypatch, capsys, batch_env):
    _install_fake_client(monkeypatch, _FakeOCRClient(status="failed"))
    manifest = _manifest_with_one_request(batch_env)

    assert cli.main(["ocr-fetch", str(manifest)]) == 1
    err = capsys.readouterr().err
    assert "will never produce results" in err
    assert "retry later" not in err


def test_ocr_fetch_expired_batch_fetches_partial_results(monkeypatch, capsys, batch_env):
    import json

    raw = _ocr_output_line("lesson01_000000_000", "x = 1")
    _install_fake_client(monkeypatch, _FakeOCRClient(status="expired", output_text=raw))
    manifest = _manifest_with_one_request(batch_env)

    assert cli.main(["ocr-fetch", str(manifest)]) == 0
    captured = capsys.readouterr()
    assert "fetching partial results" in captured.err
    records = [
        json.loads(line) for line in (batch_env / "lesson01.ocr.jsonl").read_text().splitlines()
    ]
    assert records[0]["status"] == "ok"


def test_ocr_fetch_expired_batch_without_results_is_error(monkeypatch, capsys, batch_env):
    _install_fake_client(
        monkeypatch, _FakeOCRClient(status="expired", output_text=None, error_text=None)
    )
    manifest = _manifest_with_one_request(batch_env)

    assert cli.main(["ocr-fetch", str(manifest)]) == 1
    assert "no results to fetch" in capsys.readouterr().err


def test_ocr_fetch_merge_gates_non_code_records(monkeypatch, capsys, batch_env):
    # Prose the model emits for a non-code slide (instead of the exact sentinel) must not
    # become a script snippet: --merge applies the pipeline's code-likeness gate.
    prose = "Welcome to the course! Today we will learn about machine learning together."
    raw = "\n".join(
        [
            _ocr_output_line("lesson01_000000_000", "import os\nprint(os.getcwd())"),
            _ocr_output_line("lesson01_000001_001", prose),
        ]
    )
    _install_fake_client(monkeypatch, _FakeOCRClient(output_text=raw))

    from vce import batch_ocr
    from vce.batch_ocr import OCRRequest

    manifest = batch_env / "lesson01.batch.json"
    batch_ocr.write_manifest(
        manifest,
        batch_id="batch_abc123",
        model="m",
        video="lesson01.mp4",
        requests=[
            OCRRequest("lesson01_000000_000", "lesson01.mp4", 0, "f0.png", "f0.png"),
            OCRRequest("lesson01_000001_001", "lesson01.mp4", 1000, "f1.png", "f1.png"),
        ],
    )

    assert cli.main(["ocr-fetch", str(manifest), "--merge"]) == 0
    script = (batch_env / "lesson01.py").read_text()
    assert "import os" in script
    assert "Welcome to the course" not in script


@pytest.mark.parametrize("bad", ["4", "-1"])
def test_ocr_fetch_rejects_out_of_range_score_threshold(monkeypatch, capsys, batch_env, bad):
    _install_fake_client(monkeypatch, _FakeOCRClient())
    manifest = _manifest_with_one_request(batch_env)

    assert cli.main(["ocr-fetch", str(manifest), "--merge", "--score-threshold", bad]) == 1
    assert "score_threshold must be within [0, 1]" in capsys.readouterr().err
