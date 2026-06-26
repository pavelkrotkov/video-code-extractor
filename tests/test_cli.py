"""Tests for the thin CLI shim: arg parsing, backend wiring, and error translation.

The pipeline itself is covered in ``test_pipeline.py``; here we monkeypatch the ``Pipeline`` the
CLI constructs so we can assert on what the CLI does with arguments and exceptions without running
any real stage.
"""

import pytest

import vce
from vce import cli
from vce.frames import FFmpegNotFoundError
from vce.types import BBox


class _FakeResult:
    script_path = "out/lesson.py"
    provenance_path = "out/lesson.provenance.json"
    num_snippets = 2
    frames_kept = 5
    frames_total = 8


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
    assert "usage: vce extract" in capsys.readouterr().err


# --- arg parsing --------------------------------------------------------------------------


def test_extract_defaults():
    args = cli.build_parser().parse_args(["extract", "video.mp4"])
    assert args.command == "extract"
    assert str(args.video) == "video.mp4"
    assert args.fps == 1.0
    assert args.backend == cli.PADDLE
    assert str(args.out) == "."
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


def test_paddle_primary_with_key_enables_escalation(monkeypatch, capsys):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    captured = _install_fake_pipeline(monkeypatch)

    assert cli.main(["extract", "v.mp4"]) == 0
    assert captured["primary"].name == cli.PADDLE
    assert captured["escalation"] is not None
    assert captured["escalation"].name == cli.VISION


def test_paddle_primary_without_key_disables_escalation(monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    captured = _install_fake_pipeline(monkeypatch)

    assert cli.main(["extract", "v.mp4"]) == 0
    assert captured["escalation"] is None
    assert "escalation disabled" in capsys.readouterr().err


def test_no_escalate_flag(monkeypatch, capsys):
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


def test_config_threaded_from_args(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    captured = _install_fake_pipeline(monkeypatch)

    cli.main(["extract", "v.mp4", "--fps", "2", "--score-threshold", "0.7", "--out", "build"])
    config = captured["config"]
    assert config.fps == 2.0
    assert config.score_threshold == 0.7
    assert str(config.out_dir) == "build"


# --- error translation --------------------------------------------------------------------


def test_missing_ffmpeg_is_clean_error(monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    def boom():
        raise FFmpegNotFoundError("ffmpeg not found on PATH; install it to extract frames")

    _install_fake_pipeline(monkeypatch, run=boom)

    assert cli.main(["extract", "v.mp4"]) == 1
    err = capsys.readouterr().err
    assert "vce: error:" in err
    assert "ffmpeg not found" in err


def test_missing_video_is_clean_error(monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    def boom():
        raise FileNotFoundError("video not found: v.mp4")

    _install_fake_pipeline(monkeypatch, run=boom)

    assert cli.main(["extract", "v.mp4"]) == 1
    assert "video not found" in capsys.readouterr().err


def test_bad_threshold_is_clean_error(monkeypatch, capsys):
    # Config validation fails before any stage runs, so this needs no fake pipeline / ffmpeg.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert cli.main(["extract", "v.mp4", "--score-threshold", "5"]) == 1
    err = capsys.readouterr().err
    assert "vce: error:" in err
    assert "score_threshold must be within" in err


def test_missing_paddle_extra_is_clean_error(monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    def boom():
        raise ImportError("PaddleOCR is not installed. Install the optional extra: ...")

    _install_fake_pipeline(monkeypatch, run=boom)

    assert cli.main(["extract", "v.mp4"]) == 1
    assert "PaddleOCR is not installed" in capsys.readouterr().err
