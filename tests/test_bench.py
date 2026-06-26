from pathlib import Path

import pytest

from vce.bench import (
    BackendReport,
    BenchmarkReport,
    LabeledFrame,
    format_report,
    levenshtein_distance,
    load_labeled_frames,
    main,
    run_benchmark,
    score_extraction,
)
from vce.types import Extraction, Frame

FIXTURES = Path(__file__).parent / "fixtures" / "bench"
FRAME = Frame(path=Path("f.png"), timestamp_ms=0)


# --- edit distance -------------------------------------------------------


@pytest.mark.parametrize(
    "a,b,expected",
    [
        ("", "", 0),
        ("abc", "abc", 0),
        ("abc", "abd", 1),  # one substitution
        ("abc", "ab", 1),  # one deletion
        ("ab", "abc", 1),  # one insertion
        ("kitten", "sitting", 3),  # classic example
        ("", "abc", 3),
    ],
)
def test_levenshtein_distance(a, b, expected):
    assert levenshtein_distance(a, b) == expected


# --- metrics -------------------------------------------------------------


def test_score_identical_is_perfect():
    s = score_extraction("import os", "import os")
    assert s["levenshtein"] == pytest.approx(1.0)
    assert s["token_acc"] == pytest.approx(1.0)


def test_score_both_empty_is_perfect():
    s = score_extraction("", "")
    assert s["levenshtein"] == pytest.approx(1.0)
    assert s["token_acc"] == pytest.approx(1.0)


def test_score_one_char_diff():
    # "cat" vs "car": distance 1, max len 3 -> similarity 1 - 1/3
    s = score_extraction("cat", "car")
    assert s["levenshtein"] == pytest.approx(1 - 1 / 3)


def test_score_completely_different():
    s = score_extraction("abc", "xyz")
    assert s["levenshtein"] == pytest.approx(0.0)


def test_token_accuracy_partial():
    # 1 of 2 truth tokens recovered
    s = score_extraction("import os", "import sys")
    assert s["token_acc"] == pytest.approx(0.5)


def test_token_accuracy_pred_empty():
    s = score_extraction("", "import os")
    assert s["token_acc"] == pytest.approx(0.0)


# --- fake backends -------------------------------------------------------


class PerfectBackend:
    """Returns the ground truth verbatim (used to verify it wins)."""

    name = "perfect"

    def __init__(self, truths: dict[Path, str]):
        self._truths = truths

    def extract(self, image_path: Path, frame: Frame) -> Extraction:
        return Extraction(
            frame=frame, text=self._truths[image_path], confidence=1.0, backend=self.name
        )


class LossyBackend:
    """Drops everything but the first token, so it scores poorly."""

    name = "lossy"

    def extract(self, image_path: Path, frame: Frame) -> Extraction:
        return Extraction(frame=frame, text="", confidence=0.0, backend=self.name)


class FailingBackend:
    """Raises on extract, mimicking a missing API key / extra / network error."""

    name = "failing"

    def extract(self, image_path: Path, frame: Frame) -> Extraction:
        raise RuntimeError("backend unavailable")


@pytest.fixture
def labeled():
    return load_labeled_frames(FIXTURES)


# --- fixtures loading ----------------------------------------------------


def test_load_labeled_frames(labeled):
    assert len(labeled) == 3
    for lf in labeled:
        assert isinstance(lf, LabeledFrame)
        assert lf.frame.path.exists()
        assert lf.truth  # non-empty ground truth


# --- run_benchmark -------------------------------------------------------


def test_run_benchmark_perfect_backend_wins(labeled):
    perfect = PerfectBackend({lf.frame.path: lf.truth for lf in labeled})
    lossy = LossyBackend()
    report = run_benchmark([perfect, lossy], labeled)

    assert isinstance(report, BenchmarkReport)
    assert report.winner == "perfect"
    by_name = {b.name: b for b in report.backends}
    assert by_name["perfect"].mean_levenshtein == pytest.approx(1.0)
    assert by_name["perfect"].mean_token_acc == pytest.approx(1.0)
    assert by_name["lossy"].mean_levenshtein < by_name["perfect"].mean_levenshtein


def test_run_benchmark_records_per_backend(labeled):
    perfect = PerfectBackend({lf.frame.path: lf.truth for lf in labeled})
    report = run_benchmark([perfect], labeled)
    assert len(report.backends) == 1
    assert isinstance(report.backends[0], BackendReport)
    assert report.backends[0].n_frames == 3


def test_run_benchmark_empty_backends_has_no_winner(labeled):
    report = run_benchmark([], labeled)
    assert report.winner is None
    assert report.backends == ()


def test_run_benchmark_skips_failing_backend(labeled, capsys):
    perfect = PerfectBackend({lf.frame.path: lf.truth for lf in labeled})
    report = run_benchmark([FailingBackend(), perfect], labeled)
    # the failing backend is dropped from the report; the working one still wins
    names = {b.name for b in report.backends}
    assert names == {"perfect"}
    assert report.winner == "perfect"
    assert "skipping backend 'failing'" in capsys.readouterr().err


def test_run_benchmark_empty_labeled_frames_has_no_winner():
    # No frames to evaluate: every backend scores 0.0, so there is no real winner.
    lossy = LossyBackend()
    report = run_benchmark([lossy], [])
    assert report.winner is None
    assert len(report.backends) == 1
    assert report.backends[0].mean_levenshtein == 0.0
    assert report.backends[0].mean_token_acc == 0.0


# --- reporting (kept separate from computation) --------------------------


def test_format_report_contains_table_and_winner(labeled):
    perfect = PerfectBackend({lf.frame.path: lf.truth for lf in labeled})
    lossy = LossyBackend()
    report = run_benchmark([perfect, lossy], labeled)
    text = format_report(report)
    assert "perfect" in text
    assert "lossy" in text
    assert "levenshtein" in text.lower()
    assert "perfect" in text.lower()
    # winner is named
    assert "winner" in text.lower()


def test_main_runs_offline_with_fake_backends(capsys, labeled):
    perfect = PerfectBackend({lf.frame.path: lf.truth for lf in labeled})
    lossy = LossyBackend()
    rc = main(backends=[perfect, lossy], labeled_frames=labeled)
    assert rc == 0
    out = capsys.readouterr().out
    assert "perfect" in out
    assert "winner" in out.lower()


def test_main_with_empty_labeled_frames_returns_error(capsys):
    rc = main(backends=[], labeled_frames=[])
    assert rc == 1
    err = capsys.readouterr().err
    assert "Error: No labeled frames found" in err


def test_main_with_missing_fixtures_dir_returns_error(capsys, tmp_path):
    rc = main(backends=[], fixtures_dir=tmp_path / "nope")
    assert rc == 1
    err = capsys.readouterr().err
    assert "fixtures directory does not exist" in err


def test_main_returns_error_when_all_backends_fail(capsys, labeled):
    # Every backend is skipped, so nothing is benchmarked -> non-zero exit.
    rc = main(backends=[FailingBackend()], labeled_frames=labeled)
    assert rc == 1
    err = capsys.readouterr().err
    assert "no backend was successfully benchmarked" in err
