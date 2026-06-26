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
