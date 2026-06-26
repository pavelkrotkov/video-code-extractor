"""Backend benchmark: measure extraction backends against hand-labeled frames.

The key design fork in this project is the extraction backend (PaddleOCR vs a Vision-LLM).
Rather than pick by vibes, this module *measures*: it runs each backend over a small set of
hand-labeled frames and scores the recovered text against ground truth with two complementary
metrics, then names a recommended default.

Metrics
-------
**Normalized Levenshtein similarity** — character-level edit distance turned into a similarity
in ``[0, 1]``::

    similarity = 1 - distance / max(len(pred), len(truth))

with two empty strings defined as a perfect ``1.0`` (there is nothing to disagree about). This
rewards "almost right" transcriptions: a one-character slip on a long line barely dents the score.

**Token-level accuracy** — fraction of *ground-truth* tokens that the prediction also contains,
counted with multiplicity (a multiset intersection)::

    token_acc = |multiset(pred_tokens) ∩ multiset(truth_tokens)| / |truth_tokens|

Tokens are whitespace-or-symbol runs from a simple ``\\w+|[^\\w\\s]`` regex, so ``x=1`` tokenizes
to ``["x", "=", "1"]`` and punctuation counts. Two empty strings are defined as ``1.0``; a
non-empty prediction against empty truth is ``0.0`` (nothing to recover). This metric is harsher
than Levenshtein on word-level substitutions (a hallucinated identifier scores zero for that
token) which is exactly the failure mode we care about for code.

Design
------
Metric computation (:func:`score_extraction`, :func:`levenshtein_distance`) is deliberately kept
separate from aggregation (:func:`run_benchmark`) and from formatting (:func:`format_report`), so
each can be tested and reused in isolation. Backends are injected via the
:class:`~vce.backends.base.ExtractionBackend` protocol, so the benchmark runs fully offline with
fake backends in tests; real OCR/vision runs are gated on the caller wiring up a real backend
(and, for the vision backend, an API key).
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean

from vce.backends.base import ExtractionBackend
from vce.types import Frame

# Split into word runs and individual symbols: "x=1" -> ["x", "=", "1"].
_TOKEN_RE = re.compile(r"\w+|[^\w\s]")


def levenshtein_distance(a: str, b: str) -> int:
    """Return the character-level Levenshtein (edit) distance between ``a`` and ``b``.

    Pure Python, no dependency. Counts the minimum number of single-character insertions,
    deletions, or substitutions to turn ``a`` into ``b``. Uses a rolling two-row DP for
    ``O(len(a) * len(b))`` time and ``O(min(len))`` space.
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    # Keep the inner loop over the shorter string to minimize the row width.
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            current.append(
                min(
                    previous[j] + 1,  # deletion
                    current[j - 1] + 1,  # insertion
                    previous[j - 1] + cost,  # substitution
                )
            )
        previous = current
    return previous[-1]


def _normalized_levenshtein(pred: str, truth: str) -> float:
    """``1 - distance / max(len)`` similarity in ``[0, 1]``; two empty strings are ``1.0``."""
    longest = max(len(pred), len(truth))
    if longest == 0:
        return 1.0
    return 1.0 - levenshtein_distance(pred, truth) / longest


def _tokenize(text: str) -> list[str]:
    """Split ``text`` into word and symbol tokens (see module docstring)."""
    return _TOKEN_RE.findall(text)


def _token_accuracy(pred: str, truth: str) -> float:
    """Fraction of ground-truth tokens recovered, as a multiset intersection (see module docstring)."""
    truth_tokens = _tokenize(truth)
    if not truth_tokens:
        # Nothing to recover: perfect only if the prediction is also empty of tokens.
        return 1.0 if not _tokenize(pred) else 0.0
    pred_counts = Counter(_tokenize(pred))
    matched = sum(min(count, pred_counts[token]) for token, count in Counter(truth_tokens).items())
    return matched / len(truth_tokens)


def score_extraction(pred: str, truth: str) -> dict[str, float]:
    """Score one predicted transcription ``pred`` against ground truth ``truth``.

    Returns ``{"levenshtein": <similarity 0..1>, "token_acc": <accuracy 0..1>}``. See the module
    docstring for the precise definition of each metric.
    """
    return {
        "levenshtein": _normalized_levenshtein(pred, truth),
        "token_acc": _token_accuracy(pred, truth),
    }


@dataclass(frozen=True)
class LabeledFrame:
    """A frame paired with its hand-labeled ground-truth text."""

    frame: Frame
    truth: str


@dataclass(frozen=True)
class BackendReport:
    """Aggregated scores for a single backend across all labeled frames."""

    name: str
    mean_levenshtein: float
    mean_token_acc: float
    n_frames: int

    @property
    def aggregate(self) -> float:
        """Single ranking score: the mean of the two metrics (both already in ``[0, 1]``)."""
        return (self.mean_levenshtein + self.mean_token_acc) / 2


@dataclass(frozen=True)
class BenchmarkReport:
    """The full benchmark result: per-backend scores plus the recommended winner.

    Holds data only — turning it into a table is :func:`format_report`'s job, keeping computation
    and presentation separate.
    """

    backends: tuple[BackendReport, ...]
    winner: str | None


def load_labeled_frames(directory: Path | str) -> list[LabeledFrame]:
    """Load labeled frames from ``directory``: each ``<stem>.png`` paired with ``<stem>.txt``.

    The ``.txt`` file holds the ground-truth transcription. PNGs without a sibling ``.txt`` are
    skipped. Trailing whitespace/newlines on the ground truth are stripped. Frames are returned
    sorted by filename for deterministic ordering.
    """
    directory = Path(directory)
    frames: list[LabeledFrame] = []
    for png in sorted(directory.glob("*.png")):
        truth_path = png.with_suffix(".txt")
        if not truth_path.exists():
            continue
        truth = truth_path.read_text(encoding="utf-8").rstrip()
        frames.append(LabeledFrame(frame=Frame(path=png, timestamp_ms=0), truth=truth))
    return frames


def run_benchmark(
    backends: list[ExtractionBackend],
    labeled_frames: list[LabeledFrame],
) -> BenchmarkReport:
    """Run every backend over every labeled frame and aggregate the scores.

    For each backend, calls ``.extract()`` on each labeled frame, scores the recovered text against
    the ground truth, and averages each metric across frames. The recommended winner is the backend
    with the highest :attr:`BackendReport.aggregate` score (``None`` if there are no backends, or no
    labeled frames to evaluate — every backend scoring ``0.0`` over zero frames is not a real win).
    Backends are run in the given order, which is also their order in the report.
    """
    reports: list[BackendReport] = []
    for backend in backends:
        levs: list[float] = []
        accs: list[float] = []
        for lf in labeled_frames:
            extraction = backend.extract(lf.frame.path, lf.frame)
            scores = score_extraction(extraction.text, lf.truth)
            levs.append(scores["levenshtein"])
            accs.append(scores["token_acc"])
        reports.append(
            BackendReport(
                name=backend.name,
                mean_levenshtein=fmean(levs) if levs else 0.0,
                mean_token_acc=fmean(accs) if accs else 0.0,
                n_frames=len(labeled_frames),
            )
        )
    winner = max(reports, key=lambda r: r.aggregate).name if reports and labeled_frames else None
    return BenchmarkReport(backends=tuple(reports), winner=winner)


def format_report(report: BenchmarkReport) -> str:
    """Render a :class:`BenchmarkReport` as a fixed-width comparison table naming the winner.

    Pure presentation: takes the already-computed report and returns a printable string, so the
    aggregation logic in :func:`run_benchmark` carries no formatting concerns.
    """
    header = ("backend", "levenshtein", "token_acc", "aggregate")
    rows = [
        (
            b.name,
            f"{b.mean_levenshtein:.3f}",
            f"{b.mean_token_acc:.3f}",
            f"{b.aggregate:.3f}",
        )
        for b in report.backends
    ]
    widths = [
        max(len(header[i]), *(len(r[i]) for r in rows)) if rows else len(header[i])
        for i in range(4)
    ]

    def fmt(cols: tuple[str, ...]) -> str:
        return "  ".join(col.ljust(widths[i]) for i, col in enumerate(cols))

    lines = [fmt(header), "  ".join("-" * w for w in widths)]
    lines.extend(fmt(r) for r in rows)
    n = report.backends[0].n_frames if report.backends else 0
    lines.append("")
    if report.winner is None:
        lines.append("Winner: (no backends benchmarked)")
    else:
        lines.append(f"Winner: {report.winner}  (recommended default, over {n} labeled frame(s))")
    return "\n".join(lines)


# Default location of the committed fixture set.
_DEFAULT_FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "bench"


def main(
    backends: list[ExtractionBackend] | None = None,
    labeled_frames: list[LabeledFrame] | None = None,
    fixtures_dir: Path | str = _DEFAULT_FIXTURES,
) -> int:
    """Run the benchmark and print the comparison table and winner.

    Importable and testable: pass ``backends`` and ``labeled_frames`` to run fully offline with
    fake backends. With no ``backends``, it wires up the real PaddleOCR and Vision-LLM backends —
    those require the ``paddle`` extra / an OpenAI API key respectively, so the offline path
    (explicit backends) is what the test suite exercises.
    """
    if labeled_frames is None:
        labeled_frames = load_labeled_frames(fixtures_dir)
    if not labeled_frames:
        print(f"Error: No labeled frames found in {fixtures_dir}", file=sys.stderr)
        return 1
    if backends is None:  # pragma: no cover - real backends need network / heavy extra
        from vce.backends.paddle import PaddleOCRBackend
        from vce.backends.vision import VisionLLMBackend

        backends = [PaddleOCRBackend(), VisionLLMBackend()]
    report = run_benchmark(backends, labeled_frames)
    print(format_report(report))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
