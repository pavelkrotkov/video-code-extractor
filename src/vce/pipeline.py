"""Stage 0 — the end-to-end orchestration that ties every stage together.

This is the object the ``vce extract`` command drives. It owns the *ordering* and *config* of the
pipeline so :mod:`vce.cli` stays a thin argument-parsing shim (per the issue's refactor note). The
flow mirrors ``docs/architecture.md``::

    frames (fps + scene cuts)  ->  dedup  ->  [crop]  ->  extract  ->  score/gate  ->  merge

One subtlety on ordering: the code-likeness gate (:func:`vce.scoring.score_code_likeness`) scores a
frame *from its transcribed text*, so it cannot run before a first extraction exists. We therefore
crop and run the **cheap** backend first, gate on the resulting text, and only then — for frames
that survive the gate — apply the two-tier escalation. That keeps the expensive vision backend off
both the non-code frames (dropped by the gate) and the frames the cheap backend already read
confidently.

Two-tier cost control
---------------------
``primary`` is the cheap backend (Apple Vision on macOS by default). ``escalation`` is the accurate vision
backend, invoked only for kept frames whose primary confidence is below
:attr:`PipelineConfig.escalate_below`. When no escalation backend is wired up (e.g. no API key),
the pipeline runs single-tier on the primary backend alone.

Everything heavy (ffmpeg, OCR, the OpenAI client) lives behind injected callables/objects, so the
orchestration itself is unit-testable without touching the network or the disk-bound stages.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from vce.backends.base import ExtractionBackend
from vce.codequality import clean_transcription, is_suspect, reconcile_cluster
from vce.cropping import crop_region
from vce.dedup import dedup_frames
from vce.frames import extract_frames, scene_change_frames
from vce.merge import (
    DEFAULT_CONFLICT_MARGIN,
    DEFAULT_LOW_CONFIDENCE,
    DEFAULT_SIMILARITY,
    MergeResult,
    build_provenance,
    merge_results,
    write_provenance,
)
from vce.scoring import score_code_likeness
from vce.types import BBox, Extraction, Frame, MergedSnippet


@dataclass(frozen=True)
class PipelineConfig:
    """All tunable knobs for a pipeline run, in one place.

    Defaults are the same as each stage's own defaults so a bare ``vce extract VIDEO`` behaves
    like the individual stages do. Validated in :meth:`__post_init__` so a bad value fails fast
    with a clear message instead of deep inside a stage.
    """

    out_dir: Path
    fps: float = 1.0
    scene_threshold: float = 0.3
    dedup_max_distance: int = 4
    score_threshold: float = 0.4
    escalate_below: float = 0.6
    crop: BBox | None = None
    similarity_threshold: float = DEFAULT_SIMILARITY
    low_confidence_threshold: float = DEFAULT_LOW_CONFIDENCE
    conflict_margin: float = DEFAULT_CONFLICT_MARGIN

    def __post_init__(self) -> None:
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if not 0.0 < self.scene_threshold <= 1.0:
            raise ValueError("scene_threshold must be within (0, 1]")
        if self.dedup_max_distance < 0:
            raise ValueError("dedup_max_distance must be non-negative")
        # Every remaining knob is a [0, 1] probability/score; validate them uniformly so a bad
        # value fails here (before any ffmpeg work) instead of deep inside a downstream stage.
        for name in (
            "score_threshold",
            "escalate_below",
            "similarity_threshold",
            "low_confidence_threshold",
            "conflict_margin",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within [0, 1], got {value}")


@dataclass(frozen=True)
class PipelineResult:
    """Outcome of a run: where the artifacts landed plus the counts the CLI summarizes."""

    script_path: Path
    provenance_path: Path
    snippets: tuple[MergedSnippet, ...]
    frames_total: int
    frames_kept: int

    @property
    def num_snippets(self) -> int:
        return len(self.snippets)

    @property
    def num_flagged(self) -> int:
        """Count of snippets carrying a review note (low confidence, conflict, or unresolved).

        These are the snippets the run could not confidently present as clean — surfaced by the CLI
        so a local-only (``--no-escalate``) run identifies its low-quality output rather than
        passing it off silently (issue #24)."""
        return sum(1 for snippet in self.snippets if snippet.notes)


def _artifact_base(video: Path) -> str:
    """Filename stem shared by all of a run's artifacts (script, sidecar, and frame/crop dirs).

    Falls back to ``"extracted"`` when the video name has no stem (e.g. a bare ``.`` path).
    """
    return video.stem or "extracted"


def _candidate_frames(video: Path, config: PipelineConfig) -> list[Frame]:
    """fps-sampled and scene-cut frames, merged into one timeline-ordered list.

    The two sources are complementary (see :mod:`vce.frames`); a frame captured by both at the
    same timestamp is a near-duplicate that the dedup stage collapses. Sorting by
    ``(timestamp, path)`` gives dedup the timeline order it expects and keeps the run deterministic.

    Frames are written to a ``<video>_frames`` directory rather than a generic ``frames/``: the
    frame stages *clean* their target (unlinking pre-existing ``frame_*.jpg`` / ``scene_*.jpg``)
    before writing, so a per-video name keeps a run from deleting a user's unrelated images when
    ``--out`` points at an existing directory.
    """
    frames_dir = config.out_dir / f"{_artifact_base(video)}_frames"
    sampled = extract_frames(video, frames_dir, fps=config.fps)
    scenes = scene_change_frames(video, frames_dir, threshold=config.scene_threshold)
    combined = [*sampled, *scenes]
    combined.sort(key=lambda f: (f.timestamp_ms, str(f.path)))
    return combined


def _build_script(snippets: list[MergedSnippet]) -> str:
    """Concatenate snippet code into one clean script, blank-line separated, trailing newline.

    Empty snippets are skipped; an all-empty run yields an empty string (no stray blank lines).
    """
    parts = [s.code.rstrip("\n") for s in snippets if s.code.strip()]
    if not parts:
        return ""
    return "\n\n\n".join(parts) + "\n"


def _flag_unresolved(result: MergeResult) -> MergeResult:
    """Append a review note to a snippet whose reconciled code still fails validation.

    After reconciliation and cleaning, a snippet whose code is still code-like-but-invalid (e.g. an
    OCR-mangled cell that no escalation tier resolved) is flagged rather than presented as clean.
    The note rides on :attr:`~vce.types.MergedSnippet.notes` (alongside merge's existing
    low-confidence / conflict notes) and is surfaced by the CLI; the cleaned code and provenance
    schema are untouched. A clean, valid snippet is returned unchanged.
    """
    code = result.snippet.code
    if not is_suspect(code):
        return result
    note = "unresolved: code-like text did not validate; re-run with vision escalation to repair"
    existing = result.snippet.notes
    snippet = replace(result.snippet, notes=f"{existing}; {note}" if existing else note)
    return replace(result, snippet=snippet)


class Pipeline:
    """Runs the full extract→merge pipeline for a single video.

    Backends are injected so the orchestration is testable offline and the two-tier policy is
    explicit: ``primary`` is always used; ``escalation`` (when provided) re-reads only the kept,
    low-confidence frames.
    """

    def __init__(
        self,
        primary: ExtractionBackend,
        config: PipelineConfig,
        *,
        escalation: ExtractionBackend | None = None,
    ) -> None:
        self._primary = primary
        self._escalation = escalation
        self._config = config

    def _image_for(self, frame: Frame, crops_dir: Path) -> Path:
        """The image a backend should read: the configured crop, or the full frame."""
        if self._config.crop is None:
            return frame.path
        return crop_region(frame, self._config.crop, crops_dir)

    def _should_escalate(self, extraction: Extraction) -> bool:
        """Escalate when the primary read is low-confidence *or* code-like but structurally suspect.

        Vision's recognition confidence alone misses high-confidence-but-broken OCR — mangled
        brackets, notebook output bleeding into code — so we OR it with an independent code-validity
        signal (:func:`vce.codequality.is_suspect`). A transcription that passed the code-likeness
        gate but does not parse, or carries notebook chrome / rendered output, is therefore eligible
        for the accuracy tier even at confidence 1.0 (issue #24).
        """
        return extraction.confidence < self._config.escalate_below or is_suspect(extraction.text)

    def _extract_kept(self, frame: Frame, image: Path) -> Extraction | None:
        """Cheap-extract, gate on code-likeness, then escalate if the cheap read was unsure or suspect.

        Returns ``None`` for a frame that fails the code-likeness gate (so it is dropped before any
        expensive escalation). Otherwise returns the extraction to merge — the escalation backend's
        if it ran, else the primary's.
        """
        extraction = self._primary.extract(image, frame)
        if score_code_likeness(frame, extraction.text).score < self._config.score_threshold:
            return None
        if self._escalation is not None and self._should_escalate(extraction):
            escalated = self._escalation.extract(image, frame)
            # Only adopt the escalation if it *also* reads as code. Vision is the accuracy tier, so
            # a code-like result is preferred — but an empty or off-target response must not discard
            # the primary transcription that already passed the gate (which would silently drop the
            # snippet from the script and provenance).
            if score_code_likeness(frame, escalated.text).score >= self._config.score_threshold:
                extraction = escalated
        return extraction

    def run(self, video: Path) -> PipelineResult:
        """Execute every stage for ``video`` and write the script + provenance sidecar.

        Raises whatever the underlying stages raise (e.g. :class:`~vce.frames.FFmpegNotFoundError`,
        ``FileNotFoundError`` for a missing video, ``ImportError`` for a missing backend extra);
        :mod:`vce.cli` turns those into clean user-facing messages.
        """
        video = Path(video)
        config = self._config
        base = _artifact_base(video)
        # Per-video crop dir, for the same reason as the frames dir (see _candidate_frames).
        crops_dir = config.out_dir / f"{base}_crops"

        # Create the output directory up front so an unwritable ``--out`` fails fast (as a clean
        # OSError the CLI translates) before any expensive frame extraction / OCR work is wasted.
        config.out_dir.mkdir(parents=True, exist_ok=True)

        frames = _candidate_frames(video, config)
        deduped = dedup_frames(frames, max_distance=config.dedup_max_distance)

        extractions: list[Extraction] = []
        for frame in deduped:
            extraction = self._extract_kept(frame, self._image_for(frame, crops_dir))
            if extraction is not None:
                extractions.append(extraction)

        # Reconcile overlapping captures of one cell into the most complete valid variant, cleaned
        # of notebook chrome / rendered output, via the merge step's reconciliation seam. Cluster on
        # output-stripped code (``cluster_text``) so two captures of one cell that differ only in
        # their rendered output still group together instead of duplicating in the script.
        results = merge_results(
            extractions,
            similarity_threshold=config.similarity_threshold,
            low_confidence_threshold=config.low_confidence_threshold,
            conflict_margin=config.conflict_margin,
            merge_fn=reconcile_cluster,
            cluster_text=clean_transcription,
        )
        results = [_flag_unresolved(r) for r in results]
        snippets = [r.snippet for r in results]

        script_path = config.out_dir / f"{base}.py"
        provenance_path = config.out_dir / f"{base}.provenance.json"
        script_path.write_text(_build_script(snippets), encoding="utf-8")
        write_provenance(provenance_path, build_provenance(results))

        return PipelineResult(
            script_path=script_path,
            provenance_path=provenance_path,
            snippets=tuple(snippets),
            frames_total=len(deduped),
            frames_kept=len(extractions),
        )
