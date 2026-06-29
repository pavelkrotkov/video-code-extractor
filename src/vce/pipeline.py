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

import sys
import time
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from vce.backends.base import ExtractionBackend
from vce.cropping import crop_region
from vce.dedup import dedup_frames
from vce.frames import extract_frames, scene_change_frames
from vce.merge import (
    DEFAULT_CONFLICT_MARGIN,
    DEFAULT_LOW_CONFIDENCE,
    DEFAULT_SIMILARITY,
    build_provenance,
    merge_results,
    write_provenance,
)
from vce.scoring import score_code_likeness
from vce.types import BBox, Extraction, Frame, MergedSnippet, PipelineStats


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
    stats: PipelineStats

    @property
    def num_snippets(self) -> int:
        return len(self.snippets)


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

    def run(self, video: Path) -> PipelineResult:
        """Execute every stage for ``video`` and write the script + provenance sidecar.

        Raises whatever the underlying stages raise (e.g. :class:`~vce.frames.FFmpegNotFoundError`,
        ``FileNotFoundError`` for a missing video, ``ImportError`` for a missing backend extra);
        :mod:`vce.cli` turns those into clean user-facing messages.
        """
        t_total_start = time.perf_counter()
        video = Path(video)
        config = self._config
        base = _artifact_base(video)
        crops_dir = config.out_dir / f"{base}_crops"

        # Create the output directory up front so an unwritable ``--out`` fails fast (as a clean
        # OSError the CLI translates) before any expensive frame extraction / OCR work is wasted.
        config.out_dir.mkdir(parents=True, exist_ok=True)

        # Stage 1/5: Frame extraction
        print("[1/5] Extracting frames...", file=sys.stderr)
        t0 = time.perf_counter()
        frames = _candidate_frames(video, config)
        t_frames = time.perf_counter() - t0

        # Stage 2/5: Deduplication
        print(f"[2/5] Deduplicating {len(frames)} frames...", file=sys.stderr)
        t0 = time.perf_counter()
        deduped = dedup_frames(frames, max_distance=config.dedup_max_distance)
        t_dedup = time.perf_counter() - t0

        # Stage 3/5: OCR extraction + scoring gate (cropping is inline via _image_for)
        print(f"[3/5] Extracting code from {len(deduped)} frames...", file=sys.stderr)
        t0 = time.perf_counter()
        passed: list[Extraction] = []
        needs_escalation: list[tuple[Frame, Path, Extraction]] = []
        with tqdm(deduped, desc="  OCR", unit="frame", file=sys.stderr) as pbar:
            for frame in pbar:
                image = self._image_for(frame, crops_dir)
                extraction = self._primary.extract(image, frame)
                if score_code_likeness(frame, extraction.text).score < config.score_threshold:
                    continue
                if self._escalation is not None and extraction.confidence < config.escalate_below:
                    needs_escalation.append((frame, image, extraction))
                else:
                    passed.append(extraction)
        t_ocr = time.perf_counter() - t0

        # Stage 4/5: Escalation
        escalated_count = 0
        t0 = time.perf_counter()
        if needs_escalation and self._escalation is not None:
            print(
                f"[4/5] Escalating {len(needs_escalation)} low-confidence frames...",
                file=sys.stderr,
            )
            with tqdm(needs_escalation, desc="  Escalate", unit="frame", file=sys.stderr) as pbar:
                for frame, image, primary_ext in pbar:
                    escalated = self._escalation.extract(image, frame)
                    escalated_count += 1
                    if score_code_likeness(frame, escalated.text).score >= config.score_threshold:
                        passed.append(escalated)
                    else:
                        passed.append(primary_ext)
        else:
            print("[4/5] No escalation needed.", file=sys.stderr)
        t_escalate = time.perf_counter() - t0

        extractions = passed

        # Stage 5/5: Merge
        print(f"[5/5] Merging {len(extractions)} extraction(s)...", file=sys.stderr)
        t0 = time.perf_counter()
        results = merge_results(
            extractions,
            similarity_threshold=config.similarity_threshold,
            low_confidence_threshold=config.low_confidence_threshold,
            conflict_margin=config.conflict_margin,
        )
        snippets = [r.snippet for r in results]
        t_merge = time.perf_counter() - t0

        script_path = config.out_dir / f"{base}.py"
        provenance_path = config.out_dir / f"{base}.provenance.json"
        script_text = _build_script(snippets)
        script_path.write_text(script_text, encoding="utf-8")
        write_provenance(provenance_path, build_provenance(results))

        total_time = time.perf_counter() - t_total_start
        stats = PipelineStats(
            frames_raw=len(frames),
            frames_after_dedup=len(deduped),
            frames_passed_scoring=len(extractions),
            escalated_count=escalated_count,
            snippets_merged=len(snippets),
            output_lines=script_text.count("\n"),
            output_chars=len(script_text),
            stage_times=(
                ("frames", t_frames),
                ("dedup", t_dedup),
                ("ocr", t_ocr),
                ("escalate", t_escalate),
                ("merge", t_merge),
            ),
            total_time=total_time,
        )

        return PipelineResult(
            script_path=script_path,
            provenance_path=provenance_path,
            snippets=tuple(snippets),
            frames_total=len(deduped),
            frames_kept=len(extractions),
            stats=stats,
        )
