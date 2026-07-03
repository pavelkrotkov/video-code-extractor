"""Command-line entry point: ``vce extract VIDEO`` runs the full pipeline.

This module is intentionally thin — it parses arguments, builds the backends and a
:class:`~vce.pipeline.PipelineConfig`, hands off to :class:`~vce.pipeline.Pipeline`, and turns the
stages' exceptions into clean one-line errors. All ordering and policy live in :mod:`vce.pipeline`.

Two-tier cost control: ``--backend`` chooses the *primary* (cheap) backend; when it is Apple Vision
the accurate vision backend is wired up as the escalation tier, used only for kept frames the
primary read with low confidence. Escalation needs an OpenAI key — when none is available it is
disabled (the run proceeds single-tier and says so), except when vision is itself the primary
backend, where a missing key is a hard error. The local ``macos-vision`` backend is macOS-only; on
other platforms it is a clean error pointing at the remote ``vision-gpt4v`` backend.

Bulk remote OCR runs through the OpenAI Batch API instead (issue #32): ``vce ocr-submit`` samples,
dedups, optionally crops, and submits one batch of screenshots; ``vce ocr-status`` polls it; and
``vce ocr-fetch`` saves the raw batch output, normalizes it into timestamped OCR JSONL records,
and (with ``--merge``) reuses the merge stage to write the final script + provenance.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

from vce import batch_ocr
from vce.backends.base import ExtractionBackend
from vce.backends.macos_vision import MacOSVisionBackend, UnsupportedPlatformError
from vce.backends.vision import VisionLLMBackend
from vce.cropping import crop_region
from vce.dedup import dedup_frames
from vce.frames import FFmpegNotFoundError, FrameExtractionError
from vce.merge import build_provenance, merge_results, write_provenance
from vce.ocr import DEFAULT_OCR_MODEL, resolve_ocr_model
from vce.pipeline import Pipeline, PipelineConfig, build_script, candidate_frames
from vce.types import BBox

MACOS_VISION = "macos-vision"
VISION = "vision-gpt4v"


class CLIError(RuntimeError):
    """A user-facing error with a message fit to print to stderr (no traceback)."""


def _parse_crop(value: str) -> BBox:
    """Parse a ``x,y,width,height`` crop string into a :class:`BBox` (all non-negative ints)."""
    parts = value.split(",")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("crop must be 'x,y,width,height'")
    try:
        x, y, w, h = (int(p) for p in parts)
    except ValueError:
        raise argparse.ArgumentTypeError("crop values must be integers") from None
    if x < 0 or y < 0 or w <= 0 or h <= 0:
        raise argparse.ArgumentTypeError("crop x,y must be >= 0 and width,height > 0")
    return BBox(x=x, y=y, width=w, height=h)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vce", description="Extract clean code from screencasts.")
    parser.add_argument("--version", action="store_true", help="print version and exit")
    sub = parser.add_subparsers(dest="command")

    extract = sub.add_parser("extract", help="extract code from a video")
    extract.add_argument("video", type=Path, help="path to the source video")
    extract.add_argument("--fps", type=float, default=1.0, help="frame sampling rate (default 1.0)")
    extract.add_argument(
        "--backend",
        choices=[MACOS_VISION, VISION],
        default=MACOS_VISION,
        help="primary extraction backend (default macos-vision)",
    )
    extract.add_argument(
        "--out", type=Path, default=Path("out"), help="output directory (default out/)"
    )
    extract.add_argument(
        "--score-threshold",
        type=float,
        default=0.4,
        help="drop frames scoring below this code-likeness (0..1, default 0.4)",
    )
    extract.add_argument(
        "--escalate-below",
        type=float,
        default=0.6,
        help="escalate to the vision backend below this primary confidence (0..1, default 0.6)",
    )
    extract.add_argument(
        "--no-escalate",
        action="store_true",
        help="disable the vision escalation tier (run on the primary backend only)",
    )
    extract.add_argument(
        "--crop",
        type=_parse_crop,
        default=None,
        metavar="X,Y,W,H",
        help="fixed code region to crop before extraction (pixels)",
    )
    extract.add_argument(
        "--scene-threshold",
        type=float,
        default=0.3,
        help="ffmpeg scene-change sensitivity (0..1, default 0.3)",
    )
    submit = sub.add_parser(
        "ocr-submit",
        help="sample, dedup, and submit a video's candidate frames as one OpenAI Batch OCR job",
    )
    submit.add_argument("video", type=Path, help="path to the source video")
    submit.add_argument("--fps", type=float, default=1.0, help="frame sampling rate (default 1.0)")
    submit.add_argument(
        "--scene-threshold",
        type=float,
        default=0.3,
        help="ffmpeg scene-change sensitivity (0..1, default 0.3)",
    )
    submit.add_argument(
        "--crop",
        type=_parse_crop,
        default=None,
        metavar="X,Y,W,H",
        help="fixed code region to crop before submitting (pixels)",
    )
    submit.add_argument(
        "--out", type=Path, default=Path("out"), help="output directory (default out/)"
    )
    submit.add_argument(
        "--model",
        default=None,
        help=f"OCR model (default $OPENAI_OCR_MODEL or {DEFAULT_OCR_MODEL})",
    )

    status = sub.add_parser("ocr-status", help="show the status of a submitted OCR batch")
    status.add_argument("batch", help="batch id, or the .batch.json manifest written by ocr-submit")

    fetch = sub.add_parser(
        "ocr-fetch", help="download a completed OCR batch and write normalized OCR JSONL records"
    )
    fetch.add_argument("manifest", type=Path, help="the .batch.json manifest written by ocr-submit")
    fetch.add_argument(
        "--merge",
        action="store_true",
        help="also merge the OCR records into a script + provenance sidecar",
    )

    # The OpenAI key is read only from $OPENAI_API_KEY — deliberately not a CLI flag, since secrets
    # passed as arguments leak into process listings (ps / /proc) and shell history (CWE-214).
    return parser


def _resolve_backends(
    args: argparse.Namespace,
) -> tuple[ExtractionBackend, ExtractionBackend | None, str | None]:
    """Build the primary and (optional) escalation backends from parsed args.

    Returns ``(primary, escalation, note)`` where ``note`` is a one-line heads-up to print (e.g.
    escalation disabled for want of a key), or ``None``. Raises :class:`CLIError` for the
    unrecoverable case: vision selected as the primary backend with no API key available.
    """
    api_key = os.environ.get("OPENAI_API_KEY")

    if args.backend == VISION:
        if not api_key:
            raise CLIError("the vision-gpt4v backend needs an OpenAI API key; set OPENAI_API_KEY")
        # Already the accurate backend — there is nothing more expensive to escalate to.
        return VisionLLMBackend(api_key=api_key), None, None

    # The local macos-vision backend depends on Apple's Vision framework; fail fast and clean on
    # other platforms, pointing the user at the remote vision-gpt4v backend instead of a traceback.
    if sys.platform != "darwin":
        raise CLIError(
            "the macos-vision backend requires macOS; on this platform run with "
            "--backend vision-gpt4v (needs OPENAI_API_KEY)"
        )

    primary: ExtractionBackend = MacOSVisionBackend()
    if args.no_escalate:
        return primary, None, "escalation disabled (--no-escalate); running on macos-vision only"
    if not api_key:
        return (
            primary,
            None,
            "no OpenAI API key found; vision escalation disabled (set OPENAI_API_KEY to enable)",
        )
    return primary, VisionLLMBackend(api_key=api_key), None


@contextmanager
def _clean_errors() -> Iterator[None]:
    """Translate the stages' expected exceptions into :class:`CLIError`.

    Wraps a whole command body so an expected error from *any* step — backend construction,
    config validation, or a pipeline stage — surfaces as a clean ``vce: error`` rather than a
    raw traceback. Shared by ``extract`` and the ``ocr-*`` commands.
    """
    try:
        yield
    except FFmpegNotFoundError as exc:
        raise CLIError(str(exc)) from exc
    except FrameExtractionError as exc:
        raise CLIError(f"frame extraction failed: {exc}") from exc
    except FileNotFoundError as exc:
        raise CLIError(str(exc)) from exc
    except OSError as exc:
        # e.g. PermissionError / disk full while creating the output dir or writing artifacts.
        raise CLIError(f"I/O error: {exc}") from exc
    except UnsupportedPlatformError as exc:
        # macos-vision invoked on a non-macOS host (e.g. via a direct backend call path).
        raise CLIError(str(exc)) from exc
    except ImportError as exc:
        # e.g. ocrmac isn't installed on macOS; the backend raises with install instructions.
        raise CLIError(str(exc)) from exc
    except ValueError as exc:
        # e.g. an out-of-range threshold rejected by PipelineConfig.
        raise CLIError(str(exc)) from exc


def _run_extract(args: argparse.Namespace) -> int:
    with _clean_errors():
        primary, escalation, note = _resolve_backends(args)
        if note:
            print(f"vce: {note}", file=sys.stderr)

        config = PipelineConfig(
            out_dir=args.out,
            fps=args.fps,
            scene_threshold=args.scene_threshold,
            score_threshold=args.score_threshold,
            escalate_below=args.escalate_below,
            crop=args.crop,
        )
        result = Pipeline(primary, config, escalation=escalation).run(args.video)

    s = result.stats
    mins, secs = divmod(int(s.total_time), 60)
    dedup_pct = round((1 - s.frames_after_dedup / s.frames_raw) * 100) if s.frames_raw else 0
    pass_pct = (
        round(s.frames_passed_scoring / s.frames_after_dedup * 100) if s.frames_after_dedup else 0
    )
    bar = "-" * 42
    time_str = f"{mins}m {secs:02d}s" if mins else f"{secs}s"
    print(f"\n{bar}")
    print(f"  Frames extracted:    {s.frames_raw:>8,}")
    print(f"  After dedup:         {s.frames_after_dedup:>8,}  ({dedup_pct}% removed)")
    print(f"  Passed scoring gate: {s.frames_passed_scoring:>8,}  ({pass_pct}%)")
    print(f"  Escalated:           {s.escalated_count:>8,}")
    print(f"  Snippets merged:     {s.snippets_merged:>8,}")
    print(f"  Output:  {result.script_path}")
    print(f"    lines: {s.output_lines:,}   chars: {s.output_chars:,}")
    print(f"  Time: {time_str}")
    print(bar)
    return 0


def _require_api_key() -> str:
    """The OpenAI key every ``ocr-*`` command needs, or a clean error saying how to provide it."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise CLIError("batch OCR needs an OpenAI API key; set OPENAI_API_KEY")
    return api_key


def _artifact_base(video_name: str) -> str:
    """Filename stem shared by a video's batch artifacts (mirrors the pipeline's naming)."""
    return Path(video_name).stem or "extracted"


def _run_ocr_submit(args: argparse.Namespace) -> int:
    from openai import OpenAIError

    api_key = _require_api_key()
    model = resolve_ocr_model(args.model)
    try:
        with _clean_errors():
            config = PipelineConfig(
                out_dir=args.out,
                fps=args.fps,
                scene_threshold=args.scene_threshold,
                crop=args.crop,
            )
            args.out.mkdir(parents=True, exist_ok=True)
            base = _artifact_base(args.video.name)

            print("[1/3] Extracting candidate frames...", file=sys.stderr)
            frames = candidate_frames(args.video, config)
            deduped = dedup_frames(frames, max_distance=config.dedup_max_distance)

            print(
                f"[2/3] Building requests for {len(deduped)} deduplicated frames...",
                file=sys.stderr,
            )
            crops_dir = args.out / f"{base}_crops"
            images = [
                (frame, crop_region(frame, args.crop, crops_dir) if args.crop else frame.path)
                for frame in deduped
            ]
            requests = batch_ocr.build_requests(args.video, images)
            if not requests:
                raise CLIError("no candidate frames were extracted; nothing to submit")
            input_path = args.out / f"{base}.batch_input.jsonl"
            batch_ocr.write_batch_input(input_path, requests, model)

            print(f"[3/3] Submitting batch of {len(requests)} screenshot(s)...", file=sys.stderr)
            client = batch_ocr.make_client(api_key)
            batch_id = batch_ocr.submit_batch(client, input_path)
            manifest_path = args.out / f"{base}.batch.json"
            batch_ocr.write_manifest(
                manifest_path,
                batch_id=batch_id,
                model=model,
                video=args.video.name,
                requests=requests,
            )
    except OpenAIError as exc:
        raise CLIError(f"OpenAI API error: {exc}") from exc

    print(f"Submitted batch {batch_id}: {len(requests)} screenshot(s), model {model}")
    print(f"  Manifest: {manifest_path}")
    print(f"  Check:    vce ocr-status {manifest_path}")
    print(f"  Fetch:    vce ocr-fetch {manifest_path}")
    return 0


def _run_ocr_status(args: argparse.Namespace) -> int:
    from openai import OpenAIError

    api_key = _require_api_key()
    try:
        with _clean_errors():
            manifest_path = Path(args.batch)
            if manifest_path.is_file():
                batch_id, _, _, _ = batch_ocr.read_manifest(manifest_path)
            else:
                batch_id = args.batch
            batch = batch_ocr.make_client(api_key).batches.retrieve(batch_id)
    except OpenAIError as exc:
        raise CLIError(f"OpenAI API error: {exc}") from exc

    print(f"batch {batch_id}: {batch.status}")
    counts = getattr(batch, "request_counts", None)
    if counts is not None:
        print(f"  requests: {counts.completed}/{counts.total} completed, {counts.failed} failed")
    return 0


def _run_ocr_fetch(args: argparse.Namespace) -> int:
    from openai import OpenAIError

    api_key = _require_api_key()
    try:
        with _clean_errors():
            batch_id, _, video, requests = batch_ocr.read_manifest(args.manifest)
            base = _artifact_base(video)
            out_dir = args.manifest.parent

            client = batch_ocr.make_client(api_key)
            batch = client.batches.retrieve(batch_id)
            if batch.status != "completed":
                raise CLIError(
                    f"batch {batch_id} is not complete: status={batch.status}; retry later "
                    f"(check with: vce ocr-status {args.manifest})"
                )

            # Keep the raw batch output (and error file, when present) verbatim for debugging.
            raw = client.files.content(batch.output_file_id).text if batch.output_file_id else ""
            raw_path = out_dir / f"{base}.batch_output.jsonl"
            raw_path.write_text(raw, encoding="utf-8")
            error_raw = ""
            if getattr(batch, "error_file_id", None):
                error_raw = client.files.content(batch.error_file_id).text
                (out_dir / f"{base}.batch_errors.jsonl").write_text(error_raw, encoding="utf-8")

            records = batch_ocr.parse_batch_output(raw, requests, error_raw=error_raw)
            records_path = out_dir / f"{base}.ocr.jsonl"
            batch_ocr.write_records(records_path, records)
    except OpenAIError as exc:
        raise CLIError(f"OpenAI API error: {exc}") from exc

    counts = Counter(record["status"] for record in records)
    statuses = (
        batch_ocr.STATUS_OK,
        batch_ocr.STATUS_NO_CODE,
        batch_ocr.STATUS_UNCERTAIN,
        batch_ocr.STATUS_ERROR,
    )
    summary = ", ".join(f"{counts.get(status, 0)} {status}" for status in statuses)
    print(f"Batch {batch_id} complete: {summary}")
    print(f"  OCR records: {records_path}")
    print(f"  Raw output:  {raw_path}")

    if args.merge:
        with _clean_errors():
            results = merge_results(batch_ocr.records_to_extractions(records))
            snippets = [r.snippet for r in results]
            script_path = out_dir / f"{base}.py"
            script_path.write_text(build_script(snippets), encoding="utf-8")
            provenance_path = out_dir / f"{base}.provenance.json"
            write_provenance(provenance_path, build_provenance(results))
        print(f"  Script:      {script_path}")
        print(f"  Provenance:  {provenance_path}")
    return 0


_COMMANDS = {
    "extract": _run_extract,
    "ocr-submit": _run_ocr_submit,
    "ocr-status": _run_ocr_status,
    "ocr-fetch": _run_ocr_fetch,
}


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.version:
        from vce import __version__

        print(__version__)
        return 0
    command = _COMMANDS.get(args.command or "")
    if command is None:
        print(
            "usage: vce {extract,ocr-submit,ocr-status,ocr-fetch} ...  (try 'vce --help')",
            file=sys.stderr,
        )
        return 2
    try:
        return command(args)
    except CLIError as exc:
        print(f"vce: error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
