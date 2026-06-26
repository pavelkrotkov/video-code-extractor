"""Command-line entry point: ``vce extract VIDEO`` runs the full pipeline.

This module is intentionally thin — it parses arguments, builds the backends and a
:class:`~vce.pipeline.PipelineConfig`, hands off to :class:`~vce.pipeline.Pipeline`, and turns the
stages' exceptions into clean one-line errors. All ordering and policy live in :mod:`vce.pipeline`.

Two-tier cost control: ``--backend`` chooses the *primary* (cheap) backend; when it is PaddleOCR
the accurate vision backend is wired up as the escalation tier, used only for kept frames the
primary read with low confidence. Escalation needs an OpenAI key — when none is available it is
disabled (the run proceeds single-tier and says so), except when vision is itself the primary
backend, where a missing key is a hard error.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from vce.backends.base import ExtractionBackend
from vce.backends.paddle import PaddleOCRBackend
from vce.backends.vision import VisionLLMBackend
from vce.frames import FFmpegNotFoundError, FrameExtractionError
from vce.pipeline import Pipeline, PipelineConfig
from vce.types import BBox

PADDLE = "paddleocr"
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
        choices=[PADDLE, VISION],
        default=PADDLE,
        help="primary extraction backend (default paddleocr)",
    )
    extract.add_argument("--out", type=Path, default=Path("."), help="output directory (default .)")
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
    extract.add_argument(
        "--api-key",
        default=None,
        help="OpenAI API key for the vision backend (defaults to $OPENAI_API_KEY)",
    )
    return parser


def _resolve_backends(
    args: argparse.Namespace,
) -> tuple[ExtractionBackend, ExtractionBackend | None, str | None]:
    """Build the primary and (optional) escalation backends from parsed args.

    Returns ``(primary, escalation, note)`` where ``note`` is a one-line heads-up to print (e.g.
    escalation disabled for want of a key), or ``None``. Raises :class:`CLIError` for the
    unrecoverable case: vision selected as the primary backend with no API key available.
    """
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")

    if args.backend == VISION:
        if not api_key:
            raise CLIError(
                "the vision-gpt4v backend needs an OpenAI API key; pass --api-key or set "
                "OPENAI_API_KEY"
            )
        # Already the accurate backend — there is nothing more expensive to escalate to.
        return VisionLLMBackend(api_key=api_key), None, None

    primary: ExtractionBackend = PaddleOCRBackend()
    if args.no_escalate:
        return primary, None, "escalation disabled (--no-escalate); running on paddleocr only"
    if not api_key:
        return (
            primary,
            None,
            "no OpenAI API key found; vision escalation disabled (set OPENAI_API_KEY to enable)",
        )
    return primary, VisionLLMBackend(api_key=api_key), None


def _run_extract(args: argparse.Namespace) -> int:
    # The whole setup-and-run sequence is wrapped so an expected error from *any* step — backend
    # construction, config validation, or a pipeline stage — surfaces as a clean ``vce: error``
    # rather than a raw traceback.
    try:
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
    except FFmpegNotFoundError as exc:
        raise CLIError(str(exc)) from exc
    except FrameExtractionError as exc:
        raise CLIError(f"frame extraction failed: {exc}") from exc
    except FileNotFoundError as exc:
        raise CLIError(str(exc)) from exc
    except OSError as exc:
        # e.g. PermissionError / disk full while creating the output dir or writing artifacts.
        raise CLIError(f"I/O error: {exc}") from exc
    except ImportError as exc:
        # e.g. the paddle extra isn't installed; the backend raises with install instructions.
        raise CLIError(str(exc)) from exc
    except ValueError as exc:
        # e.g. an out-of-range threshold rejected by PipelineConfig.
        raise CLIError(str(exc)) from exc

    print(
        f"Wrote {result.num_snippets} snippet(s) from {result.frames_kept}/{result.frames_total} "
        f"kept frame(s) to:\n  {result.script_path}\n  {result.provenance_path}"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.version:
        from vce import __version__

        print(__version__)
        return 0
    if args.command != "extract":
        print("usage: vce extract VIDEO [options]  (try 'vce --help')", file=sys.stderr)
        return 2
    try:
        return _run_extract(args)
    except CLIError as exc:
        print(f"vce: error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
