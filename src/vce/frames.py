"""Stage 1 — candidate frame extraction (fps sampling + scene-change frames).

Both sources are complementary, not alternatives: fps sampling catches code that flashes
briefly inside a single shot, while scene detection catches slide/editor cuts. We shell out
to ffmpeg (no extra Python dependency); timestamps come from the sampling rate (fps mode) or
from ffmpeg's ``showinfo`` ``pts_time`` (scene mode).
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

from vce.types import Frame

# ``-?`` so negative pts (B-frame offsets / edit lists) keep their sign.
_PTS_TIME_RE = re.compile(r"\bpts_time:(-?[0-9.]+)")


class FFmpegNotFoundError(RuntimeError):
    """Raised when the ffmpeg binary is not available on PATH."""


class FrameExtractionError(RuntimeError):
    """Raised when ffmpeg exits non-zero; carries ffmpeg's stderr for diagnosis."""


def _require_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe is None:
        raise FFmpegNotFoundError("ffmpeg not found on PATH; install it to extract frames")
    return exe


def _run_ffmpeg(cmd: list[str]) -> str:
    """Run an ffmpeg ``cmd`` and return its decoded stderr.

    ``check=True`` raises ``CalledProcessError`` on failure, but its default message hides the
    captured stderr, so we re-raise as :class:`FrameExtractionError` with ffmpeg's actual output
    surfaced for debugging. ``encoding="utf-8"`` keeps decoding stable across platforms regardless
    of the system locale.
    """
    try:
        proc = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.CalledProcessError as exc:
        raise FrameExtractionError(f"ffmpeg failed: {exc.stderr}") from exc
    return proc.stderr


def _timestamp_for(index: int, fps: float) -> int:
    """Source timestamp in ms of the ``index``-th (1-based) frame sampled at ``fps``."""
    return round((index - 1) * 1000.0 / fps)


def _prepare_out_dir(video: Path, out_dir: Path, glob: str) -> Path:
    """Validate ``video`` exists and return a clean ``out_dir`` with stale ``glob`` files removed.

    Removing pre-existing matches keeps ``sorted(glob(...))`` aligned with the frames ffmpeg
    writes this run, so timestamps are never mismatched against leftovers from a previous run.
    """
    video = Path(video)
    if not video.is_file():
        raise FileNotFoundError(f"video not found: {video}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob(glob):
        stale.unlink()
    return out_dir


def extract_frames(video: Path, out_dir: Path, *, fps: float = 1.0) -> list[Frame]:
    """Sample ``video`` at ``fps`` into timestamped JPEG frames under ``out_dir``."""
    if fps <= 0:
        raise ValueError("fps must be positive")
    ffmpeg = _require_ffmpeg()
    out_dir = _prepare_out_dir(video, out_dir, "frame_*.jpg")
    pattern = out_dir / "frame_%06d.jpg"
    _run_ffmpeg(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video),
            "-vf",
            f"fps={fps}",
            str(pattern),
        ]
    )
    return [
        Frame(path=path, timestamp_ms=_timestamp_for(index, fps))
        for index, path in enumerate(sorted(out_dir.glob("frame_*.jpg")), start=1)
    ]


def scene_change_frames(video: Path, out_dir: Path, *, threshold: float = 0.3) -> list[Frame]:
    """Extract frames at detected scene cuts, timestamped from ffmpeg's ``pts_time``."""
    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be in (0, 1]")
    ffmpeg = _require_ffmpeg()
    out_dir = _prepare_out_dir(video, out_dir, "scene_*.jpg")
    pattern = out_dir / "scene_%06d.jpg"
    stderr = _run_ffmpeg(
        [
            ffmpeg,
            "-hide_banner",
            "-y",
            "-i",
            str(video),
            "-vf",
            f"select='gt(scene,{threshold})',showinfo",
            "-vsync",
            "vfr",
            str(pattern),
        ]
    )
    # Only read pts_time from showinfo's own lines: a stray "pts_time:" in a file path or in
    # stream metadata must not be mistaken for a frame timestamp.
    times_ms: list[int] = []
    for line in stderr.splitlines():
        if "showinfo" not in line:
            continue
        match = _PTS_TIME_RE.search(line)
        if match:
            times_ms.append(round(float(match.group(1)) * 1000))
    paths = sorted(out_dir.glob("scene_*.jpg"))
    if len(paths) != len(times_ms):
        raise FrameExtractionError(
            f"scene frame/timestamp mismatch: {len(paths)} files but {len(times_ms)} timestamps"
        )
    return [Frame(path=path, timestamp_ms=ts) for path, ts in zip(paths, times_ms, strict=True)]
