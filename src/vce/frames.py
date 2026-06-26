"""Stage 1 — candidate frame extraction (fps sampling + scene-change frames).

Both sources are complementary, not alternatives: fps sampling catches code that flashes
briefly inside a single shot, while scene detection catches slide/editor cuts. We shell out
to ffmpeg (no extra Python dependency).

Both modes derive timestamps from ffmpeg's ``showinfo`` ``pts_time`` (the real stream timeline)
rather than from the sample index, so an input with a non-zero starting PTS (trimmed clips,
streams delayed behind audio) tags the same moment identically across both sources — which the
merge stage relies on.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from collections.abc import Iterable
from pathlib import Path

from vce.types import Frame

# ``-?`` so negative pts (B-frame offsets / edit lists) are parsed; they are clamped to 0 below.
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
    of the system locale. ``showinfo`` logs at info level, so callers must not pass
    ``-loglevel error`` when they need the timestamps.
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


def _parse_pts_ms(stderr: str) -> list[int]:
    """Frame timestamps in ms (clamped to >= 0) parsed from ffmpeg ``showinfo`` log lines.

    Only ``showinfo`` frame lines (which carry an ``n:`` index) are read, so a stray
    ``pts_time:`` inside a file path or stream metadata cannot be mistaken for a timestamp.
    Negative pts are clamped to 0 — provenance never points before the stream start, and
    :attr:`vce.types.Frame.timecode` assumes a non-negative value.
    """
    times: list[int] = []
    for line in stderr.splitlines():
        if "showinfo" not in line or " n:" not in line:
            continue
        match = _PTS_TIME_RE.search(line)
        if match:
            times.append(max(0, round(float(match.group(1)) * 1000)))
    return times


def _sorted_by_index(paths: Iterable[Path]) -> list[Path]:
    """Sort frame files by their numeric suffix (robust past the zero-padding width)."""
    return sorted(paths, key=lambda p: int(p.stem.split("_")[-1]))


def _validate_video(video: Path) -> Path:
    """Return ``video`` as a Path, raising ``FileNotFoundError`` if it isn't an existing file."""
    video = Path(video)
    if not video.is_file():
        raise FileNotFoundError(f"video not found: {video}")
    return video


def _prepare_out_dir(out_dir: Path, glob: str) -> Path:
    """Return a clean ``out_dir`` with stale ``glob`` files removed.

    Removing pre-existing matches keeps the sampled files aligned with the frames ffmpeg writes
    this run, so timestamps are never mismatched against leftovers from a previous run.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob(glob):
        stale.unlink(missing_ok=True)  # tolerate the glob->unlink race
    return out_dir


def _pair_frames(paths: list[Path], times_ms: list[int], kind: str) -> list[Frame]:
    if len(paths) != len(times_ms):
        raise FrameExtractionError(
            f"{kind} frame/timestamp mismatch: {len(paths)} files but {len(times_ms)} timestamps"
        )
    return [Frame(path=path, timestamp_ms=ts) for path, ts in zip(paths, times_ms, strict=True)]


def extract_frames(video: Path, out_dir: Path, *, fps: float = 1.0) -> list[Frame]:
    """Sample ``video`` at ``fps`` into timestamped JPEG frames under ``out_dir``."""
    if fps <= 0:
        raise ValueError("fps must be positive")
    video = _validate_video(video)  # validate inputs before checking for the ffmpeg binary
    ffmpeg = _require_ffmpeg()
    out_dir = _prepare_out_dir(out_dir, "frame_*.jpg")
    pattern = out_dir / "frame_%06d.jpg"
    stderr = _run_ffmpeg(
        [
            ffmpeg,
            "-hide_banner",
            "-y",
            "-i",
            str(video),
            "-vf",
            f"fps={fps},showinfo",
            "-vsync",
            "vfr",
            str(pattern),
        ]
    )
    paths = _sorted_by_index(out_dir.glob("frame_*.jpg"))
    return _pair_frames(paths, _parse_pts_ms(stderr), "fps")


def scene_change_frames(video: Path, out_dir: Path, *, threshold: float = 0.3) -> list[Frame]:
    """Extract frames at detected scene cuts, timestamped from ffmpeg's ``pts_time``."""
    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be in (0, 1]")
    video = _validate_video(video)  # validate inputs before checking for the ffmpeg binary
    ffmpeg = _require_ffmpeg()
    out_dir = _prepare_out_dir(out_dir, "scene_*.jpg")
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
    paths = _sorted_by_index(out_dir.glob("scene_*.jpg"))
    return _pair_frames(paths, _parse_pts_ms(stderr), "scene")
