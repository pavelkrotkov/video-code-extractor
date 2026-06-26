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

_PTS_TIME_RE = re.compile(r"pts_time:([0-9.]+)")


class FFmpegNotFoundError(RuntimeError):
    """Raised when the ffmpeg binary is not available on PATH."""


def _require_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe is None:
        raise FFmpegNotFoundError("ffmpeg not found on PATH; install it to extract frames")
    return exe


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
    subprocess.run(
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
        ],
        check=True,
        capture_output=True,
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
    proc = subprocess.run(
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
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    times_ms = [round(float(t) * 1000) for t in _PTS_TIME_RE.findall(proc.stderr)]
    paths = sorted(out_dir.glob("scene_*.jpg"))
    # Counts must agree: showinfo emits one pts_time per selected frame, each saved as one file.
    # strict=True fails loud if that invariant ever breaks rather than silently mispairing.
    return [Frame(path=path, timestamp_ms=ts) for path, ts in zip(paths, times_ms, strict=True)]
