"""Shared pytest fixtures.

Tests use tiny synthetic media generated on the fly rather than the real (git-ignored,
large) course videos, so the suite is hermetic and fast.
"""

import shutil
import subprocess

import pytest


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


@pytest.fixture(scope="session")
def sample_clip(tmp_path_factory):
    """A 2 s, 10 fps clip: 1 s red then 1 s blue (one obvious scene cut at t=1 s)."""
    if not ffmpeg_available():
        pytest.skip("ffmpeg not installed")
    out = tmp_path_factory.mktemp("video") / "clip.mp4"
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "color=c=red:s=128x96:d=1",
                "-f",
                "lavfi",
                "-i",
                "color=c=blue:s=128x96:d=1",
                "-filter_complex",
                "[0][1]concat=n=2:v=1[v]",
                "-map",
                "[v]",
                "-r",
                "10",
                "-pix_fmt",
                "yuv420p",
                str(out),
            ],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode(errors="replace") if exc.stderr else ""
        pytest.skip(f"ffmpeg could not generate the test clip: {stderr}")
    return out
