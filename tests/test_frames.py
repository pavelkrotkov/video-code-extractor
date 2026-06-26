import shutil
from pathlib import Path

import pytest

from vce.frames import (
    FFmpegNotFoundError,
    FrameExtractionError,
    _timestamp_for,
    extract_frames,
    scene_change_frames,
)

requires_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def test_timestamp_for():
    assert _timestamp_for(1, 1.0) == 0
    assert _timestamp_for(2, 1.0) == 1000
    assert _timestamp_for(3, 2.0) == 1000


def test_extract_frames_rejects_nonpositive_fps(tmp_path):
    with pytest.raises(ValueError):
        extract_frames(Path("missing.mp4"), tmp_path, fps=0)


def test_extract_frames_raises_without_ffmpeg(monkeypatch, tmp_path):
    monkeypatch.setattr("vce.frames.shutil.which", lambda _: None)
    with pytest.raises(FFmpegNotFoundError):
        extract_frames(Path("missing.mp4"), tmp_path, fps=1.0)


@requires_ffmpeg
def test_extract_frames_samples_timestamped_frames(sample_clip, tmp_path):
    frames = extract_frames(sample_clip, tmp_path, fps=1.0)
    assert len(frames) >= 2
    assert frames[0].timestamp_ms == 0
    assert frames[1].timestamp_ms == 1000
    assert all(f.path.exists() for f in frames)
    # frames are returned in chronological order
    assert [f.timestamp_ms for f in frames] == sorted(f.timestamp_ms for f in frames)


def test_extract_frames_missing_video_raises_filenotfound(tmp_path):
    with pytest.raises(FileNotFoundError):
        extract_frames(tmp_path / "nope.mp4", tmp_path, fps=1.0)


@requires_ffmpeg
def test_extract_frames_cleans_stale_output(sample_clip, tmp_path):
    stale = tmp_path / "frame_000999.jpg"
    stale.write_bytes(b"stale")
    frames = extract_frames(sample_clip, tmp_path, fps=1.0)
    # the leftover frame_*.jpg must not survive and pollute the chronological ordering
    assert stale not in {f.path for f in frames}
    assert not stale.exists()
    assert frames[0].timestamp_ms == 0


@requires_ffmpeg
def test_extract_frames_raises_on_invalid_video(tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"this is not a video")
    with pytest.raises(FrameExtractionError) as excinfo:
        extract_frames(bad, tmp_path / "out", fps=1.0)
    # ffmpeg's own stderr is surfaced in the message, not swallowed
    assert str(excinfo.value)


@requires_ffmpeg
def test_scene_change_frames_finds_the_cut(sample_clip, tmp_path):
    frames = scene_change_frames(sample_clip, tmp_path, threshold=0.3)
    assert len(frames) >= 1
    assert all(f.path.exists() for f in frames)
