import shutil
from pathlib import Path

import pytest

from vce.frames import (
    FFmpegNotFoundError,
    FrameExtractionError,
    _parse_pts_ms,
    extract_frames,
    scene_change_frames,
)

requires_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def test_parse_pts_ms_reads_showinfo_lines_only():
    stderr = (
        "[Parsed_showinfo_1 @ 0x1] n:   0 pts:     0 pts_time:0\n"
        "[Parsed_showinfo_1 @ 0x1] n:   1 pts:  1000 pts_time:1.5\n"
        "Input #0, from 'weird_pts_time:9.mp4': metadata\n"  # not a showinfo frame line
    )
    assert _parse_pts_ms(stderr) == [0, 1500]


def test_parse_pts_ms_clamps_negative():
    stderr = "[Parsed_showinfo_0 @ 0x1] n:   0 pts:-2 pts_time:-0.080000\n"
    assert _parse_pts_ms(stderr) == [0]


def test_extract_frames_rejects_nonpositive_fps(tmp_path):
    with pytest.raises(ValueError):
        extract_frames(Path("missing.mp4"), tmp_path, fps=0)


def test_extract_frames_raises_without_ffmpeg(monkeypatch, tmp_path):
    video = tmp_path / "exists.mp4"
    video.write_bytes(b"x")  # must exist so validation passes and we reach the ffmpeg check
    monkeypatch.setattr("vce.frames.shutil.which", lambda _: None)
    with pytest.raises(FFmpegNotFoundError):
        extract_frames(video, tmp_path, fps=1.0)


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
