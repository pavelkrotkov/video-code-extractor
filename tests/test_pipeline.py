"""Tests for the end-to-end orchestration.

The heavy, disk/network-bound stages are stubbed: frame extraction is monkeypatched to return
synthetic frames backed by tiny on-disk PNGs (so the real perceptual-hash dedup still runs), and
the extraction backends are injected fakes. That keeps these tests hermetic and ffmpeg-free while
still exercising the actual ordering, the code-likeness gate, the two-tier escalation, and the
script + provenance writers.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from vce import pipeline as pipeline_mod
from vce.pipeline import Pipeline, PipelineConfig, _build_script
from vce.types import BBox, Extraction, Frame, MergedSnippet

CODE = "def foo():\n    return 1"
PROSE = "the quick brown fox jumps over the lazy dog"


def _write_noise_png(path: Path, seed: int) -> None:
    """Write a 32x32 random-noise PNG; distinct seeds give perceptually distinct images."""
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path)


class FakeBackend:
    """An injectable :class:`~vce.backends.base.ExtractionBackend` driven by a per-frame function."""

    def __init__(self, name, fn):
        self.name = name
        self._fn = fn
        self.calls: list[Frame] = []

    def extract(self, image_path: Path, frame: Frame) -> Extraction:
        self.calls.append(frame)
        text, confidence = self._fn(frame)
        return Extraction(frame=frame, text=text, confidence=confidence, backend=self.name)


@pytest.fixture
def synthetic_frames(tmp_path, monkeypatch):
    """Make ``extract_frames`` yield three distinct on-disk frames; scene detection yields none."""
    frames_dir = tmp_path / "src_frames"
    frames_dir.mkdir()
    frames = []
    for i in range(3):
        path = frames_dir / f"frame_{i:06d}.jpg"
        _write_noise_png(path, seed=i + 1)
        frames.append(Frame(path=path, timestamp_ms=i * 1000))

    monkeypatch.setattr(pipeline_mod, "extract_frames", lambda *a, **k: list(frames))
    monkeypatch.setattr(pipeline_mod, "scene_change_frames", lambda *a, **k: [])
    return frames


def _config(tmp_path, **overrides):
    return PipelineConfig(out_dir=tmp_path / "out", **overrides)


# --- end-to-end ---------------------------------------------------------------------------


def test_run_writes_script_and_provenance(tmp_path, synthetic_frames):
    primary = FakeBackend("fake", lambda f: (CODE, 0.95))
    pipeline = Pipeline(primary, _config(tmp_path))

    result = pipeline.run(Path("lesson.mp4"))

    assert result.script_path.name == "lesson.py"
    assert result.provenance_path.name == "lesson.provenance.json"
    assert result.script_path.read_text() == CODE + "\n"  # one merged snippet, trailing newline
    assert result.frames_total == 3
    assert result.frames_kept == 3
    assert result.num_snippets == 1

    provenance = json.loads(result.provenance_path.read_text())
    assert {e["timestamp"] for e in provenance} == {0, 1000, 2000}
    assert all(e["cleaned_code"] == CODE for e in provenance)


def test_empty_video_stem_falls_back_to_extracted(tmp_path, synthetic_frames):
    # A path whose stem is empty (e.g. ".") must not produce ".py"/".provenance.json".
    pipeline = Pipeline(FakeBackend("fake", lambda f: (CODE, 0.95)), _config(tmp_path))
    result = pipeline.run(Path("."))
    assert result.script_path.name == "extracted.py"
    assert result.provenance_path.name == "extracted.provenance.json"


def test_gate_drops_non_code_frames(tmp_path, synthetic_frames):
    # The middle frame transcribes to prose and must be dropped before merge/provenance.
    def fn(frame):
        return (PROSE, 0.95) if frame.timestamp_ms == 1000 else (CODE, 0.95)

    pipeline = Pipeline(FakeBackend("fake", fn), _config(tmp_path))
    result = pipeline.run(Path("lesson.mp4"))

    assert result.frames_kept == 2
    provenance = json.loads(result.provenance_path.read_text())
    assert {e["timestamp"] for e in provenance} == {0, 2000}


def test_escalation_only_for_low_confidence_kept_frames(tmp_path, synthetic_frames):
    # Primary reads frame@1000 with low confidence -> escalate; the others stay on primary.
    def primary_fn(frame):
        return (CODE, 0.4) if frame.timestamp_ms == 1000 else (CODE, 0.95)

    primary = FakeBackend("primary", primary_fn)
    escalation = FakeBackend("vision", lambda f: ("def foo():\n    return 2", 0.99))
    pipeline = Pipeline(primary, _config(tmp_path, escalate_below=0.6), escalation=escalation)

    pipeline.run(Path("lesson.mp4"))

    # Escalation ran for exactly the one low-confidence frame.
    assert [f.timestamp_ms for f in escalation.calls] == [1000]
    # Primary still ran on every kept frame (it gates first).
    assert {f.timestamp_ms for f in primary.calls} == {0, 1000, 2000}


def test_escalation_kept_only_when_it_reads_as_code(tmp_path, synthetic_frames):
    # All three frames are low-confidence on the primary, so escalation runs for each. The vision
    # backend returns usable code only for frame@1000; for the others it returns prose. The prose
    # results must be discarded in favor of the gate-passing primary text, not overwrite it.
    def primary_fn(frame):
        return (CODE, 0.3)  # code-like (passes the gate) but low confidence (triggers escalation)

    def vision_fn(frame):
        return ("def bar():\n    return 9", 0.99) if frame.timestamp_ms == 1000 else (PROSE, 0.99)

    primary = FakeBackend("primary", primary_fn)
    escalation = FakeBackend("vision", vision_fn)
    pipeline = Pipeline(primary, _config(tmp_path, escalate_below=0.6), escalation=escalation)
    result = pipeline.run(Path("lesson.mp4"))

    # Nothing is dropped: every frame survives via either the vision code or the primary fallback.
    assert result.frames_kept == 3
    provenance = json.loads(result.provenance_path.read_text())
    by_ts = {e["timestamp"]: e["cleaned_code"] for e in provenance}
    assert by_ts[1000] == "def bar():\n    return 9"  # adopted the code-like vision result
    assert by_ts[0] == CODE and by_ts[2000] == CODE  # kept the primary where vision wasn't code


def test_no_escalation_when_backend_absent(tmp_path, synthetic_frames):
    primary = FakeBackend("primary", lambda f: (CODE, 0.1))  # below any threshold
    pipeline = Pipeline(primary, _config(tmp_path, escalate_below=0.6))  # no escalation wired

    result = pipeline.run(Path("lesson.mp4"))
    assert result.frames_kept == 3  # single-tier, nothing dropped by escalation


def test_crop_is_applied_before_extraction(tmp_path, synthetic_frames):
    seen: list[Path] = []

    def fn(frame):
        return (CODE, 0.95)

    backend = FakeBackend("fake", fn)
    # Wrap extract to capture the image path the backend was handed.
    original = backend.extract

    def spy(image_path, frame):
        seen.append(image_path)
        return original(image_path, frame)

    backend.extract = spy  # type: ignore[method-assign]

    pipeline = Pipeline(backend, _config(tmp_path, crop=BBox(0, 0, 16, 16)))
    pipeline.run(Path("lesson.mp4"))

    # Each backend call received a freshly written crop under the per-video crops dir, not the
    # raw frame. The dir is namespaced by the video stem so a run can't clobber unrelated files.
    assert seen and all(p.parent.name == "lesson_crops" for p in seen)
    assert all(p.exists() for p in seen)


def test_intermediate_dirs_are_namespaced_per_video(tmp_path, monkeypatch):
    # The frame stages clean their target dir, so it must be per-video (not a generic "frames/")
    # to avoid deleting a user's unrelated images when --out points at an existing directory.
    captured = {}

    def fake_extract(video, out_dir, **kwargs):
        captured["frames_dir"] = out_dir
        return []

    monkeypatch.setattr(pipeline_mod, "extract_frames", fake_extract)
    monkeypatch.setattr(pipeline_mod, "scene_change_frames", lambda *a, **k: [])

    Pipeline(FakeBackend("fake", lambda f: (CODE, 0.95)), _config(tmp_path)).run(
        Path("/videos/lesson.mp4")
    )
    assert captured["frames_dir"] == tmp_path / "out" / "lesson_frames"


# --- units --------------------------------------------------------------------------------


def test_build_script_joins_and_skips_empty():
    snippets = [
        MergedSnippet(code="import os\n"),
        MergedSnippet(code="   "),  # whitespace-only: skipped
        MergedSnippet(code="print(os.getcwd())"),
    ]
    assert _build_script(snippets) == "import os\n\n\nprint(os.getcwd())\n"


def test_build_script_empty_is_empty():
    assert _build_script([]) == ""
    assert _build_script([MergedSnippet(code="\n  \n")]) == ""


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fps": 0},
        {"score_threshold": 1.5},
        {"escalate_below": -0.1},
        {"scene_threshold": 0.0},
        {"scene_threshold": 1.5},
        {"dedup_max_distance": -1},
        {"similarity_threshold": 1.1},
        {"low_confidence_threshold": -0.5},
        {"conflict_margin": 2.0},
    ],
)
def test_config_rejects_bad_values(tmp_path, kwargs):
    with pytest.raises(ValueError):
        PipelineConfig(out_dir=tmp_path, **kwargs)
