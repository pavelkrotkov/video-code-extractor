import json
from pathlib import Path

import pytest

from vce.merge import (
    build_provenance,
    merge_snippets,
    write_provenance,
)
from vce.types import Extraction, Frame


def make_extraction(text, *, ms, confidence=0.9, name=None, backend="fake"):
    """Build an Extraction over a synthetic Frame; the frame path is derived from ``ms``."""
    frame = Frame(path=Path(f"/frames/{name or ms}.png"), timestamp_ms=ms)
    return Extraction(frame=frame, text=text, confidence=confidence, backend=backend)


# --- clustering + representative selection ------------------------------------------------


def test_near_identical_merge_into_one_citing_both_sources():
    # Same block, one with a classic OCR typo (l -> 1). They should collapse to a single snippet
    # whose sources cite both frames.
    a = make_extraction("def foo():\n    return value", ms=0, confidence=0.95)
    b = make_extraction("def foo():\n    return va1ue", ms=1000, confidence=0.80)
    merged = merge_snippets([a, b])

    assert len(merged) == 1
    snippet = merged[0]
    assert {f.timestamp_ms for f in snippet.sources} == {0, 1000}
    # The higher-confidence transcription wins as the clean representative.
    assert snippet.code == "def foo():\n    return value"
    assert snippet.notes == ""


def test_distinct_snippets_preserved_separately():
    a = make_extraction("import os\nprint(os.getcwd())", ms=0)
    b = make_extraction("class Bar:\n    pass", ms=1000)
    merged = merge_snippets([a, b])

    assert len(merged) == 2
    assert {m.code for m in merged} == {a.text, b.text}


def test_representative_is_highest_confidence():
    low = make_extraction("result = compute(x, y)", ms=0, confidence=0.40)
    high = make_extraction("result = compute(x, v)", ms=1000, confidence=0.92)
    merged = merge_snippets([low, high])

    assert len(merged) == 1
    assert merged[0].code == "result = compute(x, v)"  # from the high-confidence extraction


def test_representative_path_tiebreak_prefers_lexicographically_smaller_prefix():
    # Equal confidence and timestamp; the only tie-break left is the path. When one path is a
    # prefix of the other ("/f/a" vs "/f/a/b"), the lexicographically smaller "/f/a" must win.
    # (A negated-codepoint tuple under max would wrongly pick the longer path here.)
    short = Extraction(
        frame=Frame(path=Path("/f/a"), timestamp_ms=0), text="value = aaaa", confidence=0.9
    )
    long = Extraction(
        frame=Frame(path=Path("/f/a/b"), timestamp_ms=0), text="value = aaab", confidence=0.9
    )
    merged = merge_snippets([long, short])  # input order shouldn't matter

    assert len(merged) == 1
    assert merged[0].code == "value = aaaa"  # from the shorter, lexicographically smaller path


def test_empty_returns_empty():
    assert merge_snippets([]) == []


def test_output_ordered_by_earliest_source_timestamp():
    late = make_extraction("late = True", ms=5000)
    early = make_extraction("early = True", ms=10)
    merged = merge_snippets([late, early])

    assert [m.sources[0].timestamp_ms for m in merged] == [10, 5000]


def test_whitespace_and_cursor_noise_do_not_split_a_group():
    # Trailing spaces / a stray trailing blank line are normalized away before comparison.
    a = make_extraction("a = 1\nb = 2", ms=0, confidence=0.7)
    b = make_extraction("a = 1   \nb = 2\n", ms=1000, confidence=0.9)
    merged = merge_snippets([a, b])

    assert len(merged) == 1
    assert {f.timestamp_ms for f in merged[0].sources} == {0, 1000}


# --- conflict / low-confidence flagging ---------------------------------------------------


def test_conflicting_pair_is_flagged_in_notes():
    # Two differing transcriptions of similar shape with near-equal confidence: no clear winner.
    a = make_extraction("total = a + b", ms=0, confidence=0.88)
    b = make_extraction("total = a - b", ms=1000, confidence=0.86)
    merged = merge_snippets([a, b])

    assert len(merged) == 1
    assert "conflict" in merged[0].notes


def test_clear_confidence_winner_is_not_flagged_as_conflict():
    a = make_extraction("total = a + b", ms=0, confidence=0.30)
    b = make_extraction("total = a - b", ms=1000, confidence=0.95)
    merged = merge_snippets([a, b], low_confidence_threshold=0.0)

    assert "conflict" not in merged[0].notes
    assert merged[0].code == "total = a - b"


def test_low_confidence_representative_is_flagged():
    a = make_extraction("y = 2", ms=0, confidence=0.20)
    merged = merge_snippets([a])

    assert "low confidence" in merged[0].notes


def test_similarity_threshold_controls_grouping():
    a = make_extraction("value = 1", ms=0)
    b = make_extraction("value = 2", ms=1000)
    # A strict threshold refuses to merge the one-character difference.
    strict = merge_snippets([a, b], similarity_threshold=0.99)
    assert len(strict) == 2


@pytest.mark.parametrize("bad", [-0.1, 1.5])
def test_out_of_range_thresholds_raise(bad):
    with pytest.raises(ValueError, match=r"within \[0, 1\]"):
        merge_snippets([], similarity_threshold=bad)


# --- injectable merge function ------------------------------------------------------------


def test_merge_fn_is_injected_and_used():
    a = make_extraction("def foo():\n    return value", ms=0, confidence=0.80)
    b = make_extraction("def foo():\n    return va1ue", ms=1000, confidence=0.95)
    calls = []

    def fake_llm_merge(cluster):
        calls.append(tuple(e.frame.timestamp_ms for e in cluster))
        return "def foo():\n    return value  # reconciled"

    merged = merge_snippets([a, b], merge_fn=fake_llm_merge)

    assert merged[0].code == "def foo():\n    return value  # reconciled"
    assert calls == [(0, 1000)]  # called once, with the whole cluster


# --- provenance ---------------------------------------------------------------------------


def test_provenance_has_entry_per_source_extraction():
    a = make_extraction("def foo():\n    return value", ms=0, confidence=0.95)
    b = make_extraction("def foo():\n    return va1ue", ms=1000, confidence=0.80)
    c = make_extraction("class Bar:\n    pass", ms=2000)
    extractions = [a, b, c]

    merged = merge_snippets(extractions)
    provenance = build_provenance(extractions, merged)

    assert len(provenance) == len(extractions)
    assert set(provenance[0]) == {"timestamp", "screenshot", "raw_ocr", "cleaned_code"}
    # Each near-identical source points at the same cleaned representative code.
    by_ts = {e["timestamp"]: e for e in provenance}
    assert by_ts[0]["raw_ocr"] == "def foo():\n    return value"
    assert by_ts[0]["cleaned_code"] == "def foo():\n    return value"
    assert by_ts[1000]["raw_ocr"] == "def foo():\n    return va1ue"
    assert by_ts[1000]["cleaned_code"] == "def foo():\n    return value"
    assert by_ts[2000]["cleaned_code"] == "class Bar:\n    pass"


def test_provenance_attributes_per_extraction_when_one_frame_feeds_two_clusters():
    # The same frame is run through two backends, producing two very different transcriptions that
    # land in separate clusters. Each extraction's raw_ocr must map to *its own* cleaned code, not
    # whichever snippet happened to be written last for that frame.
    frame = Frame(path=Path("/frames/hard.png"), timestamp_ms=0)
    paddle = Extraction(
        frame=frame, text="def foo():\n    return 1", confidence=0.9, backend="paddle"
    )
    vision = Extraction(frame=frame, text="import numpy as np", confidence=0.9, backend="vision")
    extractions = [paddle, vision]

    merged = merge_snippets(extractions)
    assert len(merged) == 2  # distinct transcriptions -> two snippets, both citing the same frame

    provenance = build_provenance(extractions, merged)
    by_raw = {e["raw_ocr"]: e["cleaned_code"] for e in provenance}
    assert by_raw["def foo():\n    return 1"] == "def foo():\n    return 1"
    assert by_raw["import numpy as np"] == "import numpy as np"


def test_provenance_is_ordered_by_timestamp():
    a = make_extraction("a = 1", ms=3000)
    b = make_extraction("b = 2", ms=10)
    provenance = build_provenance([a, b], merge_snippets([a, b]))
    assert [e["timestamp"] for e in provenance] == [10, 3000]


def test_write_provenance_round_trips(tmp_path):
    a = make_extraction("a = 1", ms=0)
    entries = build_provenance([a], merge_snippets([a]))
    out = tmp_path / "out.provenance.json"

    write_provenance(out, entries)

    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded == entries
    assert out.read_text(encoding="utf-8").endswith("\n")
