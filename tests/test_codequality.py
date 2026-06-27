"""Tests for the language-aware code-quality signals (validity, suspicion, cleaning, reconcile)."""

from pathlib import Path

from vce.codequality import (
    clean_transcription,
    contains_notebook_chrome,
    detect_language,
    is_suspect,
    parses_as_python,
    reconcile_cluster,
)
from vce.types import Extraction, Frame


def _ext(text, *, ms=0, confidence=0.9):
    return Extraction(
        frame=Frame(path=Path(f"/f/{ms}.png"), timestamp_ms=ms), text=text, confidence=confidence
    )


# --- language detection -------------------------------------------------------------------


def test_detect_language_recognizes_python_markers():
    assert detect_language("import os") == "python"
    assert detect_language("from a.b import c") == "python"
    assert detect_language("def f(x):\n    return x") == "python"
    assert detect_language("class Foo:\n    pass") == "python"
    assert detect_language("for i in range(3):\n    pass") == "python"


def test_detect_language_recognizes_keyword_less_statements():
    # Assignments and calls carry no def/import/class keyword but are still Python; without this a
    # broken `y = jnp.ones((3, 3)` would never be validated or flagged.
    assert detect_language("y = jnp.ones((3, 3))") == "python"
    assert detect_language("self.data = []") == "python"
    assert detect_language("model.fit(x, y)") == "python"


def test_detect_language_returns_none_for_prose():
    assert detect_language("the quick brown fox") is None
    assert detect_language("Select one of the options from the menu") is None


def test_keyword_less_invalid_python_is_suspect_and_flagged():
    # The gemini gap: a keyword-less assignment that does not parse must be caught.
    assert is_suspect("y = jnp.ones((3, 3)")  # missing closing paren
    assert not is_suspect("y = jnp.ones((3, 3))")  # valid -> not suspect


def test_pandas_constructors_are_not_treated_as_rendered_output():
    # DataFrame(...) / Series(...) are source, not printed output, and must survive cleaning.
    assert not contains_notebook_chrome("DataFrame([1, 2, 3, 4, 5])")
    assert (
        clean_transcription("df = DataFrame([1, 2, 3, 4, 5])") == "df = DataFrame([1, 2, 3, 4, 5])"
    )


# --- python validity ----------------------------------------------------------------------


def test_parses_as_python_true_for_valid_code():
    assert parses_as_python("def f():\n    return 1")
    assert parses_as_python("\n\nx = [1, 2, 3]\n")  # leading/trailing blanks tolerated


def test_parses_as_python_false_for_broken_or_empty():
    assert not parses_as_python("def __init__(self, max_len):\n    self.data = [")  # unclosed
    assert not parses_as_python("def f(:\n    return")  # mangled signature
    assert not parses_as_python("   ")


def test_parses_as_python_rejects_module_level_statements():
    # codex: compile() (not ast.parse) catches statements that parse but are invalid at module
    # level — the fingerprint of OCR dropping an enclosing block / its indentation.
    assert not parses_as_python("return 1")
    assert not parses_as_python("break")
    assert parses_as_python("def f():\n    return 1")  # valid in context


def test_is_suspect_flags_dedented_return():
    # A leaked top-level return in an otherwise Python-detected snippet is now flagged.
    assert is_suspect("x = 1\nreturn x")


def test_bare_array_repr_without_prompt_is_treated_as_output():
    # codex: a rendered repr captured without its Out[n]: prompt still parses, but it is output,
    # not source — so it is both suspect and stripped from the cleaned code.
    raw = "x = compute()\narray([0., 0., 0., 0., 0.])"
    assert is_suspect(raw)
    assert clean_transcription(raw) == "x = compute()"


# --- notebook chrome / rendered output ----------------------------------------------------


def test_contains_notebook_chrome_detects_prompts_and_arrays():
    assert contains_notebook_chrome("In [12]:\nimport numpy")
    assert contains_notebook_chrome("x = 1\nOut[3]:")
    assert contains_notebook_chrome("array([0.1, 0.2, 0.3, 0.4, 0.5])")


def test_contains_notebook_chrome_false_for_plain_code():
    assert not contains_notebook_chrome("def f():\n    return (1, 2)")
    assert not contains_notebook_chrome("shape = (1, 28, 28)")  # short literal, not output


def test_clean_transcription_strips_chrome_keeps_code():
    raw = "In [1]:\nimport numpy as np\narr = np.zeros(3)\nOut[1]:\narray([0., 0., 0.])"
    assert clean_transcription(raw) == "import numpy as np\narr = np.zeros(3)"


def test_clean_transcription_leaves_valid_code_untouched():
    code = "def foo():\n    return 1"
    assert clean_transcription(code) == code


def test_clean_preserves_multiline_numeric_literal_in_valid_code():
    # P1 (codex): rows of a real multi-line literal look like rendered arrays but must NOT be
    # stripped. Valid source is returned untouched.
    code = "data = [\n    [1, 2, 3, 4],\n    [5, 6, 7, 8],\n]"
    assert clean_transcription(code) == code


def test_clean_strips_inline_notebook_prompts():
    # codex: a prompt sharing a visual line with its payload must still be stripped.
    assert clean_transcription("In [1]: import numpy as np") == "import numpy as np"
    assert clean_transcription("Out[1]: array([1, 2, 3])") == ""


def test_clean_strips_nonnumeric_output_after_out_prompt():
    # codex: an Out[n]: prompt opens an output region; the following repr is dropped even when it is
    # not numeric (e.g. a <... object at 0x...> repr).
    raw = "model = Net()\nOut[7]:\n<Net object at 0x10f>"
    assert clean_transcription(raw) == "model = Net()"


def test_clean_preserves_comma_literal_rows_in_unparseable_snippet():
    # gemini round 2: even when the whole snippet fails to parse, comma-separated Python list rows
    # are valid Python and must survive the fallback cleaning (no silent data loss).
    raw = "weights = [\n    [1, 2, 3, 4, 5, 6],\n    [7, 8, 9, 10, 11, 12],\n    broken("
    cleaned = clean_transcription(raw)
    assert "[1, 2, 3, 4, 5, 6]," in cleaned
    assert "[7, 8, 9, 10, 11, 12]," in cleaned


def test_clean_strips_space_separated_numpy_repr_in_unparseable_snippet():
    # The complement: a printed numpy repr uses space separators, fails to parse, and is stripped.
    raw = "x = process(\n[0.1 0.2 0.3 0.4 0.5 0.6]\nbroken("
    assert "0.1 0.2" not in clean_transcription(raw)


# --- suspicion ----------------------------------------------------------------------------


def test_is_suspect_true_for_high_confidence_but_invalid_python():
    # The crux of issue #24: code-like, would-be high confidence, but does not parse.
    assert is_suspect("def __init__(self, max_len):\n    self.data = [")


def test_is_suspect_true_when_notebook_output_present():
    assert is_suspect("model = Net()\nOut[7]:\n<Net object at 0x10f>")


def test_is_suspect_false_for_valid_code_and_for_prose():
    assert not is_suspect("def foo():\n    return 1")
    assert not is_suspect("the quick brown fox")  # not code -> handled by the gate, not here
    assert not is_suspect("")


# --- reconciliation -----------------------------------------------------------------------


def test_reconcile_cluster_prefers_complete_valid_variant():
    # Two captures of one cell: a truncated/broken one and the full valid one. The full valid
    # variant wins even though the broken one is shorter (fewer non-blank lines) -- and it is
    # returned cleaned.
    broken = _ext("def f():\n    return [", ms=0, confidence=0.99)
    good = _ext("def f():\n    return [1, 2]", ms=1000, confidence=0.80)
    assert reconcile_cluster([broken, good]) == "def f():\n    return [1, 2]"


def test_reconcile_cluster_strips_chrome_from_winner():
    a = _ext("In [2]:\nx = compute()\nOut[2]:\narray([1, 2, 3, 4, 5])", ms=0)
    assert reconcile_cluster([a]) == "x = compute()"


def test_reconcile_prefers_confidence_over_length_for_non_python():
    # codex (round 1): for non-Python clusters (valid==0 for all), a higher-confidence clean read
    # must beat a lower-confidence variant that merely has an extra noise line.
    clean = _ext("const x = 5;", ms=0, confidence=0.95)
    noisy = _ext("const x = 5;\n|", ms=1000, confidence=0.80)  # extra cursor line, longer
    assert reconcile_cluster([noisy, clean]) == "const x = 5;"


def test_reconcile_prefers_completeness_among_valid_variants():
    # codex (round 2): among variants that all compile, the most complete one wins even at lower
    # confidence, so a fuller cell is not dropped for a shorter higher-confidence capture.
    full = _ext("import os\nimport sys\nx = 1", ms=0, confidence=0.80)
    short = _ext("import os\nx = 1", ms=1000, confidence=0.95)
    assert reconcile_cluster([short, full]) == "import os\nimport sys\nx = 1"
