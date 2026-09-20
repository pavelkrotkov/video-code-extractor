from pathlib import Path

from vce.codequality import clean_transcription, is_suspect, parses_as_python, reconcile_cluster
from vce.types import Extraction, Frame


def _ext(text: str, confidence: float, ms: int) -> Extraction:
    return Extraction(Frame(Path(f"/f/{ms}.png"), ms), text, confidence)


def test_python_validity_catches_structural_ocr_errors():
    assert parses_as_python("def f():\n    return 1")
    assert not parses_as_python("x = 1\nreturn x")
    assert is_suspect("y = jnp.ones((3, 3)")
    assert is_suspect("    return foo()")
    assert is_suspect("break")
    assert not is_suspect("the quick brown fox")


def test_clean_transcription_removes_notebook_output():
    raw = "In [1]: import numpy as np\nx = compute()\nOut[1]:\narray([0., 0., 0., 0.])"
    assert clean_transcription(raw) == "import numpy as np\nx = compute()"
    assert is_suspect(raw)
    continued = "In [1]: def f():\n   ...:     return 1"
    assert clean_transcription(continued) == "def f():\n    return 1"
    source = "Out[1] = compute()\nprint(Out[1])"
    assert clean_transcription(source) == source
    assert not is_suspect(source)


def test_clean_transcription_preserves_python_literal_rows():
    raw = "data = [\n    [1, 2, 3, 4],\n    [5, 6, 7, 8],\n]\nbroken("
    cleaned = clean_transcription(raw)
    assert "[1, 2, 3, 4]," in cleaned
    assert "[5, 6, 7, 8]," in cleaned
    source = "x = (\n    array([1., 2., 3., 4.])\n)"
    assert clean_transcription(source) == source


def test_out_prompt_drops_nonnumeric_repr():
    raw = "model = Net()\nOut[7]:\n<Net object at 0x10f>"
    assert clean_transcription(raw) == "model = Net()"


def test_reconcile_prefers_complete_valid_variant():
    broken = _ext("def f():\n    return [", 0.99, 0)
    good = _ext("def f():\n    return [1, 2]", 0.80, 1000)
    assert reconcile_cluster([broken, good]) == good.text
    valid = _ext("raise ValueError()", 0.70, 2000)
    malformed = _ext("raise ValueError(", 0.99, 3000)
    assert reconcile_cluster([malformed, valid]) == valid.text


def test_reconcile_prefers_confidence_for_non_python():
    clean = _ext("const x = 5;", 0.95, 0)
    noisy = _ext("const x = 5;\n|", 0.80, 1000)
    assert reconcile_cluster([noisy, clean]) == clean.text
    js = _ext("const descriptiveVariable = computeSomething(argumentOne, argumentTwo);", 0.95, 2000)
    pythonish = _ext("descriptiveVariable = computeSomething(argumentOne, argumentTwo);", 0.80, 3000)
    assert reconcile_cluster([pythonish, js]) == js.text
