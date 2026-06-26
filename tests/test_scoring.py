from pathlib import Path

import pytest

from vce.scoring import score_code_likeness
from vce.types import Candidate, Frame

FRAME = Frame(path=Path("f.jpg"), timestamp_ms=0)

PYTHON = """\
import numpy as np


def loss_fn(params, batch):
    preds = model(params, batch)
    return jnp.mean((preds - batch) ** 2)
"""

JS = """\
const sum = (a, b) => {
  return a + b;
};
"""

PROSE = (
    "If you return to the class later, you can import these ideas and build "
    "for the future without worrying about the details for now."
)


def _score(text: str) -> float:
    cand = score_code_likeness(FRAME, text)
    assert isinstance(cand, Candidate)
    assert cand.frame is FRAME
    return cand.score


@pytest.mark.parametrize("text", [PYTHON, JS])
def test_code_scores_high(text):
    assert _score(text) > 0.6


def test_prose_scores_low():
    assert _score(PROSE) < 0.2


def test_empty_scores_zero():
    assert _score("") == 0.0
    assert _score("   \n\t ") == 0.0


def test_single_import_is_mid_high():
    assert _score("import numpy as np") > 0.3


def test_shell_install_is_mid_high():
    assert _score("pip install requests") > 0.3


def test_brace_style_class_header_scores_high():
    assert _score("class Foo {\n    int x = 0;\n}") > 0.6


def test_sql_is_recognized_as_code():
    assert _score("SELECT name FROM users WHERE active = true") > 0.3


def test_html_is_recognized_as_code():
    assert _score('<div class="card">hello</div>') > 0.3


def test_upper_case_constant_counts_as_identifier():
    # MAX_RETRIES alone is a weak-but-nonzero code signal, clearly above empty.
    assert _score("MAX_RETRIES") > 0.0


def test_prose_with_sql_words_stays_low():
    # "from" and "where" as English words must not trip the SQL signal.
    prose = "Where do you come from, and where are you going from here today?"
    assert _score(prose) < 0.2


def test_prose_select_from_sentence_stays_low():
    # a sentence with both "select" and "from" must not match the SQL signal
    assert _score("Please select one of the options from the dropdown menu.") < 0.2


def test_prose_pluralization_stays_low():
    # "individual(s)" / "word(s)" must not trip the function-call signal
    assert _score("Ask the individual(s) and word(s) you trust about the option(s).") < 0.2


def test_semicolon_with_trailing_comment_counts():
    assert _score("x = 1;  // initialize counter") > 0.3


@pytest.mark.parametrize("text", [PYTHON, JS, PROSE, "", "x", "pip install x"])
def test_scores_are_bounded(text):
    assert 0.0 <= _score(text) <= 1.0
