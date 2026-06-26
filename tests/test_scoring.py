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


@pytest.mark.parametrize("text", [PYTHON, JS, PROSE, "", "x", "pip install x"])
def test_scores_are_bounded(text):
    assert 0.0 <= _score(text) <= 1.0
