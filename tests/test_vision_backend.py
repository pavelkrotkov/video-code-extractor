import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from vce.backends.base import ExtractionBackend
from vce.backends.vision import (
    OCR_SYSTEM_PROMPT,
    VisionLLMBackend,
    _confidence,
    _strip_fence,
)
from vce.types import Frame

FRAME = Frame(path=Path("f.jpg"), timestamp_ms=0)


class FakeChatClient:
    """Mimics the OpenAI client surface: client.chat.completions.create(...).choices[0].message."""

    def __init__(self, content, finish_reason="stop"):
        self.captured = {}

        def create(**kwargs):
            self.captured = kwargs
            message = SimpleNamespace(content=content)
            choice = SimpleNamespace(message=message, finish_reason=finish_reason)
            return SimpleNamespace(choices=[choice])

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))


@pytest.fixture
def png(tmp_path):
    from PIL import Image

    path = tmp_path / "crop.png"
    Image.new("RGB", (10, 10), "white").save(path)
    return path


def test_satisfies_backend_protocol():
    assert isinstance(VisionLLMBackend(), ExtractionBackend)
    assert VisionLLMBackend().name == "vision-gpt4v"


def test_prompt_forbids_inference():
    lowered = OCR_SYSTEM_PROMPT.lower()
    assert "do not infer" in lowered
    assert "ocr" in lowered
    assert "preserve indentation" in lowered


@pytest.mark.parametrize(
    "content,expected",
    [
        ("```python\nimport jax\n```", "import jax"),
        ("```\nx = 1\ny = 2\n```", "x = 1\ny = 2"),
        ("import os", "import os"),
        ("Here is the code:\n```js\nconst a = 1;\n```", "const a = 1;"),
        # truncated: opening fence but no closing fence (e.g. hit token limit)
        ("```python\nimport jax\nx = 1", "import jax\nx = 1"),
    ],
)
def test_strip_fence(content, expected):
    assert _strip_fence(content) == expected


def test_confidence_penalizes_ambiguous_markers():
    assert _confidence("clean code") == pytest.approx(0.9)
    assert _confidence("a[?]b[?]") == pytest.approx(0.7)
    assert _confidence("[?]" * 20) == pytest.approx(0.1)  # floored


def test_confidence_low_for_empty_text():
    assert _confidence("") == pytest.approx(0.1)
    assert _confidence("   \n ") == pytest.approx(0.1)


def test_strip_fence_drops_leading_newline_keeps_indentation():
    assert _strip_fence("```python\n\n    indented = 1\n```") == "    indented = 1"


def test_strip_fence_unfenced_preserves_indentation():
    # raw (unfenced) model output must keep the first line's indentation
    assert _strip_fence("\n    return x\n") == "    return x"


def test_extract_caps_confidence_on_truncated_completion(png):
    fake = FakeChatClient("```python\nimport jax\n```", finish_reason="length")
    ext = VisionLLMBackend(client=fake).extract(png, FRAME)
    assert ext.confidence <= 0.3


def test_extract_raises_on_empty_choices(png):
    class EmptyChoices:
        chat = SimpleNamespace(
            completions=SimpleNamespace(create=lambda **k: SimpleNamespace(choices=[]))
        )

    with pytest.raises(RuntimeError, match="no choices"):
        VisionLLMBackend(client=EmptyChoices()).extract(png, FRAME)


def test_build_messages_uses_guessed_mime_type(tmp_path):
    from PIL import Image

    jpg = tmp_path / "crop.jpg"
    Image.new("RGB", (10, 10), "white").save(jpg)
    fake = FakeChatClient("```\nok\n```")
    VisionLLMBackend(client=fake).extract(jpg, FRAME)
    url = fake.captured["messages"][1]["content"][1]["image_url"]["url"]
    assert url.startswith("data:image/jpeg;base64,")


def test_extract_parses_fenced_response(png):
    backend = VisionLLMBackend(client=FakeChatClient("```python\nimport jax\nx = 1\n```"))
    ext = backend.extract(png, FRAME)
    assert ext.text == "import jax\nx = 1"
    assert ext.backend == "vision-gpt4v"
    assert ext.frame is FRAME


def test_extract_sends_image_and_prompt(png):
    fake = FakeChatClient("```\nok\n```")
    VisionLLMBackend(client=fake, model="gpt-4o").extract(png, FRAME)
    assert fake.captured["model"] == "gpt-4o"
    assert fake.captured["temperature"] == 0
    messages = fake.captured["messages"]
    assert messages[0]["content"] == OCR_SYSTEM_PROMPT
    image_part = messages[1]["content"][1]
    assert image_part["type"] == "image_url"
    assert image_part["image_url"]["url"].startswith("data:image/png;base64,")


def test_extract_without_openai_installed_raises(monkeypatch, png):
    monkeypatch.setitem(sys.modules, "openai", None)
    with pytest.raises(ImportError, match="openai"):
        VisionLLMBackend().extract(png, FRAME)
