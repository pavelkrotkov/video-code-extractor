# video-code-extractor (`vce`)

Extract clean, **provenance-tracked** source code from programming-screencast videos
(lecture recordings where code flashes on screen occasionally).

Naïvely OCR-ing every frame produces garbage — duplicates, cursor artifacts, line-number
noise, autocomplete popups, and hallucinated code. `vce` instead runs a staged pipeline:
detect candidate frames → keep only code-bearing ones → crop the code region → OCR / vision
extract → merge overlapping snippets across time → emit a clean script **plus a sidecar
JSON** that records, for every line, the timestamp and screenshot it came from.

See [`docs/architecture.md`](docs/architecture.md) for the full design and prior-art survey.

## Pipeline

```mermaid
flowchart LR
    V[video] --> F[1. frame extraction<br/>fps + scene cuts]
    F --> D[2. dedup<br/>perceptual hash]
    D --> S[3. code-likeness<br/>scoring gate]
    S --> C[4. crop code region]
    C --> X[5. extract<br/>PaddleOCR / GPT-4V]
    X --> M[6. merge across frames<br/>+ provenance]
    M --> O[clean .py + .provenance.json]
```

## Status

Early scaffolding. Stages are tracked as GitHub issues under the project epic; the shared
types (`vce.types`) and the `ExtractionBackend` protocol (`vce.backends.base`) are in place.
The stages are now wired end-to-end behind the `vce extract` command.

## Usage

```bash
vce extract LESSON.mp4 --out build/        # -> build/LESSON.py + build/LESSON.provenance.json
```

Frames are sampled (`--fps`, plus scene cuts) → de-duplicated → gated for code-likeness
(`--score-threshold`) → optionally cropped (`--crop X,Y,W,H`) → transcribed → merged into a clean
script plus a provenance sidecar. The `--backend` flag picks the primary (cheap) backend; with
`paddleocr` selected, frames it reads with low confidence are escalated to the vision backend
(`--escalate-below`, needs `OPENAI_API_KEY`; disable with `--no-escalate`).

## Develop

```bash
uv sync --dev              # install deps + dev tools
uv run pytest -m "not paddle"
uv run ruff check .
uv run ruff format --check .
```

PaddleOCR is an optional, heavy extra: `uv sync --extra paddle` and run `pytest -m paddle`.

## Downloading the source course (separate tool)

[`tools/download_lessons.py`](tools/download_lessons.py) is the standalone script that
fetches the DeepLearning.AI JAX lessons used during development. Course videos are
git-ignored (`*.mp4`) and never committed.
