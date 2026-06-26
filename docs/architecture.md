Yes. The best approach is absolutely to **detect candidate frames, crop the likely code region, OCR / vision-LLM them, then merge duplicates across time**. The important point: do **not** naïvely OCR every frame and paste the results together. That produces garbage, duplicates, partial snippets, cursor artifacts, line-number noise, autocomplete popups, and hallucinated code.

My practical recommendation:

**Use a two-stage pipeline: cheap OCR / CV to find candidate frames, then a vision LLM or strong OCR only on those frames, then a code-aware cleanup pass.** Existing tools can help, but I would not expect a polished, one-click “extract code from coding lesson videos” tool. The serious work is either research-grade or subtitle/OCR tooling that you have to adapt.

## Best answer in one sentence

For your use case — lessons where code flashes occasionally — I would build or use a small script that samples frames at 1–2 fps plus scene-change frames, deduplicates near-identical images, runs PaddleOCR or a vision LLM on candidate frames, scores the text for “code-likeness,” saves timestamped screenshots, and asks an LLM to merge the surviving snippets into a clean script with screenshot/time provenance.

That is the robust version of what you described.

---

## Existing tools / repos worth looking at

| Option                            |                                                                               What it does |             Good for your case? | My judgment                                                           |
| --------------------------------- | -----------------------------------------------------------------------------------------: | ------------------------------: | --------------------------------------------------------------------- |
| **PaddleOCR**                     | General OCR toolkit for images/PDFs, local, open source, strong text detection/recognition |                             Yes | Best local OCR engine to start with                                   |
| **PySceneDetect**                 |                                           Detects cuts / scene changes and can save images |                          Partly | Useful frame reducer, not enough by itself                            |
| **timminator/VideOCR**            |                  GUI/CLI for extracting burned-in subtitles using PaddleOCR or Google Lens |                           Maybe | Useful if code appears in a stable crop region, but subtitle-oriented |
| **knakamura13/videocr-PaddleOCR** |                                          Hardcoded subtitle extractor with crop parameters |                           Maybe | Good model for “OCR region of video,” not code-specific               |
| **CaptiOCR**                      |                         Real-time screen-region OCR with Tesseract and duplicate filtering |                           Maybe | Nice for interactive/manual capture while playing videos              |
| **PSC2CODE**                      |                        Research prototype for extracting code from programming screencasts |                Yes conceptually | Closest code-specific repo, but old/researchy                         |
| **Codemotion**                    |  Research prototype for extracting code and dynamic edits from programming tutorial videos |                Yes conceptually | Interesting but explicitly not well documented                        |
| **CodeSCAN**                      |     Newer dataset/benchmark for coding screencast analysis, especially VS Code screenshots | Useful for building classifiers | Not a turnkey extractor                                               |

### 1. PaddleOCR: best local OCR base

PaddleOCR is the strongest obvious local OCR base layer: the project describes itself as a lightweight OCR/document parsing toolkit and says it supports 100+ languages. For code screenshots, its detector is often more useful than raw Tesseract because you need bounding boxes, not just a text blob. ([GitHub][1])

Tesseract is still fine for simple high-resolution frames, but it is more fragile on low-resolution video, antialiased fonts, syntax highlighting, dark themes, and tiny punctuation. Tesseract’s own GitHub describes it as an OCR engine and command-line program, with Tesseract 4 adding an LSTM-based engine focused on line recognition. ([GitHub][2])

### 2. Vision LLMs: better accuracy, but use them carefully

There is direct research on this. A 2024 study compared Tesseract, Google Vision, GPT-4V, and Gemini on code extraction from programming tutorial frames at 1080p, 720p, 480p, and 360p. It found GPT-4V had the best normalized Levenshtein-distance results across qualities, while Tesseract degraded sharply at low resolution; for token-level scores, Tesseract averaged 0.31 at 360p while GPT-4V averaged 0.89 and Gemini 0.86. ([MDPI][3])

But the same study observed a real problem with vision LLMs: they may “autocomplete” code, producing syntactically plausible but not actually visible text, so the right prompt is not “write the code from this screenshot” but “act as OCR; extract only visible code; preserve uncertainty; do not infer missing lines.” ([MDPI][3])

That means: **LLMs are excellent for OCR repair and snippet merging, but dangerous as the only source of truth.** Keep the screenshots.

### 3. PySceneDetect: useful, but not sufficient

PySceneDetect is a Python/OpenCV scene cut and transition detection tool/library. It can save the first and last frame of detected scenes and export timecodes. ([GitHub][4])

Use it to reduce work, especially if the videos cut to slides/code blocks. But for your case — narrator on screen, code briefly flashing — pure scene detection can miss changes because the “scene” may not change enough. A code overlay appearing inside the same camera shot may be a small visual difference. So use PySceneDetect as one candidate source, not the detector.

FFmpeg also has scene-change filters; its documentation describes `scdet` as detecting video scene changes and emitting frame metadata such as `lavfi.scd.score` and `lavfi.scd.time`. ([FFmpeg][5])

### 4. Video OCR tools: good starting points, subtitle-biased

**timminator/VideOCR** extracts burned-in subtitles from videos using PaddleOCR locally or Google Lens in a hybrid mode, and it has both GUI and command-line usage. ([GitHub][6])

**knakamura13/videocr-PaddleOCR** is also subtitle-oriented, but its README has the exact trick you probably need: it supports crop parameters so OCR is run only on the relevant part of the video, which speeds up processing and improves accuracy. ([GitHub][7])

**CaptiOCR** is interesting if you want a semi-manual workflow: you select a rectangular screen region, it repeatedly screenshots that region, runs Tesseract locally, and uses duplicate/novelty filtering to stitch text over time. Its README explicitly mentions ROVER plus TF-IDF novelty scoring to filter duplicates while preserving new content. ([GitHub][8])

These tools are not code-aware. They will not understand that `l` vs `1`, `{` vs `(`, `:` vs `;`, `rn` vs `m`, or indentation matters. But they are good scaffolding.

---

## Code-specific research / repos

### PSC2CODE: closest to the real problem

PSC2CODE is the closest match conceptually. The repo exists, though it looks like a research artifact rather than a polished tool. Its README says it includes Python source code, a web app for viewing/labeling/validating images, playlists, raw videos, and extracted images. ([GitHub][9])

The paper’s pipeline is almost exactly the pipeline I would use: remove non-informative frames, remove non-code/noisy-code frames with a CNN classifier, detect/crop likely code editor regions using edge detection and clustering, OCR the cropped region, and then correct OCR errors using cross-frame information plus a statistical source-code language model. ([SOAR][10])

PSC2CODE reported a CNN valid-code-frame F1 score of 0.95 and built applications such as a programming screencast search engine and an enhanced video player. ([arXiv][11])

### Codemotion: relevant, but not turnkey

Codemotion is another directly relevant research system. The paper says it segments videos into regions likely to contain code, performs OCR, recognizes source code, and merges related code edits into intervals. In its evaluation on 20 YouTube videos, it found 94.2% of code-containing segments with an OCR error rate of 11.2%. ([Philip Guo][12])

The GitHub repo exists, but its own README says: “This repo is not really documented,” which is a giant warning label. ([GitHub][13])

### ACE: important paper, but I did not find a usable public repo

The ACE paper, “Extracting Code from Programming Tutorial Videos,” describes a code-extraction approach based on consolidating code across frames and using statistical language models for corrections at token, line-structure, and fragment levels. ([Machine Learning for Big Code][14])

This is exactly the right idea, but I did not find a clearly usable public ACE implementation. Treat it as architecture inspiration, not a tool to install.

### CodeSCAN: useful if you want to train/benchmark detectors

CodeSCAN is a newer 2025 research project/dataset for screencast analysis. It introduces 12,000 VS Code screenshots across 24 programming languages, 25 fonts, over 90 themes, layout changes, and realistic interactions; it benchmarks IDE element detection, black-and-white conversion, and OCR. ([arXiv][15])

The GitHub repo is a project-page/code repo rather than a packaged extractor, but it is useful if you want to build a classifier that says, “this frame contains code / this box is the editor / ignore file tree / ignore terminal / ignore narrator.” ([GitHub][16])

---

## The pipeline I would actually use

### Stage 1: extract candidate frames

Do not OCR 30 fps video. That is wasteful and noisy.

Start with something like:

```bash
mkdir -p frames_1fps
ffmpeg -i lesson.mp4 -vf "fps=1,scale=1920:-1" frames_1fps/frame_%06d.jpg
```

Then add scene-change frames:

```bash
mkdir -p scene_frames
ffmpeg -i lesson.mp4 -vf "select='gt(scene,0.08)',scale=1920:-1" -vsync vfr scene_frames/scene_%06d.jpg
```

Or use PySceneDetect:

```bash
scenedetect -i lesson.mp4 detect-adaptive save-images -o scene_frames
```

For code flashes, I would generally use **1–2 fps sampling plus scene frames**. If code is flashed for less than half a second, go to 4 fps. If it is lecture slides / demos, 1 fps is usually enough.

### Stage 2: deduplicate

Compute perceptual hash or SSIM between adjacent frames. Keep only frames that changed enough. Also keep the timestamp.

You want output like:

```text
00:03:14.000 frame_000194.jpg
00:03:15.000 frame_000195.jpg
00:07:42.000 frame_000462.jpg
```

### Stage 3: detect “code-ish” frames

Run a cheap OCR pass or text detector. A frame is a candidate if it has enough visible text and the text contains code-like signals:

```text
def, class, import, from, return, if, else, for, while
{}, [], (), =>, ==, !=, <=, >=
camelCase, snake_case, dotted.names
; at line ends
indentation
HTML/XML tags
SQL SELECT/FROM/WHERE
shell prompts, pip install, npm install
```

This is a much better detector than scene change. Scene detection asks: “did pixels change?” You want: “does this frame contain code?”

### Stage 4: crop likely code region

If the course layout is stable, manually define a crop once:

```text
x=250 y=120 width=1400 height=800
```

Then apply it to every candidate frame.

If the layout changes, use OCR bounding boxes and merge text-dense rectangles. Ignore facecam boxes, slides title text, browser chrome, file tree, and terminal unless those are relevant.

### Stage 5: OCR / vision extraction

For each candidate crop, run:

1. **PaddleOCR** locally for cheap extraction.
2. **Vision LLM** only for frames with poor OCR, low confidence, tiny text, or snippets you actually care about.
3. Preserve timestamp + screenshot filename + raw OCR.

Prompt the vision LLM like this:

```text
Extract only the code that is visibly present in this screenshot.
Do not infer missing lines.
Do not complete partial code.
Preserve indentation, punctuation, capitalization, and line breaks.
If a character is ambiguous, mark it as [?] or provide alternatives.
Return only a fenced code block, no explanation.
```

### Stage 6: merge across frames

This is where most crude OCR projects fail. You need to merge repeated, partially overlapping snippets.

Use one of these strategies:

**For static snippets:** group near-identical OCR outputs by edit distance, keep the best/highest-confidence version.

**For live coding:** treat every frame as a possible version of a file and run diffs over time.

**For brief flashed snippets:** use an LLM with all OCR candidates and screenshots/time metadata:

```text
Here are OCR outputs from timestamped screenshots.
Merge duplicates.
Keep only code that appears in screenshots.
When two versions conflict, prefer the version visible in more frames or the clearer screenshot.
Do not invent missing imports or setup.
Output a clean script plus comments indicating timestamps.
```

I would output something like:

```python
# From lesson.mp4 @ 00:03:14–00:03:18
import jax
import jax.numpy as jnp

# From lesson.mp4 @ 00:07:42–00:07:48
def loss_fn(params, batch):
    ...
```

And keep a sidecar JSON:

```json
[
  {
    "timestamp": "00:03:14.000",
    "screenshot": "frames/frame_000194.jpg",
    "raw_ocr": "...",
    "cleaned_code": "..."
  }
]
```

That sidecar is essential because you will need to audit mistakes.

---

## My ranked recommendation

### Best practical path

Use:

1. **FFmpeg / PySceneDetect** for frame extraction.
2. **PaddleOCR** for local detection/OCR.
3. **A vision LLM** for hard frames.
4. **A code LLM** for merging and repairing OCR output.
5. **Keep timestamped screenshots** as evidence.

This is the best balance of quality, cost, and controllability.

### Best no-code / low-code path

Try **VideOCR** or **CaptiOCR** with a manually selected crop region, then feed the extracted text and screenshots to a vision/code LLM for cleanup. This is the fastest way to see whether your videos are amenable to automation.

### Best research repo to adapt

Start from **PSC2CODE**, not because it is polished, but because its architecture is closest to the real task: valid-code-frame detection, editor-region cropping, OCR, cross-frame correction. ([GitHub][9])

### Best modern research asset

Use **CodeSCAN** if you want to train or benchmark a detector for VS Code / editor screenshots. It is not the extractor, but it is the dataset/benchmark most aligned with “detect code regions in programming screencasts.” ([arXiv][15])

---

## Bottom line

There is no magic one-click tool I would trust. The winning architecture is:

```text
video
  -> sample frames + scene frames
  -> deduplicate
  -> OCR/text-detect cheaply
  -> score for code-likeness
  -> crop likely code regions
  -> OCR or vision-LLM extraction
  -> merge overlapping snippets across timestamps
  -> code-aware cleanup
  -> final script + screenshot/time provenance
```

For occasional flashed snippets, this should work very well. For continuous live coding with scrolling, popups, cursor occlusion, and partial edits, it becomes a real computer-vision/code-reconstruction problem — exactly why PSC2CODE, Codemotion, ACE, and CodeSCAN exist.

[1]: https://github.com/PADDLEPADDLE/PADDLEOCR?utm_source=chatgpt.com "GitHub - PaddlePaddle/PaddleOCR: Turn any PDF or image ..."
[2]: https://github.com/tesseract-ocr/tesseract?utm_source=chatgpt.com "Tesseract Open Source OCR Engine (main repository)"
[3]: https://www.mdpi.com/2227-7390/12/7/1036 "Optimizing OCR Performance for Programming Videos: The Role of Image Super-Resolution and Large Language Models | MDPI"
[4]: https://github.com/breakthrough/pyscenedetect "GitHub - Breakthrough/PySceneDetect: :movie_camera: Python and OpenCV-based scene cut/transition detection program & library. · GitHub"
[5]: https://ffmpeg.org/ffmpeg-filters.html?utm_source=chatgpt.com "FFmpeg Filters Documentation"
[6]: https://github.com/timminator/VideOCR "GitHub - timminator/VideOCR: Extract hardcoded subtitles from videos via a simple GUI using machine learning. Supports 200+ languages. · GitHub"
[7]: https://github.com/knakamura13/videocr-PaddleOCR "GitHub - knakamura13/videocr-PaddleOCR: Extract hardcoded subtitles from videos using machine learning · GitHub"
[8]: https://github.com/carlosacchi/captiocr "GitHub - carlosacchi/captiocr: CaptiOCR - A real-time screen text extraction tool using Tesseract OCR. Capture, recognize, and log on-screen text dynamically. Future updates will include on-demand language installation, resizable selection areas, and live text overlays. · GitHub"
[9]: https://github.com/baolingfeng/PSC2CODE "GitHub - baolingfeng/PSC2CODE · GitHub"
[10]: https://soarsmu.github.io/papers/2020/Bao2020psc2code.pdf "TOSEM2903-21"
[11]: https://arxiv.org/abs/2103.11610?utm_source=chatgpt.com "psc2code: Denoising Code Extraction from Programming Screencasts"
[12]: https://pg.ucsd.edu/publications/Codemotion-programming-tutorial-video-interfaces_LAS-2018.pdf "paper"
[13]: https://github.com/kandarpksk/codemotion-las2018 "GitHub - kandarpksk/codemotion-las2018: Codemotion: Expanding the Design Space of Interactions with Computer Programming Tutorial Videos · GitHub"
[14]: https://ml4code.github.io/publications/yadid2016extracting/ "
    
      Extracting Code from Programming Tutorial Videos · Machine Learning for Big Code and Naturalness
    
  "
[15]: https://arxiv.org/abs/2409.18556?utm_source=chatgpt.com "CodeSCAN: ScreenCast ANalysis for Video Programming Tutorials"
[16]: https://github.com/a-nau/codescan "GitHub - a-nau/codescan: Code of our VISAPP '25 paper \"CodeSCAN: ScreenCast ANalysis for Video Programming Tutorials\". · GitHub"

