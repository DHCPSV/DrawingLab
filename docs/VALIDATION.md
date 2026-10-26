# v0.1.0 validation record

Date: 2026-10-26. Platform: Windows, Python 3.11.16, local NVIDIA RTX 3060 Laptop GPU.

## Passed

- A fresh project venv was created by `setup.ps1`; all core and test dependencies installed successfully from PyPI.
- `python -m pip check`: no broken requirements.
- `pytest tests --basetemp=.tmp/test-release-final -o cache_dir=.cache/pytest-final -q`: **95 passed in 6.68s**. One Starlette/AnyIO deprecation warning, no test failures.
- `node --check static/app.js`: passed.
- New regressions cover boot without Klein libraries/models, UTF-8 BOM backend selection, external model-directory isolation and invalid Content-Length recovery.
- Existing tests cover missing model/HTTP 503, model errors and later requests, stale completion/correction results, lock/region bounds, edit geometry and API save/export behavior. Model doubles are contract tests, not AI quality evidence.
- Real UI at the local application server: prompt entry, disabling auto-suggest, drawing a new cat-head outline using pointer drags, saving its 10-stroke editable project, exporting PNG, requesting Klein completion, and showing the raw model reference.
- A rapid repeated click of the completion button was exercised during the first real request; the page completed normally with the original 10 strokes intact. This is a behavioral smoke check, not an instrumented assertion of GPU invocation count.
- Real offline Klein completed the new input with seed 42, 768px, 4 steps. First request: 49.19s including model load; subsequent same-input request: 13.11s. These are observed examples, not a benchmark guarantee.
- SHA-256 of UI-exported PNG before and after model completion is identical. Thus unadopted AI references did not leak into the artwork export.
- Real UI undo reduced the drawing from 10 to 9 strokes; redo restored 10, and the exported pixels matched the original exactly. Erasing the top contour changed exported pixels; undoing the eraser restored the original exactly.
- Real local cleanup proposed 7 stroke adjustments in 0.11s. Moving strength to 70% did not change the exported artwork until Apply was clicked. The applied export changed; one Undo restored the original pixels exactly.
- `scripts/verify_release_api.py --port 8766`: 6 checks passed against the real local server, including invalid image rejection followed by successful save/reopen/export, without any model call.
- The included images were visually inspected: an original angular cat-head outline and the actual model-completed sitting cat. No private source drawing, stock image or third-party character was used.

## Honest limits / not passed as evidence

- The model output changes some input contours. `alignment_warning: true`, `semantic_alignment_verified: false`, `personal_style_learned: false`; generation success does not certify anatomy, exact shape preservation or professional quality.
- The browser screenshot API repeatedly timed out. No fabricated UI screenshot is included. README images are unaltered application input/output PNGs.
- GPU inference reused the previously installed local model and GPU libraries. A clean download of all 15GB of weights and clean GPU dependency installation were not repeated. Basic CPU installation was tested independently in a new venv.
- Full manual UI coverage of region locking and an AI-generated correction proposal was not completed in this run (the cleanup proposal/strength/apply/undo path was exercised). Their offline tests pass; the retained Playwright regression scripts have not been rerun for this release.
- Linux/macOS, CPU-only model inference, Counterfeit and Doodle full generation were not tested for this release.
- No CI has been added. Remote commit visibility, README image links, release tag and GitHub author attribution must be checked after publication; local validation does not establish those facts.

## Demo provenance

- `examples/cat-sketch.json`: actual JSON saved from the UI for the new 10-stroke sample.
- `docs/images/cat-input.png`: actual UI PNG export before generation.
- `docs/images/cat-klein-output.png`: first real generation's `candidate.png`, copied without enhancement.
- `examples/cat-metrics.json`: first real generation's recorded metrics, including model revision, prompt, device, timing and limitations. No API secrets or private artwork.
- The original source tree was not a Git checkout. This release is a separate directory; original environments, weights, caches, private drawings and old diagnostic scripts remain outside the release.
