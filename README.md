# DrawingLab v0.1

**English** | [简体中文](README.zh-CN.md)

The GitHub repository homepage displays English by default. Use the links above to switch languages.

Draw a few strokes and let a local AI model suggest a completed reference while you stay in control of your own drawing.

DrawingLab is a local web drawing assistant with a fixed 3 px black pen, an eraser, undo/redo, faint AI references, gentle vector stroke corrections, JSON project saving, and PNG export. This first public release preserves the V4 prototype's interactions and project format: `v0.1.0` is the release version; API `version: 4` and project `app_version: 4` refer to the existing protocol.

## Real demo

These images come from an actual run, not a mock, a promotional rendering, or a UI screenshot. The left image is an original cat outline drawn with mouse drags in the browser canvas and saved with **Export PNG**. The right image is the **raw Klein-generated reference** saved by the server. The user has not adopted that reference as strokes, so it is not the result of Export PNG.

| Original input: a cat-head outline in 10 strokes | Complete reference generated locally by Klein |
| --- | --- |
| ![Actual sketch input](docs/images/cat-input.png) | ![Actual Klein model output](docs/images/cat-klein-output.png) |

Prompt, translated from Chinese: a sitting kitten; retain the pointed ears and facial outline; complete the body, paws, and curled tail; simple black line art on a plain white background.

On 2026-10-26, an RTX 3060 Laptop GPU produced this 768×768 result using 4 steps, seed 42, BF16, and CPU offload. The first request took **49.19 seconds**, including 20.016 seconds of loading and 29.062 seconds of inference. Peak GPU tensor allocation was approximately **8.58 GiB**, which is not total device memory usage or a performance guarantee for other devices. This example triggered a contour-deviation warning: the model did not preserve every outline precisely. It does not establish semantic accuracy or drawing quality.

Importable source: [cat-sketch.json](examples/cat-sketch.json). Recorded metrics: [cat-metrics.json](examples/cat-metrics.json). Browser screenshot capture timed out in the validation environment, so this demo uses unretouched input/output images, not UI screenshots.

## Quick start: use the canvas first

Recommended: Windows 10/11 and Python **3.11 or 3.12**. The basic canvas requires no GPU, model, or API key.

```powershell
# Run in the project directory; python must be Python 3.11/3.12
./setup.ps1
./start.ps1
```

If your default Python version differs, use `./setup.ps1 -Python 'C:\path\to\Python311\python.exe'`. Open **http://127.0.0.1:8765**. You can also double-click `start.cmd`. Only one instance can use a given port.

The first setup creates a project-local `.venv` and installs dependencies from PyPI. When model files are missing, the app explicitly reports that the model is not ready; drawing, geometric line cleanup, saving, and export still work. It does not fabricate AI results. There is no frontend build step: FastAPI serves the HTML, CSS, and JavaScript directly.

On other systems, you can manually create a Python 3.11/3.12 venv, run `python -m pip install -r requirements.txt -r requirements-test.txt`, and start `python -m drawinglab.server`. This release has not been validated on Linux or macOS hardware.

## Install the local AI model

The default backend is **FLUX.2 klein 4B**. An NVIDIA CUDA GPU with BF16 support is recommended; this release was validated on a device with 12 GB VRAM. Model files alone occupy about **14.883 GiB**. PyTorch, Python packages, caches, and runtime memory are additional requirements; allow at least 30 GB of free disk space. CPU offload also uses system RAM. CPU inference can be selected explicitly, but its performance has not been validated for this release.

```powershell
./setup.ps1 -WithModels
./start.ps1
```

This explicitly downloads CUDA 12.8 PyTorch, Klein dependencies, and model files. It requires access to PyPI, PyTorch, and Hugging Face. Downloads are large; any network or data charges depend on your connection. Read the upstream model terms before use. The **4B** model's downloaded license is Apache 2.0; this statement does not apply to other model variants. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

The model is pinned to revision `e7b7dc27f91deacad38e78976d1f2b499d76a294`. The downloader supports resuming and verifies SHA-256 for files with hashes in the manifest; other small files are checked by size. At runtime, the app loads local safetensors with `local_files_only=True` and does not call a cloud model API. Model weights are not included in this repository.

```powershell
./.venv/Scripts/python.exe scripts/download_klein.py --plan
./.venv/Scripts/python.exe scripts/download_klein.py --verify-only
```

The original CounterfeitXL + anime-painter backend remains available. Install it in a separate checkout/venv with `./setup.ps1 -WithModels -Engine counterfeit` to avoid mixing its Diffusers dependencies with Klein's. Optional DoodleAssist integration code is also retained, but its base model and upstream licensing require separate review. **These integrations have not completed end-to-end generation validation and are not validated v0.1 backends.**

## Workflow

The current app controls are in Chinese; the labels below help you find them.

1. Describe what you want to draw. Klein accepts Chinese prompts. After you draw a few strokes, the app generates a faint reference automatically; you can also click **更新整幅参考** (update the full reference).
2. Keep drawing over the suggestions you want. **换一个** (another variation) changes the seed; **固定参考** (pin reference) pauses updates. Reference opacity can be adjusted from 3% to 20%.
3. **叠加 AI 原始参考图** (overlay the raw AI reference) shows the model's complete suggestion. It includes content you have not adopted and is excluded from model input, project strokes, and PNG export.
4. **AI 修稿** (AI correction) proposes small curve adjustments; **整理线条** (clean up lines) performs local geometric smoothing only. You can select a region, lock strokes, adjust correction strength, and compare against the original. The drawing changes only after **应用修稿** (apply correction). The whole operation can be undone.
5. **保存项目** (save project) writes editable JSON to `data/projects`; **导出 PNG** (export PNG) writes your actual strokes to `data/exports`. Faint AI references and unapplied previews are excluded from the PNG. The complete model image is saved separately at `data/sessions/<session>/candidate.png` for diagnostics/reference only.
6. Open an existing project, or import the example through **打开项目 → 导入 JSON** (open project → import JSON). Shortcuts: `B` pen, `E` eraser, `S` selection, `Ctrl+Z` undo, `Ctrl+Shift+Z` / `Ctrl+Y` redo.

Unsaved strokes exist only in the current browser tab. Save before closing, refreshing, or changing versions. You can keep drawing during inference. Stale results cannot overwrite newer revisions; each session keeps only its latest pending job, and one GPU worker runs globally.

## Configuration and data

See [.env.example](.env.example). The application **does not load `.env` automatically**. Set the documented environment variables in your current PowerShell process:

```powershell
$env:DRAWINGLAB_ENGINE = 'klein'
$env:DRAWINGLAB_PORT = '8765'
# Optional: reuse a model directory containing a flux2-klein-4b subdirectory
$env:DRAWINGLAB_MODEL_DIR = 'D:\your-models'
./start.ps1
```

Models default to `models/`. `DRAWINGLAB_MODEL_DIR` changes only the model read location; caches, temporary files, logs, and user drawings remain in this project. `models/active-engine.json` can select a backend, with environment variables taking precedence. Without configuration, Klein is the default. No API key, cloud account, or paid subscription is required. Do not commit credentials, personal artwork, or environment directories to Git.

The server binds only to `127.0.0.1` for local, single-user use. It has no public-facing authentication, account isolation, or multi-tenant design. Do not expose it directly to the public internet.

## Validation

```powershell
./.venv/Scripts/python.exe -m pytest tests --basetemp=.tmp/test-release -o cache_dir=.cache/pytest -q
./.venv/Scripts/python.exe -m pip check
node --check static/app.js
```

In a freshly installed basic environment, **95 Python tests passed**, along with `pip check` and JavaScript syntax validation. Six additional local HTTP checks passed for error recovery, saving, and export. Tests cover geometry, stale-result scheduling, input validation, saving/export, missing models, model-error recovery, corrections, and locking. Test doubles do not establish model drawing quality; the real GPU demo is recorded separately above. See [docs/VALIDATION.md](docs/VALIDATION.md) for the full release validation record.

The retained browser regression scripts use Playwright: run `npm install`, start the app, then run `npm run test:browser`. V3 transport tests use labeled PNG fixtures, not model-quality acceptance tests. Development diagnostics for old protocols or private legacy drawings are excluded from the release package and remain in the original development directory.

## Limitations and licensing

- References update after you pause drawing, not in real time on every stroke. Speed depends on GPU, memory, image content, and model loading.
- Generated raster references can drift from your outlines. Statistical style hints are not a trained personal-style model, and anatomy, fingers, or local semantic details are not guaranteed to be correct.
- Gentle corrections use heuristics to preserve stroke topology, processing at most 256 eligible strokes with 2,500 points per stroke. Parts of complex drawings may remain unchanged. The app does not automatically adopt all AI-generated black lines as your strokes.
- Real GPU validation reused existing model files and GPU dependencies. A fresh basic installation was verified, but the full approximately 15 GB model download and a fresh GPU dependency installation were not repeated.
- The original project provided no software license. This release **does not grant an additional open-source license**. Public visibility is not a blanket grant to copy, modify, commercialize, or redistribute the application. Third-party models and libraries retain their own terms; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
