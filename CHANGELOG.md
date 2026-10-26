# Changelog

## 0.1.0

- Package the existing V4 drawing assistant as its first public-source release.
- Preserve the drawing, erasing, undo/redo, completion, raw-reference overlay, local cleanup, AI correction, region locks, project save/load and PNG export features.
- Start the canvas without model dependencies; default new installations to Klein.
- Install dependencies in a fresh Python 3.11/3.12 venv and select the requested model backend directly.
- Accept UTF-8 BOM in the Windows-written backend configuration and allow an explicit existing model directory.
- Ignore repeated submission of the same active completion/correction from the UI.
- Return a controlled error for malformed Content-Length.
- Include an original sketch and its real local GPU inference result, with measured limitations.
- Exclude old personal drawings, local environments, caches, weights, and private diagnostic fixtures from the release.
