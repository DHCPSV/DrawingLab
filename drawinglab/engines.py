"""Choose a locally installed backend without changing the original environment."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys


def create_engine(root: Path):
    root = Path(root).resolve()
    settings = root / "models" / "active-engine.json"
    default = json.loads(settings.read_text(encoding="utf-8-sig")).get("engine", "klein") if settings.is_file() else "klein"
    choice = os.environ.get("DRAWINGLAB_ENGINE", default).lower()
    if choice == "klein":
        libraries = root / ".runtime" / "klein-libs"
        # Only this application process sees these packages. The original venv
        # remains available for the SDXL/Doodle backends and offline tests.
        if libraries.is_dir():
            sys.path.insert(0, str(libraries))
        from .klein_engine import KleinModelEngine
        return KleinModelEngine(root)
    if choice == "doodle":
        from .doodle_engine import DoodleEngine
        return DoodleEngine(root)
    if choice == "counterfeit":
        from .inference import ModelEngine
        return ModelEngine(root)
    raise ValueError("DRAWINGLAB_ENGINE 必须为 klein、doodle 或 counterfeit")
