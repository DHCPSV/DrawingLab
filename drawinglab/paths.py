from pathlib import Path
import os
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]

def configure_paths() -> Path:
    """Force all application-owned caches and temporary files into the project."""
    directories = {
        "HF_HOME": ".cache/huggingface",
        "HF_HUB_CACHE": ".cache/huggingface/hub",
        "HF_XET_CACHE": ".cache/huggingface/xet",
        "TORCH_HOME": ".cache/torch",
        "TORCHINDUCTOR_CACHE_DIR": ".cache/torchinductor",
        "TRITON_CACHE_DIR": ".cache/triton",
        "XDG_CACHE_HOME": ".cache",
        "PIP_CACHE_DIR": ".cache/pip",
        "MPLCONFIGDIR": ".cache/matplotlib",
        "TEMP": ".tmp", "TMP": ".tmp", "TMPDIR": ".tmp",
    }
    for key, relative in directories.items():
        directory = ROOT / relative
        directory.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(directory)
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["DO_NOT_TRACK"] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    tempfile.tempdir = str(ROOT / ".tmp")
    for relative in ("models", "data/projects", "data/sessions", "logs"):
        (ROOT / relative).mkdir(parents=True, exist_ok=True)
    return ROOT
