"""Prepare optional DoodleAssist assets strictly inside this project.

No third-party executable is launched. The small author checkpoint is loaded with
torch weights_only=True and converted to safetensors. The Civitai base must be the
official foolkatGODOF_v3 checkpoint; a blocked download is reported, never replaced.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from drawinglab.paths import configure_paths
from drawinglab.doodle_engine import SOURCE_REVISION, CONTROL_SOURCE_SHA256, CONTROL_SHA256, BASE_URL

DIRECTORY = ROOT / "models/doodle-assist"
ADAPTER_URL = "https://drive.usercontent.google.com/download?id=1auWNemiIeChDxkUfPoGLBqdiF9AZhFM-&export=download&confirm=t"
BASE_API = "https://civitai.com/api/v1/model-versions/142306"
SD15_REVISION = "451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
CONFIG_FILES = ["model_index.json", "scheduler/scheduler_config.json", "text_encoder/config.json",
                "tokenizer/merges.txt", "tokenizer/vocab.json", "tokenizer/special_tokens_map.json",
                "tokenizer/tokenizer_config.json", "unet/config.json", "vae/config.json",
                "feature_extractor/preprocessor_config.json", "v1-inference.yaml"]


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(4 * 1024 * 1024):
            result.update(block)
    return result.hexdigest()


def local_path(path: Path) -> Path:
    result = path.resolve()
    if not result.is_relative_to(ROOT):
        raise ValueError("All DoodleAssist inputs and outputs must be inside the project directory.")
    return result


def download(url: str, target: Path, sha256: str | None = None, max_bytes: int = 20_000_000):
    target = local_path(target)
    if target.is_file() and (sha256 is None or digest(target) == sha256):
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "DrawingLab-local-DoodleAssist/1.0"})
    size = 0
    with urllib.request.urlopen(request, timeout=45) as response, partial.open("wb") as file:
        if not response.url.startswith("https://"):
            raise RuntimeError("Refusing non-HTTPS redirect")
        while block := response.read(4 * 1024 * 1024):
            size += len(block)
            if size > max_bytes:
                raise RuntimeError(f"Download exceeds expected bound: {target.name}")
            file.write(block)
    if sha256 and digest(partial) != sha256:
        raise RuntimeError(f"SHA-256 mismatch: {target.name}")
    os.replace(partial, target)
    print(f"Prepared {target.relative_to(ROOT)} ({size:,} bytes)", flush=True)


def prepare_adapter(offline: bool):
    source = DIRECTORY / "source/controlnext.py"
    binary = DIRECTORY / "controlnext-48000.bin"
    if not offline:
        download(f"https://raw.githubusercontent.com/MarkMoHR/DoodleAssist/{SOURCE_REVISION}/models/controlnext.py",
                 source, CONTROL_SOURCE_SHA256)
        download(ADAPTER_URL, binary, CONTROL_SHA256)
    if not source.is_file() or digest(source) != CONTROL_SOURCE_SHA256:
        raise RuntimeError("Missing or incorrect pinned ControlNeXt source")
    if not binary.is_file() or digest(binary) != CONTROL_SHA256:
        raise RuntimeError("Missing or incorrect official adapter")
    from safetensors.torch import save_file
    import torch
    # The official .bin contains plain tensors. Never fall back to unsafe pickle.
    weights = torch.load(binary, map_location="cpu", weights_only=True)
    if not isinstance(weights, dict) or not all(isinstance(value, torch.Tensor) for value in weights.values()):
        raise RuntimeError("Adapter must contain only a dictionary of tensors")
    safe = DIRECTORY / "controlnext-48000.safetensors"
    save_file(weights, str(safe), metadata={"source_sha256": CONTROL_SHA256, "source_revision": SOURCE_REVISION})
    return {"source_revision": SOURCE_REVISION, "source_sha256": CONTROL_SOURCE_SHA256,
            "official_adapter_sha256": CONTROL_SHA256, "safetensors_sha256": digest(safe)}


def acquire_base() -> Path:
    request = urllib.request.Request(BASE_API, headers={"User-Agent": "DrawingLab-local-DoodleAssist/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        manifest = json.load(response)
    (DIRECTORY / "base-download-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    candidates = [entry for entry in manifest.get("files", [])
                  if entry.get("name", "").lower() == "foolkatgodof_v3.safetensors"]
    if len(candidates) != 1:
        raise RuntimeError("Official version manifest did not identify the exact required base checkpoint")
    entry = candidates[0]
    sha256 = entry.get("hashes", {}).get("SHA256")
    if not sha256:
        raise RuntimeError("Official base has no published SHA-256; supply the verified checkpoint manually")
    destination = DIRECTORY / "foolkatGODOF_v3.safetensors"
    download(entry["downloadUrl"], destination, sha256.lower(), max_bytes=5_000_000_000)
    return destination


def convert_base(checkpoint: Path, offline: bool):
    checkpoint = local_path(checkpoint)
    if checkpoint.suffix.lower() != ".safetensors" or not checkpoint.is_file():
        raise ValueError("--base-checkpoint must point to a local .safetensors file within this project")
    config = DIRECTORY / "base-config"
    if not offline:
        for name in CONFIG_FILES:
            download(f"https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/{SD15_REVISION}/{name}",
                     config / name)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from diffusers import StableDiffusionPipeline
    pipeline = StableDiffusionPipeline.from_single_file(
        str(checkpoint), config=str(config), original_config=str(config / "v1-inference.yaml"),
        local_files_only=True, torch_dtype=torch.float16,
        safety_checker=None, feature_extractor=None, requires_safety_checker=False,
    )
    pipeline.save_pretrained(str(DIRECTORY / "base"), safe_serialization=True)
    return {"checkpoint": str(checkpoint.relative_to(ROOT)), "checkpoint_sha256": digest(checkpoint),
            "sd15_config_revision": SD15_REVISION, "dtype": "float16"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-only", action="store_true")
    parser.add_argument("--base-checkpoint", type=Path, help="Official foolkatGODOF_v3.safetensors already inside project")
    parser.add_argument("--offline", action="store_true", help="Convert/validate already-local files without any network")
    args = parser.parse_args()
    configure_paths()
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    receipt = prepare_adapter(args.offline)
    receipt.update(base_source=BASE_URL, base_ready=False,
                   notice="Private local experiment. Upstream repository has no explicit root LICENSE file.")
    try:
        if not args.adapter_only:
            checkpoint = args.base_checkpoint
            if checkpoint is None and args.offline:
                raise RuntimeError("Offline base conversion requires --base-checkpoint")
            checkpoint = checkpoint or acquire_base()
            receipt["base"] = convert_base(checkpoint, args.offline)
            receipt["base_ready"] = True
    except (OSError, ValueError, RuntimeError) as exc:
        receipt["base_error"] = str(exc)
        print(f"Base unavailable: {exc}\nOfficial download: {BASE_URL}\n"
              "Place foolkatGODOF_v3.safetensors in models/doodle-assist, then run:\n"
              r".venv\Scripts\python.exe scripts\download_doodle_models.py --base-checkpoint models/doodle-assist/foolkatGODOF_v3.safetensors", flush=True)
        (DIRECTORY / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        return 2
    (DIRECTORY / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
