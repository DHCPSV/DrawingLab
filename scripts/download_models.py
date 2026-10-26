"""Download only the official, pinned local SDXL components (stdlib only).

Run with --plan first to print the exact manifest and size. Downloads use HTTPS,
HTTP Range resumption, atomic final names, and published LFS SHA-256 checksums.
No Python code, pickle checkpoint, or duplicate fp32 model is downloaded.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
BASE_REVISION = "4708675873bd09833aabc3fd4cb2de5fcd1726ac"
CONTROL_REVISION = "18185a73b6e7fe49f2f2de1bb9d7db0b74a41773"

# Hugging Face /api/models/{repo}?blobs=true, checked 2026-09-07.
BASE_FILES = [
    ("model_index.json", 577, None),
    ("scheduler/scheduler_config.json", 474, None),
    ("text_encoder/config.json", 565, None),
    ("text_encoder/model.fp16.safetensors", 246144152, "1aa4aa92f9aa8dc9668e3745fbab3047926834c489307c163f850167c0ed7bf1"),
    ("text_encoder_2/config.json", 575, None),
    ("text_encoder_2/model.fp16.safetensors", 1389382176, "08c64bacea5a24bffbe3d86271017926ffc8103c1b5ffb30525b6ea13dc74ddd"),
    ("tokenizer/merges.txt", 524619, None),
    ("tokenizer/special_tokens_map.json", 472, None),
    ("tokenizer/tokenizer_config.json", 737, None),
    ("tokenizer/vocab.json", 1059962, None),
    ("tokenizer_2/merges.txt", 524619, None),
    ("tokenizer_2/special_tokens_map.json", 460, None),
    ("tokenizer_2/tokenizer_config.json", 725, None),
    ("tokenizer_2/vocab.json", 1059962, None),
    ("unet/config.json", 1675, None),
    ("unet/diffusion_pytorch_model.fp16.safetensors", 5135149760, "03f920e662cc354599c94f0e6428362e3d7887ba6b18fa55865518412d395192"),
    ("vae/config.json", 602, None),
    ("vae/diffusion_pytorch_model.fp16.safetensors", 167335342, "4902cfc6e3ac21b22b2eab4140cdb9f835415f01436754dc8d43128a5bac604d"),
]
CONTROL_FILES = [
    ("config.json", 1235, None),
    ("diffusion_pytorch_model.safetensors", 2502139104, "053e9427d3936ac2f1ed766d43cfc119a49a44938aa2e3475c7f01e4a0c476f3"),
]
COMPONENTS = {
    "base": ("gsdf/CounterfeitXL", BASE_REVISION, "counterfeit-xl", BASE_FILES),
    "control": ("xinsir/anime-painter", CONTROL_REVISION, "anime-painter", CONTROL_FILES),
}
PRINT_LOCK = threading.Lock()


def log(message: str) -> None:
    with PRINT_LOCK:
        print(message, flush=True)


def configure_local_paths(root: Path) -> None:
    paths = {
        "HF_HOME": root / ".cache/huggingface",
        "HF_HUB_CACHE": root / ".cache/huggingface/hub",
        "HF_XET_CACHE": root / ".cache/huggingface/xet",
        "TORCH_HOME": root / ".cache/torch",
        "TEMP": root / ".cache/tmp",
        "TMP": root / ".cache/tmp",
        "TMPDIR": root / ".cache/tmp",
        "XDG_CACHE_HOME": root / ".cache",
    }
    for name, path in paths.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def verify(path: Path, size: int, sha256: str | None) -> bool:
    if not path.is_file() or path.stat().st_size != size:
        return False
    if sha256 and file_digest(path) != sha256:
        return False
    if path.suffix == ".json":
        try:
            with path.open(encoding="utf-8") as source:
                json.load(source)
        except (ValueError, UnicodeError):
            return False
    return True


def download_file(root: Path, repo: str, revision: str, directory: str,
                  spec: tuple[str, int, str | None], verify_only: bool) -> dict:
    filename, expected_size, expected_hash = spec
    destination = root / "models" / directory / filename
    label = f"{directory}/{filename}"
    if verify(destination, expected_size, expected_hash):
        log(f"OK {label} ({expected_size:,} bytes)")
        return {"file": label, "bytes": expected_size, "sha256": expected_hash, "verified": True}
    if verify_only:
        raise RuntimeError(f"Missing, incomplete, or invalid: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # The partial lives inside the same project directory and resumes after interruption.
    partial = destination.with_name(destination.name + ".part")
    url = f"https://huggingface.co/{repo}/resolve/{revision}/{filename}?download=true"
    for attempt in range(1, 6):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            if offset > expected_size:
                partial.unlink()
                offset = 0
            if offset < expected_size:
                headers = {"User-Agent": "DrawingLab-local-model-downloader/1.0"}
                if offset:
                    headers["Range"] = f"bytes={offset}-"
                request = urllib.request.Request(url, headers=headers)
                log(f"GET {label}: {offset:,}/{expected_size:,} bytes (attempt {attempt})")
                with urllib.request.urlopen(request, timeout=60) as response:
                    if not response.url.startswith("https://"):
                        raise RuntimeError("Refusing a non-HTTPS model redirect")
                    if offset and response.status == 206:
                        content_range = response.headers.get("Content-Range", "")
                        if not content_range.startswith(f"bytes {offset}-"):
                            raise RuntimeError(f"Unexpected resume range: {content_range}")
                        mode = "ab"
                    elif response.status == 200:
                        offset = 0  # Server ignored Range; safely restart instead of appending.
                        mode = "wb"
                    else:
                        raise RuntimeError(f"Unexpected HTTP status {response.status}")
                    written = offset
                    last_log = time.monotonic()
                    with partial.open(mode) as output:
                        while block := response.read(4 * 1024 * 1024):
                            written += len(block)
                            if written > expected_size:
                                raise RuntimeError(f"Server returned more bytes than the pinned manifest: {label}")
                            output.write(block)
                            if time.monotonic() - last_log >= 10:
                                log(f"... {label}: {written:,}/{expected_size:,} ({written / expected_size:.1%})")
                                last_log = time.monotonic()
            if not verify(partial, expected_size, expected_hash):
                # A full corrupt file cannot be repaired by appending more bytes.
                if partial.exists() and partial.stat().st_size >= expected_size:
                    partial.unlink()
                raise RuntimeError(f"Size or SHA-256 verification failed: {label}")
            # JSON validation also happens after the .part suffix is removed.
            if destination.suffix == ".json":
                with partial.open(encoding="utf-8") as source:
                    json.load(source)
            os.replace(partial, destination)
            log(f"DONE {label}: {expected_size:,} bytes, verified")
            return {"file": label, "bytes": expected_size, "sha256": expected_hash, "verified": True}
        except (OSError, ValueError, RuntimeError) as exc:
            if attempt == 5:
                raise RuntimeError(f"Could not download {label}: {exc}") from exc
            log(f"RETRY {label}: {exc}")
            time.sleep(min(attempt * 2, 10))
    raise RuntimeError(f"Download failed: {label}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--plan", action="store_true", help="Print the pinned manifest without network or downloads")
    parser.add_argument("--verify-only", action="store_true", help="Verify every local byte and model checksum; no network")
    parser.add_argument("--component", choices=("all", "base", "control"), default="all")
    parser.add_argument("--workers", type=int, default=2, help="Parallel file transfers, 1 to 4 (default: 2)")
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error("--workers must be between 1 and 4")
    root = args.root.resolve()
    jobs = []
    for name, (repo, revision, directory, files) in COMPONENTS.items():
        if args.component in ("all", name):
            for spec in files:
                jobs.append((root, repo, revision, directory, spec, args.verify_only))
    total = sum(job[4][1] for job in jobs)
    log(f"Pinned model download: {total:,} bytes ({total / 1024**3:.3f} GiB), {len(jobs)} files")
    if args.plan:
        for _, repo, revision, directory, spec, _ in jobs:
            log(json.dumps({"repo": repo, "revision": revision, "destination": f"models/{directory}/{spec[0]}", "bytes": spec[1], "sha256": spec[2]}))
        return 0
    configure_local_paths(root)
    receipts = []
    # Larger files start first so small tokenizer assets do not delay their transfer.
    jobs.sort(key=lambda job: job[4][1], reverse=True)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(download_file, *job) for job in jobs]
            for future in concurrent.futures.as_completed(futures):
                receipts.append(future.result())
    except (KeyboardInterrupt, Exception) as exc:
        log(f"STOPPED: {exc}. Partial files remain in the project; run the same command to resume.")
        return 1
    receipt = root / "models" / f"download-receipt-{args.component}.json"
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps({"base_revision": BASE_REVISION, "control_revision": CONTROL_REVISION,
                                   "total_bytes": total, "files": receipts}, indent=2), encoding="utf-8")
    log(f"All selected files verified. Exact total: {total:,} bytes. Receipt: {receipt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
