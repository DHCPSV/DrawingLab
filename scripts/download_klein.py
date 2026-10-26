"""Download the pinned official FLUX.2 klein 4B assets into this project.

Transfers resume .part files and verify published SHA-256 for every weight.
No pickle checkpoints, executable model code, or duplicate transformer assets.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from drawinglab.klein_manifest import KLEIN_DIRECTORY, KLEIN_FILES, KLEIN_MODEL, KLEIN_REVISION
from download_models import configure_local_paths, download_file, log


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', action='store_true')
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--workers', type=int, choices=range(1, 5), default=3)
    args = parser.parse_args()
    total = sum(spec[1] for spec in KLEIN_FILES)
    log(f'FLUX.2 klein 4B: {total:,} bytes ({total / 1024**3:.3f} GiB), {len(KLEIN_FILES)} files')
    if args.plan:
        log(json.dumps({'repo': KLEIN_MODEL, 'revision': KLEIN_REVISION, 'directory': f'models/{KLEIN_DIRECTORY}', 'files': KLEIN_FILES}, indent=2))
        return 0
    configure_local_paths(ROOT)
    if not args.verify_only:
        present = sum(min((ROOT / 'models' / KLEIN_DIRECTORY / name).stat().st_size, size)
                      for name, size, _ in KLEIN_FILES if (ROOT / 'models' / KLEIN_DIRECTORY / name).is_file())
        if shutil.disk_usage(ROOT).free < total - present + 1024**3:
            parser.error('Not enough free space in the project drive for model assets and 1 GiB working space.')
    receipts = []
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(download_file, ROOT, KLEIN_MODEL, KLEIN_REVISION, KLEIN_DIRECTORY, spec, args.verify_only)
                       for spec in sorted(KLEIN_FILES, key=lambda item: item[1], reverse=True)]
            for future in concurrent.futures.as_completed(futures):
                receipts.append(future.result())
    except (KeyboardInterrupt, Exception) as exc:
        log(f'STOPPED: {exc}. Project-local partial files can be resumed with the same command.')
        return 1
    receipt = ROOT / 'models' / 'download-receipt-klein.json'
    receipt.write_text(json.dumps({'model': KLEIN_MODEL, 'revision': KLEIN_REVISION, 'total_bytes': total, 'files': receipts}, indent=2), encoding='utf-8')
    log(f'All Klein assets verified: {receipt}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
