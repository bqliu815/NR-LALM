#!/usr/bin/env python3
"""Download and verify the two public LIBSVM data sets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=EXPERIMENT_DIR / "configs" / "data_manifest.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=EXPERIMENT_DIR / "data" / "source",
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for item in manifest["files"]:
        destination = args.output_dir / item["file"]
        if not destination.exists():
            partial = destination.with_suffix(destination.suffix + ".part")
            print(f"downloading {item['dataset']} -> {destination}", flush=True)
            urllib.request.urlretrieve(item["url"], partial)
            partial.replace(destination)
        actual = sha256_file(destination)
        if actual != item["sha256"]:
            raise RuntimeError(f"checksum mismatch for {destination}")
        print(f"verified {item['dataset']}: {actual}", flush=True)


if __name__ == "__main__":
    main()
