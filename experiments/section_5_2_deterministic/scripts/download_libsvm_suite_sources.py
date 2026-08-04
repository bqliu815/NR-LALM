#!/usr/bin/env python3
"""Download every frozen LIBSVM-suite source without deriving samples."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from prepare_libsvm_suite_case import download, sha256


def download_case(
    package_root: Path, case: dict[str, Any]
) -> dict[str, Any]:
    target = (
        package_root
        / "data"
        / "libsvm_binary_suite_v1"
        / "source"
        / str(case["source_filename"])
    )
    download(str(case["source_url"]), target)
    return {
        "case_index": int(case["case_index"]),
        "case_id": str(case["case_id"]),
        "source_filename": str(case["source_filename"]),
        "source_url": str(case["source_url"]),
        "source_path": str(target.resolve()),
        "source_bytes": target.stat().st_size,
        "source_sha256": sha256(target),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--package-root", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    manifest_path = args.manifest.resolve()
    package_root = args.package_root.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = list(manifest["cases"])
    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(download_case, package_root, case): case
            for case in cases
        }
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)
    records.sort(key=lambda record: int(record["case_index"]))
    report = {
        "schema": "libsvm_binary_suite_source_downloads_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(manifest_path),
        "downloaded_cases": len(records),
        "total_source_bytes": sum(
            int(record["source_bytes"]) for record in records
        ),
        "records": records,
    }
    report_path = args.report.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "report": str(report_path),
                "downloaded_cases": len(records),
                "total_source_bytes": report["total_source_bytes"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
