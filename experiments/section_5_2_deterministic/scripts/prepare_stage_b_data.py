#!/usr/bin/env python3
"""Download and deterministically prepare the 15 Section 5.2 data sets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=EXPERIMENT_DIR / "configs" / "paper_stage_b_v2.json",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=EXPERIMENT_DIR / "configs" / "paper_data_manifest.json",
    )
    parser.add_argument(
        "--dataset-index",
        type=int,
        action="append",
        help="prepare only this data-set index; may be repeated",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    case_positions = {
        str(case["case_id"]): position
        for position, case in enumerate(manifest["cases"])
    }
    selected = (
        list(range(len(config["datasets"])))
        if args.dataset_index is None
        else args.dataset_index
    )
    for dataset_index in selected:
        dataset = config["datasets"][dataset_index]
        case_index = case_positions[str(dataset["name"])]
        command = [
            sys.executable,
            str(EXPERIMENT_DIR / "scripts" / "prepare_libsvm_suite_case.py"),
            "--manifest",
            str(args.manifest),
            "--case-index",
            str(case_index),
            "--package-root",
            str(EXPERIMENT_DIR),
            "--download",
        ]
        if args.overwrite:
            command.append("--overwrite")
        print(
            f"preparing data set {dataset_index}: {dataset['name']}",
            flush=True,
        )
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
