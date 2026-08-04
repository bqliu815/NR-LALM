#!/usr/bin/env python3
"""Sequential local launcher for all 15 Section 5.2 data sets."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    for dataset_index in range(15):
        command = [
            sys.executable,
            str(EXPERIMENT_DIR / "scripts" / "run_stage_b_dataset.py"),
            "--dataset-index",
            str(dataset_index),
            "--timeout",
            str(args.timeout),
        ]
        if args.overwrite:
            command.append("--overwrite")
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
