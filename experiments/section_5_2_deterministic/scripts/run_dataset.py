#!/usr/bin/env python3
"""Run all eight balanced orders for one Section 5.2 data set."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-index", required=True, type=int)
    parser.add_argument(
        "--config",
        type=Path,
        default=EXPERIMENT_DIR / "configs" / "paper_benchmark.json",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=EXPERIMENT_DIR / "results" / "paper_run",
    )
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    if not 0 <= args.dataset_index < len(config["datasets"]):
        raise ValueError("dataset index is outside the configured range")
    output_dir = args.results_root / f"dataset_{args.dataset_index}"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    environment = dict(os.environ)
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "PYTHONHASHSEED": "0",
        }
    )
    for order_index, order in enumerate(config["timing_method_orders"]):
        for position, method in enumerate(order):
            output = output_dir / (
                f"order_{order_index}_position_{position}_{method}_raw.json"
            )
            if output.exists() and not args.overwrite:
                status = 0
            else:
                command = [
                    sys.executable,
                    str(EXPERIMENT_DIR / "scripts" / "run_experiment.py"),
                    "--config",
                    str(args.config),
                    "--dataset-index",
                    str(args.dataset_index),
                    "--method",
                    str(method),
                    "--output",
                    str(output),
                ]
                try:
                    status = subprocess.run(
                        command,
                        env=environment,
                        timeout=args.timeout,
                        check=False,
                    ).returncode
                except subprocess.TimeoutExpired:
                    status = 124
            rows.append(
                {
                    "order_index": order_index,
                    "position": position,
                    "method": method,
                    "exit_status": status,
                }
            )
    with (output_dir / "process_status.tsv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream,
            delimiter="\t",
            fieldnames=["order_index", "position", "method", "exit_status"],
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
