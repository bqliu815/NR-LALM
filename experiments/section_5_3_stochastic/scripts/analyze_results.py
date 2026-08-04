#!/usr/bin/env python3
"""Validate complete Section 5.3 runs and aggregate the plotted KKT metric."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


METHODS = ("NR-LALM", "NR-LALM+SOC", "MLALM", "S-SQP")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if tuple(config["displayed_methods"]) != METHODS:
        raise ValueError("unexpected method order")
    repeats = int(config["oracle_repeats"])
    curve_points = int(config["curve_points"])
    config_hash = sha256_file(config_path)
    records: list[dict[str, object]] = []

    for dataset_index, dataset in enumerate(config["datasets"]):
        for repeat in range(repeats):
            for method_index, method in enumerate(METHODS):
                task_index = (
                    dataset_index * repeats + repeat
                ) * len(METHODS) + method_index
                path = args.raw_dir / f"task_{task_index:04d}.json"
                if not path.is_file():
                    raise FileNotFoundError(path)
                record = json.loads(path.read_text(encoding="utf-8"))
                task = record.get("task", {})
                expected = (task_index, dataset["name"], repeat, method)
                actual = (
                    int(task.get("task_index", -1)),
                    task.get("dataset_name"),
                    int(task.get("repeat", -1)),
                    task.get("method"),
                )
                if actual != expected:
                    raise ValueError(f"wrong task identity in {path}: {actual}")
                if record.get("config_sha256") != config_hash:
                    raise ValueError(f"configuration mismatch in {path}")
                output = record.get("output", {})
                if output.get("status") != "completed":
                    raise RuntimeError(f"incomplete run in {path}: {output.get('status')}")
                trace = output.get("trace", [])
                if len(trace) != curve_points:
                    raise ValueError(f"wrong trace length in {path}")
                values = np.asarray(
                    [row["optimized_pair_residual_sq"] for row in trace],
                    dtype=float,
                )
                calls = np.asarray(
                    [row["component_calls"] for row in trace], dtype=float
                )
                if (
                    np.any(values <= 0.0)
                    or not np.all(np.isfinite(values))
                    or calls[0] != 0.0
                    or np.any(np.diff(calls) <= 0.0)
                ):
                    raise ValueError(f"invalid KKT trace in {path}")
                if not math.isfinite(float(output["endpoint"]["optimized_pair_residual_sq"])):
                    raise ValueError(f"nonfinite endpoint in {path}")
                records.append(record)

    rows: list[dict[str, object]] = []
    for dataset in config["datasets"]:
        dataset_name = str(dataset["name"])
        for method in METHODS:
            group = [
                record
                for record in records
                if record["task"]["dataset_name"] == dataset_name
                and record["task"]["method"] == method
            ]
            if len(group) != repeats:
                raise ValueError(f"wrong replicate count for {dataset_name}/{method}")
            call_grids = np.asarray(
                [[row["component_calls"] for row in record["output"]["trace"]] for record in group],
                dtype=float,
            )
            if not np.all(call_grids == call_grids[0]):
                raise ValueError(f"call-grid mismatch for {dataset_name}/{method}")
            values = np.asarray(
                [[row["optimized_pair_residual_sq"] for row in record["output"]["trace"]] for record in group],
                dtype=float,
            )
            means = np.mean(values, axis=0)
            for checkpoint, (calls, mean) in enumerate(zip(call_grids[0], means)):
                rows.append(
                    {
                        "dataset": dataset_name,
                        "method": method,
                        "checkpoint": checkpoint,
                        "runs": repeats,
                        "component_calls": int(calls),
                        "optimized_pair_residual_sq_mean": float(mean),
                    }
                )

    args.output_dir.mkdir(parents=True, exist_ok=False)
    curve_path = args.output_dir / "curves.csv"
    with curve_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "status": "PASS",
        "datasets": [item["name"] for item in config["datasets"]],
        "methods": list(METHODS),
        "repeats": repeats,
        "curve_points": curve_points,
        "raw_records": len(records),
        "aggregation": config["display"]["aggregation"],
        "plotted_metric": "optimized_pair_residual_sq",
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
