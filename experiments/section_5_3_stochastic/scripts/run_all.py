#!/usr/bin/env python3
"""Run all 20 Section 5.3 replicates, aggregate them, and render the figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=EXPERIMENT_DIR / "configs" / "paper_v1.json")
    parser.add_argument("--output-root", type=Path, default=EXPERIMENT_DIR / "results" / "paper_run")
    parser.add_argument("--skip-runs", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    raw_dir = args.output_root / "raw"
    analysis_dir = args.output_root / "analysis"
    figure_dir = args.output_root / "figure"
    raw_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_runs:
        tasks = len(config["datasets"]) * int(config["oracle_repeats"])
        for index in range(tasks):
            subprocess.run(
                [
                    sys.executable,
                    str(EXPERIMENT_DIR / "scripts" / "run_repeat.py"),
                    "--config",
                    str(args.config),
                    "--output-dir",
                    str(raw_dir),
                    "--array-index",
                    str(index),
                ],
                check=True,
            )
    subprocess.run(
        [
            sys.executable,
            str(EXPERIMENT_DIR / "scripts" / "analyze_results.py"),
            "--config",
            str(args.config),
            "--raw-dir",
            str(raw_dir),
            "--output-dir",
            str(analysis_dir),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(EXPERIMENT_DIR / "scripts" / "render_figure.py"),
            "--config",
            str(args.config),
            "--analysis-dir",
            str(analysis_dir),
            "--output-dir",
            str(figure_dir),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
