#!/usr/bin/env python3
"""Validate the public source package and its paper configurations."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tomllib


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".cff", ".csv", ".json", ".md", ".py", ".sbatch", ".toml", ".txt"}
GENERATED_SUFFIXES = {
    ".csv",
    ".jpeg",
    ".jpg",
    ".jsonl",
    ".log",
    ".npy",
    ".npz",
    ".out",
    ".pdf",
    ".pickle",
    ".pkl",
    ".png",
    ".svg",
    ".tsv",
}
FORBIDDEN_NAMES = {"RELEASE_CHECKLIST.md"}
INTERNAL_PATH_WORDS = {
    "audit",
    "development",
    "heldout",
    "provenance",
    "qualification",
    "screening",
    "tuning",
}
FORBIDDEN_TEXT = (
    "/Users/",
    "paper——SOCLALM",
    "formal_experiments/",
    "numerical_experiments/",
    "NV_H100",
    "TO_BE_FILLED_BEFORE_LAUNCH",
    "authors' development archive",
    "failed development routes",
    "first 3 streams",
)


def release_paths() -> list[Path]:
    completed = subprocess.run(
        [
            "git",
            "-C",
            str(ROOT),
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
        ],
        check=True,
        capture_output=True,
    )
    return [
        ROOT / item.decode("utf-8")
        for item in completed.stdout.split(b"\0")
        if item
    ]


def main() -> None:
    issues: list[str] = []
    for path in release_paths():
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        relative_words = {
            word.lower()
            for part in relative.parts
            for word in part.replace("-", "_").split("_")
        }
        if path.name in FORBIDDEN_NAMES:
            issues.append(f"internal release file: {relative}")
        if relative_words & INTERNAL_PATH_WORDS:
            issues.append(f"internal development path: {relative}")
        if path.suffix.lower() in GENERATED_SUFFIXES:
            issues.append(f"generated artifact: {relative}")
        if path.stat().st_size > 5 * 1024 * 1024:
            issues.append(f"large file: {relative}")
        if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            issues.append(f"generated Python file: {relative}")
        if "data" in path.parts and path.name != "README.md":
            issues.append(f"data payload: {relative}")
        if "results" in path.parts and path.name != "README.md":
            issues.append(f"result payload: {relative}")
        if (
            relative != Path("scripts/validate_release.py")
            and (path.suffix in TEXT_SUFFIXES or path.name in {"LICENSE", ".gitignore"})
        ):
            text = path.read_text(encoding="utf-8")
            for token in FORBIDDEN_TEXT:
                if token in text:
                    issues.append(f"machine/internal reference {token!r}: {relative}")
        if path.suffix == ".json":
            json.loads(path.read_text(encoding="utf-8"))
        if path.suffix == ".toml":
            tomllib.loads(path.read_text(encoding="utf-8"))
    if issues:
        raise SystemExit("release validation failed:\n- " + "\n- ".join(issues))
    print("release validation: PASS")


if __name__ == "__main__":
    main()
